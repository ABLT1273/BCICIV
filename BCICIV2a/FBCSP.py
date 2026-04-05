import joblib
import numpy as np
import mne
mne.set_log_level('WARNING') # 屏蔽 CSP 计算时的冗长提示信息
from scipy.signal import cheby2, sosfiltfilt
from mne.decoding import CSP
from sklearn.feature_selection import  mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import norm

class FilterBank(BaseEstimator, TransformerMixin):
    """
    第一阶段：Filter Bank (频带划分)
    使用 9 个互不重叠的切比雪夫 II 型带通滤波器 [cite: 50]
    频率范围: 4-8 Hz, 8-12 Hz, ..., 36-40 Hz [cite: 51]
    """
    def __init__(self, sfreq, bands=[(4,8), (8,12), (12,16), (16,20), (20,24), (24,28), (28,32), (32,36), (36,40)]):
        self.sfreq = sfreq
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X 形状应为 (n_trials, n_channels, n_samples)
        X_filtered = []
        for low, high in self.bands:
            # 设计切比雪夫 II 型滤波器，阻带衰减设为 40 dB [cite: 50]
            sos = cheby2(N=4, rs=40, Wn=[low, high], btype='bandpass', fs=self.sfreq, output='sos')
            # 零相位滤波避免相移
            filtered_band = sosfiltfilt(sos, X, axis=-1)
            X_filtered.append(filtered_band)
        # 返回形状: (n_bands, n_trials, n_channels, n_samples)
        return np.array(X_filtered)

class NBPWClassifier(BaseEstimator, ClassifierMixin):
    """
    朴素贝叶斯 Parzen 窗 (NBPW) 分类器。
    严格按照论文公式 (16) 到 (22) 实现。
    """
    def __init__(self):
        self.classes_ = None
        self.prior_ = {}
        self.train_data_ = {}
        self.h_opt_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = len(y)
        
        for c in self.classes_:
            X_c = X[y == c]
            n_c = len(X_c)
            # 计算先验概率 P(w)
            self.prior_[c] = n_c / n_samples
            self.train_data_[c] = X_c
            
            # 计算正态最优平滑参数 h_opt, 对应论文公式 (21)
            # h_opt = (4 / 3n)^(1/5) * sigma
            sigma = np.std(X_c, axis=0)
            # 避免标准差为0的情况
            sigma[sigma == 0] = 1e-6 
            self.h_opt_[c] = ((4.0 / (3.0 * n_c)) ** 0.2) * sigma
            
        return self

    def predict_proba(self, X):
        n_samples, n_features = X.shape
        probas = np.zeros((n_samples, len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            X_c = self.train_data_[c]
            h_c = self.h_opt_[c]
            n_c = len(X_c)
            
            # 计算 p(x|w), 对应论文公式 (18) 和 (19)
            # 假设特征条件独立，用 Parzen 窗估计边缘概率密度
            p_x_given_w = np.ones(n_samples)
            for j in range(n_features):
                # 提取测试样本的第 j 个特征，形状 (n_samples, 1)
                x_j = X[:, j][:, np.newaxis] 
                # 提取训练样本的第 j 个特征，形状 (1, n_c)
                x_train_j = X_c[:, j][np.newaxis, :] 
                
                # 计算高斯核，对应论文公式 (20)
                diff = x_j - x_train_j
                kernel_vals = norm.pdf(diff, loc=0, scale=h_c[j])
                
                # 对所有训练样本求平均得到概率密度估计
                p_xj_given_w = np.sum(kernel_vals, axis=1) / n_c
                p_x_given_w *= p_xj_given_w
            
            # 乘以先验概率，得到未归一化的后验概率
            probas[:, i] = p_x_given_w * self.prior_[c]
            
        # 归一化 (对应公式 16)
        row_sums = probas.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1e-10 # 防止除零
        return probas / row_sums

    def predict(self, X):
        probas = self.predict_proba(X)
        # 对应公式 (22): w = argmax p(w|x)
        return self.classes_[np.argmax(probas, axis=1)]

class PairedMIBIF(BaseEstimator, TransformerMixin):
    """
    基于互信息的最优个体特征选择 (MIBIF)，包含“成对保留”机制。
    论文指出：如果一个特征被选中，其对应的成对特征如果未被选中也会被包含进来。
    """
    def __init__(self, k=4, m=2, n_bands=9):
        self.k = k
        self.m = m # 每个频带提取 2*m 个特征 (对于 2a 数据集 m=2)
        self.n_bands = n_bands
        self.selected_indices_ = []

    def fit(self, X, y):
        # 计算所有特征的互信息 (MI)
        mi_scores = mutual_info_classif(X, y)
        
        # 降序排序，选取前 k 个特征的索引
        top_k_indices = np.argsort(mi_scores)[::-1][:self.k]
        
        final_indices = set(top_k_indices)
        
        # 强制包含成对特征
        # 在 mne 的 CSP 中，n_components=4 时，0和3是一对，1和2是一对。
        for idx in top_k_indices:
            band_idx = idx // (2 * self.m)
            intra_band_idx = idx % (2 * self.m)
            
            # 计算其成对特征的相对索引: (2*m - 1) - intra_band_idx
            pair_intra_idx = (2 * self.m - 1) - intra_band_idx
            pair_idx = band_idx * (2 * self.m) + pair_intra_idx
            
            final_indices.add(pair_idx)
            
        self.selected_indices_ = sorted(list(final_indices))
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

class BinaryFBCSP_Pipeline:
    """
    单个二分类的 FBCSP 流水线 (用于 OVR 架构中的子模块)
    """
    def __init__(self, m=2, k=4):
        self.m = m
        self.csps = []
        self.selector = PairedMIBIF(k=k, m=m)
        self.clf = NBPWClassifier()
        
    def fit(self, X_fb, y_binary):
        n_bands = X_fb.shape[0]
        self.csps = []
        features = []
        
        # 1. 独立计算每个频带的 CSP
        for i in range(n_bands):
            csp = CSP(n_components=2*self.m, reg=None, log=True, norm_trace=False)
            csp.fit(X_fb[i], y_binary)
            self.csps.append(csp)
            features.append(csp.transform(X_fb[i]))
            
        X_csp = np.concatenate(features, axis=1)
        
        # 2. 特征选择与分类器训练
        X_selected = self.selector.fit_transform(X_csp, y_binary)
        self.clf.fit(X_selected, y_binary)
        return self

    def predict_proba(self, X_fb):
        n_bands = X_fb.shape[0]
        features = []
        for i in range(n_bands):
            features.append(self.csps[i].transform(X_fb[i]))
        X_csp = np.concatenate(features, axis=1)
        X_selected = self.selector.transform(X_csp)
        
        # 返回正类 (当前类) 的概率
        # 假设 y_binary 中 1 为正类，0 为负类 (Rest)
        idx_positive = np.where(self.clf.classes_ == 1)[0][0]
        return self.clf.predict_proba(X_selected)[:, idx_positive]

class OVR_FBCSP_Ensemble:
    """
    严格按照论文范式的 One-Versus-Rest 多分类扩展。
    为 4 个类别分别训练完全独立的 FBCSP 提取器和 NBPW 分类器。
    """
    def __init__(self, classes=[1, 2, 3, 4], m=2, k=4):
        self.classes = classes
        self.models = {}
        for c in classes:
            self.models[c] = BinaryFBCSP_Pipeline(m=m, k=k)
            
    def fit(self, X_fb, y):
        for c in self.classes:
            # 构造二分类标签：当前类为 1，其余为 0
            y_binary = np.where(y == c, 1, 0)
            self.models[c].fit(X_fb, y_binary)
        return self
        
    def predict(self, X_fb):
        n_samples = X_fb.shape[1]
        probas = np.zeros((n_samples, len(self.classes)))
        
        # 收集每个 OVR 二分类器的正类概率
        for i, c in enumerate(self.classes):
            probas[:, i] = self.models[c].predict_proba(X_fb)
            
        # 对应论文公式 (25): w = argmax p_OVR(w|x)
        best_class_indices = np.argmax(probas, axis=1)
        return np.array([self.classes[idx] for idx in best_class_indices])

# --- 模拟运行示例 ---
if __name__ == "__main__":
    from framework.runtime import prepare_runtime_environment

    prepare_runtime_environment()

    try:
        from framework.data import load_subject_train_test
    except ImportError as exc:
        raise ImportError("无法从统一框架读取 BCICIV2a 数据。") from exc

    print(f"=== 1. 使用 MOABB 库加载规范并带有真实验证标签的 BCI 2A 数据 ===")

    X_train, X_test, y_train, y_test, sfreq = load_subject_train_test(subject_id=1)
    
    if X_train is not None and X_test is not None:
        print(f"数据加载完成! 训练集(Session T)维度: {X_train.shape}, 测试/评估集(Session E)维度: {X_test.shape}")
        
        print("\n=== 2. 开始构建并训练 FBCSP (OVR+MIBIF+NBPW) 模型 ===")
        fb = FilterBank(sfreq=int(sfreq))
        
        # 1. 初始化频带划分模块并处理数据
        # 所有类别的二分类器共用同一组频带数据，所以在外部先处理好以节省算力
        X_train_fb = fb.transform(X_train)
        X_test_fb = fb.transform(X_test)
        
        # 2. 初始化OVR模型集合
        # 内部自动包含4组（CSP->PairedMIBIF->NBPW）
        ovr_model=OVR_FBCSP_Ensemble(classes=[1,2,3,4],m=2,k=4)

        # ---训练---
        ovr_model.fit(X_train_fb,y_train)
        print("模型训练完成")

        # ---测试---
        print("\n=== 3. 开始对官方测试集(E)进行预测及评估 ===")

        y_pred = ovr_model.predict(X_test_fb)
        acc = accuracy_score(y_test, y_pred)
        print(f"测试集准确率 (Accuracy): {acc * 100:.2f}%\n")
        
        print("详细分类评估报告:")
        print(classification_report(y_test, y_pred, target_names=['左手(1)', '右手(2)', '双足(3)', '舌头(4)']))

        # === 保存机制 ===
        model_pipeline = {
            'filter_bank': fb,
            'ovr_ensemble': ovr_model,
            'sfreq': sfreq
        }
        save_path = "fbcsp_pretrained_moabb_A01.pkl"
        print(f"=== 将训练好的模型保存到 {save_path} ===")
        joblib.dump(model_pipeline, save_path)
        print("模型保存成功！后续测试时可通过: `joblib.load('fbcsp_pretrained_moabb_A01.pkl')` 恢复。")

        print("\n=== 4. 计算 10x10-fold Cross-Validation (用 Kappa 评估) ===")
        # 使用 RepeatedStratifiedKFold 进行 10次 10折 分层交叉验证 (共 100 折)
        n_splits = 10
        n_repeats = 10
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        
        kappa_scores = []
        print(f"开始执行 {n_splits} 折 {n_repeats} 次重复的交叉验证 (共 {n_splits * n_repeats} 个 Folds)...这需要一定时间计算。")
        
        # 预先进行 FilterBank 以节省时间，这里我们依然仅对训练集 Session T 进行 CV
        for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
            # X_train_fb 维度: (n_bands, n_trials, n_channels, n_samples)
            X_fold_train = X_train_fb[:, train_idx, :, :]
            y_fold_train = y_train[train_idx]
            
            X_fold_val = X_train_fb[:, val_idx, :, :]
            y_fold_val = y_train[val_idx]
            
            # 使用相同参数初始化模型
            cv_model = OVR_FBCSP_Ensemble(classes=[1,2,3,4], m=2, k=4)
            cv_model.fit(X_fold_train, y_fold_train)
            
            y_fold_pred = cv_model.predict(X_fold_val)
            kappa = cohen_kappa_score(y_fold_val, y_fold_pred)
            kappa_scores.append(kappa)
            
            if (fold_idx + 1) % 10 == 0:
                print(f"已完成 {fold_idx + 1} / {n_splits * n_repeats} 折计算...")
                
        kappa_scores = np.array(kappa_scores)
        print("\n--- 10×10-Fold CV 验证结果汇总 (Session T) ---")
        print(f"最大 Kappa 值 (Maximum Kappa): {np.max(kappa_scores):.4f}")
        print(f"平均 Kappa 值 (Mean Kappa): {np.mean(kappa_scores):.4f}")
        print(f"最小 Kappa 值 (Minimum Kappa): {np.min(kappa_scores):.4f}")
        print(f"标准差 (Std. Deviation): {np.std(kappa_scores):.4f}")

    else:
        print(f"未能获取 MOABB 资源，或者载入出错，退出运行。")
