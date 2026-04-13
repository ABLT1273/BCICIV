import sys
from pathlib import Path

# 确保 BCICIV2a 根目录在 sys.path，无论是直接运行还是作为包导入都可以找到 models.FBCSP
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import mne
mne.set_log_level('WARNING')
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import RepeatedStratifiedKFold

# 复用 FBCSP 中已实现的组件
from models.FBCSP import FilterBank, NBPWClassifier, PairedMIBIF


class DiscriminativeBandSelector(BaseEstimator, TransformerMixin):
    """
    判别式频带选择器 (Discriminative Band Selector)。

    对每个频带计算 Fisher 判别分数：在该频带上，用各通道对数方差特征的
    类间距与类内方差比值之和来衡量频带判别性，选出得分最高的 n_select 个频带。

    与 FBCSP 不同，DFBCSP 不对全部 9 个频带做 CSP，而是先筛掉低判别性的频带，
    降低计算量并减少噪声频带干扰。
    """

    def __init__(self, n_select: int = 4):
        self.n_select = n_select
        self.selected_bands_: list[int] = []
        self.band_scores_: np.ndarray | None = None

    def _fisher_score_for_band(self, X_band: np.ndarray, y_binary: np.ndarray) -> float:
        """
        计算单个频带的平均 Fisher 判别分数。

        Parameters
        ----------
        X_band : (n_trials, n_channels, n_samples)
        y_binary : (n_trials,)  —— 0 / 1 二分类标签

        Returns
        -------
        float : 所有通道 Fisher 分数的平均值
        """
        eps = 1e-10
        # 对每个 trial 的每个通道计算对数方差，shape: (n_trials, n_channels)
        log_var = np.log(np.var(X_band, axis=-1) + eps)

        mask1 = y_binary == 1
        mask0 = y_binary == 0

        if mask1.sum() < 2 or mask0.sum() < 2:
            return 0.0

        mu1 = np.mean(log_var[mask1], axis=0)   # (n_channels,)
        mu0 = np.mean(log_var[mask0], axis=0)
        s1  = np.var(log_var[mask1], axis=0)    # 类内方差
        s0  = np.var(log_var[mask0], axis=0)

        denom = s1 + s0
        denom[denom == 0] = eps

        fisher_per_channel = (mu1 - mu0) ** 2 / denom
        return float(np.mean(fisher_per_channel))

    def fit(self, X_fb: np.ndarray, y_binary: np.ndarray):
        """
        Parameters
        ----------
        X_fb : (n_bands, n_trials, n_channels, n_samples)
        y_binary : (n_trials,)
        """
        n_bands = X_fb.shape[0]
        scores = np.array(
            [self._fisher_score_for_band(X_fb[b], y_binary) for b in range(n_bands)]
        )
        self.band_scores_ = scores
        # 取分数最高的 n_select 个频带，保持频率升序排列
        top_idx = np.argsort(scores)[::-1][: self.n_select]
        self.selected_bands_ = sorted(top_idx.tolist())
        return self

    def transform(self, X_fb: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        (n_select, n_trials, n_channels, n_samples)
        """
        return X_fb[self.selected_bands_]


class BinaryDFBCSP_Pipeline:
    """
    单个二分类的 DFBCSP 流水线。

    流程：FilterBank（外部） → DiscriminativeBandSelector → CSP (per selected band)
          → PairedMIBIF → NBPW

    与 BinaryFBCSP_Pipeline 的唯一区别：在 CSP 之前增加了判别式频带筛选。
    """

    def __init__(self, m: int = 2, k: int = 4, n_select: int = 4):
        self.m = m
        self.k = k
        self.n_select = n_select
        self.band_selector = DiscriminativeBandSelector(n_select=n_select)
        self.csps: list = []
        self.selector = PairedMIBIF(k=k, m=m)
        self.clf = NBPWClassifier()

    def fit(self, X_fb: np.ndarray, y_binary: np.ndarray):
        """
        Parameters
        ----------
        X_fb : (n_bands, n_trials, n_channels, n_samples)
        y_binary : (n_trials,)  —— 0/1
        """
        # 1. 判别式频带选择
        self.band_selector.fit(X_fb, y_binary)
        X_sel = self.band_selector.transform(X_fb)   # (n_select, ...)

        n_sel = X_sel.shape[0]
        self.csps = []
        features = []

        # 2. 对每个选中频带独立拟合 CSP
        for i in range(n_sel):
            csp = CSP(n_components=2 * self.m, reg=None, log=True, norm_trace=False)
            csp.fit(X_sel[i], y_binary)
            self.csps.append(csp)
            features.append(csp.transform(X_sel[i]))

        X_csp = np.concatenate(features, axis=1)

        # 3. MIBIF 特征选择 + NBPW 分类器
        self.selector = PairedMIBIF(k=self.k, m=self.m, n_bands=n_sel)
        X_feat = self.selector.fit_transform(X_csp, y_binary)
        self.clf.fit(X_feat, y_binary)
        return self

    def predict_proba(self, X_fb: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        (n_trials,)  —— 正类 (label=1) 的后验概率
        """
        X_sel = self.band_selector.transform(X_fb)
        features = [self.csps[i].transform(X_sel[i]) for i in range(len(self.csps))]
        X_csp = np.concatenate(features, axis=1)
        X_feat = self.selector.transform(X_csp)

        idx_pos = np.where(self.clf.classes_ == 1)[0][0]
        return self.clf.predict_proba(X_feat)[:, idx_pos]


class OVR_DFBCSP_Ensemble:
    """
    判别式频带选择的 One-Versus-Rest 多分类 DFBCSP。

    每个 OVR 子分类器独立完成：频带筛选 → CSP → MIBIF → NBPW。
    不同类别的 OVR 子问题可能选出不同的最优频带子集。
    """

    def __init__(self, classes: list = [1, 2, 3, 4], m: int = 2, k: int = 4, n_select: int = 4):
        self.classes = classes
        self.models: dict = {}
        for c in classes:
            self.models[c] = BinaryDFBCSP_Pipeline(m=m, k=k, n_select=n_select)

    def fit(self, X_fb: np.ndarray, y: np.ndarray):
        for c in self.classes:
            y_binary = np.where(y == c, 1, 0)
            self.models[c].fit(X_fb, y_binary)
        return self

    def predict(self, X_fb: np.ndarray) -> np.ndarray:
        n_samples = X_fb.shape[1]
        probas = np.zeros((n_samples, len(self.classes)))

        for i, c in enumerate(self.classes):
            probas[:, i] = self.models[c].predict_proba(X_fb)

        best_idx = np.argmax(probas, axis=1)
        return np.array([self.classes[idx] for idx in best_idx])


# ---------------------------------------------------------------------------
# 独立运行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # _ROOT 已在模块顶部加入 sys.path，framework.* 此处可直接导入
    from framework.runtime import prepare_runtime_environment

    prepare_runtime_environment()

    try:
        from framework.data import load_subject_train_test
    except ImportError as exc:
        raise ImportError("无法从统一框架读取 BCICIV2a 数据。") from exc

    from framework.constants import LABEL_TO_INT
    from framework.paths import get_model_dir

    print("=== 1. 使用 MOABB 库加载 BCI 2A 数据 ===")
    X_train, X_test, y_train, y_test, sfreq = load_subject_train_test(subject_id=1)

    if X_train is None or X_test is None:
        print("未能获取 MOABB 资源，退出。")
        raise SystemExit(1)

    # MOABB 返回字符串标签，转换为整数 (left_hand→1, right_hand→2, feet→3, tongue→4)
    y_train = np.array([LABEL_TO_INT[lbl] for lbl in y_train])
    y_test  = np.array([LABEL_TO_INT[lbl] for lbl in y_test])

    print(f"训练集(Session T): {X_train.shape}，测试集(Session E): {X_test.shape}")

    print("\n=== 2. 构建并训练 DFBCSP (OVR + MIBIF + NBPW, n_select=4) ===")
    fb = FilterBank(sfreq=int(sfreq))

    X_train_fb = fb.transform(X_train)   # (9, n_train, n_ch, n_samples)
    X_test_fb  = fb.transform(X_test)    # (9, n_test,  n_ch, n_samples)

    dfbcsp = OVR_DFBCSP_Ensemble(classes=[1, 2, 3, 4], m=2, k=4, n_select=4)
    dfbcsp.fit(X_train_fb, y_train)
    print("模型训练完成")

    # 打印各 OVR 子分类器选出的频带
    bands_label = [(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32),(32,36),(36,40)]
    print("\n各 OVR 子分类器选中的频带：")
    for c in [1, 2, 3, 4]:
        sel = dfbcsp.models[c].band_selector.selected_bands_
        sel_ranges = [bands_label[i] for i in sel]
        scores = dfbcsp.models[c].band_selector.band_scores_
        print(f"  类别 {c}: {sel_ranges}  (Fisher scores: {[f'{scores[i]:.4f}' for i in sel]})")

    print("\n=== 3. 对官方测试集(Session E)进行预测及评估 ===")
    y_pred = dfbcsp.predict(X_test_fb)
    acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率 (Accuracy): {acc * 100:.2f}%\n")
    print("详细分类评估报告:")
    print(classification_report(y_test, y_pred,
                                target_names=['左手(1)', '右手(2)', '双足(3)', '舌头(4)']))

    # === 保存模型 ===
    model_pipeline = {'filter_bank': fb, 'ovr_ensemble': dfbcsp, 'sfreq': sfreq}
    save_path = get_model_dir() / "dfbcsp_pretrained_moabb_A01.pkl"
    print(f"=== 将训练好的模型保存到 {save_path} ===")
    joblib.dump(model_pipeline, save_path)
    print("模型保存成功！")

    print("\n=== 4. 计算 10×10-Fold Cross-Validation (Kappa) ===")
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    kappa_scores = []
    total_folds = 100
    print(f"开始执行 10 折 10 次重复 CV (共 {total_folds} 折)...")

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
        X_fold_train = X_train_fb[:, train_idx, :, :]
        y_fold_train = y_train[train_idx]
        X_fold_val   = X_train_fb[:, val_idx, :, :]
        y_fold_val   = y_train[val_idx]

        cv_model = OVR_DFBCSP_Ensemble(classes=[1, 2, 3, 4], m=2, k=4, n_select=4)
        cv_model.fit(X_fold_train, y_fold_train)
        y_fold_pred = cv_model.predict(X_fold_val)
        kappa_scores.append(cohen_kappa_score(y_fold_val, y_fold_pred))

        if (fold_idx + 1) % 10 == 0:
            print(f"  已完成 {fold_idx + 1} / {total_folds} 折...")

    kappa_scores = np.array(kappa_scores)
    print("\n--- 10×10-Fold CV 验证结果汇总 (Session T) ---")
    print(f"最大 Kappa: {np.max(kappa_scores):.4f}")
    print(f"平均 Kappa: {np.mean(kappa_scores):.4f}")
    print(f"最小 Kappa: {np.min(kappa_scores):.4f}")
    print(f"标准差:     {np.std(kappa_scores):.4f}")
