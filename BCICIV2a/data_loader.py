import numpy as np

def load_moabb_data(subject_id=[]):
    """
    使用 MOABB 库加载完整的 BCI Competition IV 2a 数据集。
    不再需要手动去读取 GDF 或 MAT 文件，MOABB 会自动下载或读取本地缓存，并严格划分出 Train/Test 试次。
    这里直接把连续信号和 MNE 转换为我们模型所用的 X(n_trials, n_channels, n_samples) 以及对应的 y 标签。
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as e:
        print(f"导入 MOABB 失败: {e}")
        print("请尝试运行: pip install moabb")
        return None, None, None, None, None
        
    try:
        dataset = BNCI2014_001()
        dataset.subject_list = subject_id
        
        # paradigm 是自动按时间窗口(0.5s~2.5s)分段(epoching)、只保留选定类别的统一调度器
        # 这里声明 BCI 2a 的四个类别以及对应的提取窗口
        paradigm = MotorImagery(n_classes=4, tmin=0.5, tmax=2.5, channels=None)
        
        # 自动解析下载得到的数据
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    except Exception as e:
        print(f"提取数据失败: {e}")
        return None, None, None, None, None
    
    # moabb 加载出来的 y 是带名字的字符串如 ['left_hand', 'right_hand']等，转为我们的数字索引 [1,2,3,4]
    label_dict = {'left_hand': 1, 'right_hand': 2, 'feet': 3, 'tongue': 4}
    y_int = np.array([label_dict[label] for label in y])

    # 提取评估数据或用于测试的标识
    # moabb 会通过 metadata 把原训练集（'0train'）和原用于评估预测的（'1test'）完整区分。
    is_train = metadata['session'].str.contains('train').values
    is_test = metadata['session'].str.contains('test').values
    
    X_train, y_train = X[is_train], y_int[is_train]
    X_test, y_test = X[is_test], y_int[is_test]
    
    # scipy 或 sklearn 训练通常接受维度顺序和直接拿出来的一致
    return X_train, X_test, y_train, y_test, 250

if __name__=="__main__":    

    print("\n=== 使用 moabb 官方开源库载入数据 ===")
    X_tr, X_te, y_tr, y_te, sfreq_moabb = load_moabb_data(subject_id=1)
    if X_tr is not None:
        print(f"MOABB 提取 (Subject 1) -> 训练集 Session_T: X={X_tr.shape}, y={y_tr.shape}")
        print(f"MOABB 提取 (Subject 1) -> 测试集 Session_E: X={X_te.shape}, y={y_te.shape}")