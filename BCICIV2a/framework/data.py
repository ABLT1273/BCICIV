from __future__ import annotations

import numpy as np

from .constants import BNCI2014001_CHANNEL_NAMES, LABEL_TO_DISPLAY_NAME


def load_subject_epochs(
    subject_id: int,
    tmin: float = 0.5,
    tmax: float = 2.5,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, "object", float]:
    """
    统一读取一个被试的 epoch 数据。

    这个接口同时服务于：
    - 混合特征降维脚本
    - TRCA / Wavelet / CNN 对比脚本
    """

    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise ImportError(
            "没有检测到 moabb，请先在当前虚拟环境里安装：\n"
            "test_newPyEnv/.venv/bin/pip install moabb"
        ) from exc

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        events=list(LABEL_TO_DISPLAY_NAME.keys()),
        n_classes=4,
        channels=channels,
        tmin=tmin,
        tmax=tmax,
    )

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    return X, np.asarray(y), metadata, 250.0


def load_subject_train_test(
    subject_id: int,
    tmin: float = 0.5,
    tmax: float = 2.5,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """按 Session T / Session E 读取一个被试的训练集和测试集。"""

    X, y, metadata, sfreq = load_subject_epochs(
        subject_id=subject_id,
        tmin=tmin,
        tmax=tmax,
        channels=channels,
    )

    is_train = metadata["session"].astype(str).str.contains("train").to_numpy()
    is_test = metadata["session"].astype(str).str.contains("test").to_numpy()
    return X[is_train], X[is_test], y[is_train], y[is_test], sfreq


def select_named_channels(X: np.ndarray, channel_names: list[str]) -> np.ndarray:
    indices = [BNCI2014001_CHANNEL_NAMES.index(name) for name in channel_names]
    return X[:, indices, :]


def select_c3_c4_from_full_channels(X: np.ndarray) -> np.ndarray:
    """从 BCICIV2a 的 22 通道数据里切出 C3 / C4。"""

    if X.ndim != 3 or X.shape[1] != len(BNCI2014001_CHANNEL_NAMES):
        raise ValueError(
            "当前数据形状与 BCICIV2a 的 22 通道设置不一致，"
            f"收到的形状是 {X.shape}。"
        )

    c3_index = BNCI2014001_CHANNEL_NAMES.index("C3")
    c4_index = BNCI2014001_CHANNEL_NAMES.index("C4")
    return X[:, [c3_index, c4_index], :]

