from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from framework.constants import INT_TO_LABEL, LABEL_TO_DISPLAY_NAME, LABEL_TO_INT
from framework.data import load_subject_epochs, select_c3_c4_from_full_channels
from framework.paths import get_hybrid_results_dir
from framework.plotting import plot_3d_embedding


FREQUENCY_BANDS = {
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


@dataclass
class HybridReductionConfig:
    subject_id: int = 1
    output_dir: Path | None = None
    show: bool = False
    supervised_umap: bool = False
    tmin: float = 0.5
    tmax: float = 2.5


def integrate_band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """对指定频带的功率谱做积分。"""

    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    if np.count_nonzero(band_mask) < 2:
        raise ValueError(
            f"频带 [{low_freq}, {high_freq}] Hz 内的频率点太少，无法稳定积分。"
        )
    return np.trapezoid(psd[..., band_mask], freqs[band_mask], axis=-1)


def extract_peak_frequency(
    psd: np.ndarray,
    freqs: np.ndarray,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """在指定频带内寻找峰值频率。"""

    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    band_psd = psd[..., band_mask]
    band_freqs = freqs[band_mask]
    peak_indices = np.argmax(band_psd, axis=-1)
    return band_freqs[peak_indices]


def extract_c3_c4_frequency_features(
    X: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, list[str]]:
    """
    从 C3 / C4 提取频域特征。

    每个频带提取：
    - C3 / C4 对数功率
    - C3 / C4 相对功率
    - 左右差值
    - 峰值频率
    """

    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(
            f"期望输入形状为 (n_trials, 2, n_samples)，实际收到 {X.shape}。"
        )

    nperseg = min(256, X.shape[-1])
    freqs, psd = welch(
        X,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
    )

    eps = 1e-12
    total_power = integrate_band_power(psd, freqs, 4.0, 40.0) + eps
    c3_index = 0
    c4_index = 1

    feature_columns: list[np.ndarray] = []
    feature_names: list[str] = []

    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        band_power = integrate_band_power(psd, freqs, low_freq, high_freq) + eps
        relative_power = band_power / total_power
        peak_frequency = extract_peak_frequency(psd, freqs, low_freq, high_freq)

        log_power = np.log10(band_power)
        log_power_diff = log_power[:, c3_index] - log_power[:, c4_index]
        relative_power_diff = (
            relative_power[:, c3_index] - relative_power[:, c4_index]
        )

        feature_columns.extend(
            [
                log_power[:, c3_index],
                log_power[:, c4_index],
                relative_power[:, c3_index],
                relative_power[:, c4_index],
                log_power_diff,
                relative_power_diff,
                peak_frequency[:, c3_index],
                peak_frequency[:, c4_index],
            ]
        )
        feature_names.extend(
            [
                f"{band_name}_log_power_C3",
                f"{band_name}_log_power_C4",
                f"{band_name}_relative_power_C3",
                f"{band_name}_relative_power_C4",
                f"{band_name}_log_power_C3_minus_C4",
                f"{band_name}_relative_power_C3_minus_C4",
                f"{band_name}_peak_freq_C3",
                f"{band_name}_peak_freq_C4",
            ]
        )

    return np.column_stack(feature_columns), feature_names


def extract_fbcsp_features(
    X: np.ndarray,
    labels_int: np.ndarray,
    sfreq: float,
    m: int = 2,
    k: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """
    使用现有 FBCSP 组件，把多通道 EEG 提取成监督式空间滤波特征。
    """

    try:
        from mne.decoding import CSP
        from models.FBCSP import FilterBank, PairedMIBIF
    except ImportError as exc:
        raise ImportError(
            "导入 FBCSP 相关模块失败，请确认 FBCSP.py 与依赖已经可用。"
        ) from exc

    filter_bank = FilterBank(sfreq=int(sfreq))
    X_fb = filter_bank.transform(X)

    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    n_bands = X_fb.shape[0]

    for class_id in sorted(np.unique(labels_int)):
        y_binary = (labels_int == class_id).astype(int)
        csp_feature_per_band: list[np.ndarray] = []

        for band_index in range(n_bands):
            csp = CSP(n_components=2 * m, reg=None, log=True, norm_trace=False)
            band_features = csp.fit_transform(X_fb[band_index], y_binary)
            csp_feature_per_band.append(band_features)

        X_csp = np.concatenate(csp_feature_per_band, axis=1)
        selector = PairedMIBIF(k=k, m=m, n_bands=n_bands)
        X_selected = selector.fit_transform(X_csp, y_binary)
        feature_blocks.append(X_selected)

        class_label = INT_TO_LABEL[int(class_id)]
        for selected_index in selector.selected_indices_:
            band_idx = selected_index // (2 * m)
            component_idx = selected_index % (2 * m)
            low_freq, high_freq = filter_bank.bands[band_idx]
            feature_names.append(
                f"fbcsp_ovr_{class_label}_{low_freq:02d}_{high_freq:02d}_csp_{component_idx}"
            )

    return np.concatenate(feature_blocks, axis=1), feature_names


def reduce_features_to_3d(
    features: np.ndarray,
    labels: np.ndarray,
    supervised: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """标准化后用 UMAP 压缩到 3 维。"""

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "没有检测到 umap-learn，请先安装：\n"
            "test_newPyEnv/.venv/bin/pip install umap-learn"
        ) from exc

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    reducer_kwargs = {
        "n_components": 3,
        "n_neighbors": min(30, max(5, len(features_scaled) - 1)),
        "min_dist": 0.15,
        "metric": "euclidean",
        "random_state": 42,
    }
    if supervised:
        reducer_kwargs["target_metric"] = "categorical"
        reducer_kwargs["target_weight"] = 0.35

    reducer = umap.UMAP(**reducer_kwargs)
    if supervised:
        encoded_labels = LabelEncoder().fit_transform(labels)
        embedding = reducer.fit_transform(features_scaled, y=encoded_labels)
    else:
        embedding = reducer.fit_transform(features_scaled)

    return embedding, features_scaled


def save_feature_package(
    output_dir: Path,
    subject_id: int,
    embedding_name: str,
    raw_features: np.ndarray,
    scaled_features: np.ndarray,
    embedding: np.ndarray,
    c3_c4_features: np.ndarray,
    fbcsp_features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    metadata: object,
) -> Path:
    """保存融合特征与降维结果，便于后续复用。"""

    output_path = output_dir / f"subject_{subject_id:02d}_hybrid_fbcsp_{embedding_name}.npz"
    sessions = (
        np.asarray(metadata["session"])
        if "session" in metadata
        else np.full(len(labels), "unknown", dtype=object)
    )
    runs = (
        np.asarray(metadata["run"])
        if "run" in metadata
        else np.full(len(labels), "unknown", dtype=object)
    )
    np.savez(
        output_path,
        raw_features=raw_features,
        scaled_features=scaled_features,
        embedding=embedding,
        c3_c4_features=c3_c4_features,
        fbcsp_features=fbcsp_features,
        labels=labels,
        sessions=sessions,
        runs=runs,
        feature_names=np.asarray(feature_names, dtype=object),
    )
    return output_path


def build_config_from_namespace(args: object) -> HybridReductionConfig:
    """把统一 CLI 参数转成当前范式的配置对象。"""

    output_dir = args.output_dir if args.output_dir is not None else get_hybrid_results_dir()
    return HybridReductionConfig(
        subject_id=args.subject,
        output_dir=output_dir,
        show=args.show,
        supervised_umap=args.supervised_umap,
    )


def run_from_config(config: HybridReductionConfig) -> dict[str, object]:
    """执行混合特征 + UMAP 范式。"""

    output_dir = config.output_dir or get_hybrid_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("正在读取 BCICIV2a 被试的全通道数据...", flush=True)
    X_all, labels, metadata, sfreq = load_subject_epochs(
        subject_id=config.subject_id,
        tmin=config.tmin,
        tmax=config.tmax,
        channels=None,
    )

    print("正在从全通道数据中提取 C3 / C4 通道...", flush=True)
    X_c3_c4 = select_c3_c4_from_full_channels(X_all)
    labels_int = np.asarray([LABEL_TO_INT[label] for label in labels])

    print("正在提取 C3 / C4 频域特征...", flush=True)
    c3_c4_features, c3_c4_feature_names = extract_c3_c4_frequency_features(
        X_c3_c4,
        sfreq,
    )

    print("正在提取 FBCSP 空间滤波特征...", flush=True)
    fbcsp_features, fbcsp_feature_names = extract_fbcsp_features(
        X_all,
        labels_int,
        sfreq,
    )

    print("正在拼接 C3/C4 特征与 FBCSP 特征...", flush=True)
    features = np.concatenate([c3_c4_features, fbcsp_features], axis=1)
    feature_names = c3_c4_feature_names + fbcsp_feature_names

    reduction_name = "supervised UMAP" if config.supervised_umap else "UMAP"
    print(f"正在执行 {reduction_name} 三维降维，首次运行可能会稍等几秒...", flush=True)
    embedding, scaled_features = reduce_features_to_3d(
        features=features,
        labels=labels,
        supervised=config.supervised_umap,
    )

    print("正在保存特征结果...", flush=True)
    embedding_name = "supervised_umap3d" if config.supervised_umap else "umap3d"
    feature_package_path = save_feature_package(
        output_dir=output_dir,
        subject_id=config.subject_id,
        embedding_name=embedding_name,
        raw_features=features,
        scaled_features=scaled_features,
        embedding=embedding,
        c3_c4_features=c3_c4_features,
        fbcsp_features=fbcsp_features,
        labels=labels,
        feature_names=feature_names,
        metadata=metadata,
    )

    print("正在绘制 3D 散点图...", flush=True)
    figure_path = output_dir / f"subject_{config.subject_id:02d}_hybrid_fbcsp_{embedding_name}.png"
    plot_3d_embedding(
        embedding=embedding,
        labels=labels,
        title=f"BCICIV2a Subject {config.subject_id}: C3/C4 + FBCSP -> {reduction_name} 3D",
        save_path=figure_path,
        show=config.show,
    )

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"被试编号: {config.subject_id}")
    print(f"全通道原始 trial 数据形状: {X_all.shape}")
    print(f"C3/C4 原始 trial 数据形状: {X_c3_c4.shape}")
    print(f"C3/C4 频域特征形状: {c3_c4_features.shape}")
    print(f"FBCSP 特征形状: {fbcsp_features.shape}")
    print(f"融合后特征矩阵形状: {features.shape}")
    print(f"{reduction_name} 3D 嵌入形状: {embedding.shape}")
    print("类别样本数:")
    for label, count in zip(unique_labels, counts, strict=True):
        print(f"  - {LABEL_TO_DISPLAY_NAME.get(label, label)}: {count}")
    print(f"特征包已保存到: {feature_package_path}")
    print(f"3D 散点图已保存到: {figure_path}")

    return {
        "subject_id": config.subject_id,
        "output_dir": output_dir,
        "feature_package_path": feature_package_path,
        "figure_path": figure_path,
        "feature_names": feature_names,
        "labels": labels,
    }
