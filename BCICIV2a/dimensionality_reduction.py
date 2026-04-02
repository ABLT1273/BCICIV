"""
在 BCI Competition IV 2a 数据集上：
1. 读取一个被试的运动想象数据。
2. 保留 C3 / C4 的经典频域特征分支。
3. 引入 FBCSP 分支，从全通道里提取空间滤波特征。
4. 把两路特征拼接后，用 supervised UMAP 降到 3 维。
5. 用 matplotlib 画 3D 散点图，不同类别用不同颜色区分。

说明：
- 这里不再只依赖 C3 / C4 的手工频域特征，而是额外引入 FBCSP。
- FBCSP 会利用多通道空间模式，把不同任务在空间滤波域里的差异显式提取出来。
- supervised UMAP 会把类别标签 y 也用于降维，让左手/右手/脚/舌在低维空间里更容易分开。
- 数据读取使用 MOABB 对 BCICIV2a 的官方封装：BNCI2014_001。
- 第一次运行时，如果本地没有缓存数据，MOABB 可能会自动下载数据集。

建议运行方式：
test_newPyEnv/.venv/bin/python test_newPyEnv/BCICIV/BCICIV2a/dimensionality_reduction.py --subject 1

如果你希望弹出图窗而不只是保存图片：
test_newPyEnv/.venv/bin/python test_newPyEnv/BCICIV/BCICIV2a/dimensionality_reduction.py --subject 1 --show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def prepare_runtime_environment() -> None:
    """
    为 MNE / MOABB / matplotlib 准备一个项目内的可写缓存目录。

    背景：
    - MNE 默认会往用户主目录下的 ~/.mne 写配置。
    - matplotlib 默认也会尝试使用 ~/.matplotlib 作为缓存目录。
    - 当前工作环境里，用户主目录不一定可写，所以这里主动把这些目录重定向到脚本旁边。
    """

    script_dir = Path(__file__).resolve().parent
    project_root = Path(__file__).resolve().parents[2]
    runtime_root = script_dir / ".runtime_cache"
    runtime_root.mkdir(parents=True, exist_ok=True)

    mne_home = runtime_root / ".mne"
    mpl_config = runtime_root / ".matplotlib"

    # 优先复用项目里已经存在的数据缓存，避免明明本地有数据却又去重新下载。
    existing_data_candidates = [
        project_root / ".mne_data",
        project_root / "mne_data",
    ]
    mne_data = next(
        (candidate for candidate in existing_data_candidates if candidate.exists()),
        runtime_root / "mne_data",
    )

    # 这些目录都要先创建好，避免后续库初始化时因为目录不存在而报错。
    for directory in (mne_home, mne_data, mpl_config):
        directory.mkdir(parents=True, exist_ok=True)

    # 这里直接覆盖环境变量，而不是使用 setdefault。
    # 原因是某些运行环境已经设置了 HOME=用户主目录，而那个目录未必可写，
    # 如果不强制改写，MNE 仍然会尝试往 ~/.mne 写配置文件并报权限错误。
    os.environ["HOME"] = str(runtime_root)
    os.environ["MNE_HOME"] = str(mne_home)
    os.environ["MNE_DATA"] = str(mne_data)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(mne_data)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)


# 一定要在导入 mne / moabb / matplotlib 之前准备好环境变量。
prepare_runtime_environment()

import matplotlib

# 默认使用非交互式后端，保证脚本在终端 / 服务器环境里也能稳定保存图片。
# 如果用户显式传入 --show，再保留 matplotlib 默认交互行为。
if "--show" not in os.sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import LabelEncoder, StandardScaler


# 统一定义类别名称，后面画图和打印结果时都会复用。
LABEL_TO_DISPLAY_NAME = {
    "left_hand": "Left hand",
    "right_hand": "Right hand",
    "feet": "Feet",
    "tongue": "Tongue",
}

# 给每个类别指定固定颜色，保证每次运行时颜色含义一致。
LABEL_TO_COLOR = {
    "left_hand": "#1f77b4",
    "right_hand": "#d62728",
    "feet": "#2ca02c",
    "tongue": "#ff7f0e",
}

LABEL_TO_INT = {
    "left_hand": 1,
    "right_hand": 2,
    "feet": 3,
    "tongue": 4,
}

INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}

BNCI2014001_CHANNEL_NAMES = [
    "FZ",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "P1",
    "PZ",
    "P2",
    "POZ",
]

# 这里挑选最常用的运动想象频带。
FREQUENCY_BANDS = {
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def load_subject_data(
    subject_id: int = 1,
    tmin: float = 0.5,
    tmax: float = 2.5,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, "object", float]:
    """
    读取一个被试的 BCICIV2a 数据。

    参数
    ----------
    subject_id:
        被试编号。BCICIV2a 一般是 1~9。
    tmin, tmax:
        每个 trial 的时间窗，单位是秒。
        这里沿用常见做法，截取 cue 后 0.5s 到 2.5s 的运动想象片段。
    channels:
        指定要加载的通道列表。
        - 传入 ["C3", "C4"] 时，只加载两个运动想象核心通道。
        - 传入 None 时，加载范式允许的全部 EEG 通道，供 FBCSP 使用。

    返回
    -------
    X:
        形状为 (n_trials, n_channels, n_samples) 的 EEG trial 数据。
    y:
        字符串标签，例如 left_hand / right_hand / feet / tongue。
    metadata:
        MOABB 返回的元信息，里面有 session、run 等字段。
    sfreq:
        采样率。BCICIV2a 固定为 250 Hz。
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

    # 如果 channels=None，就读取全部 EEG 通道；
    # 如果传入具体通道列表，就只保留这些通道。
    paradigm = MotorImagery(
        events=list(LABEL_TO_DISPLAY_NAME.keys()),
        n_classes=4,
        channels=channels,
        tmin=tmin,
        tmax=tmax,
    )

    try:
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    except Exception as exc:
        cache_dir = os.environ.get("MNE_DATA", "<未设置>")
        raise RuntimeError(
            "读取 BCICIV2a 数据失败。第一次运行时，MOABB 可能需要联网下载数据；"
            f"如果你已经下载过数据，请检查当前 MNE_DATA 缓存目录是否可用：{cache_dir}"
        ) from exc

    # y 转成 numpy 数组，便于后面统一做掩码索引。
    y = np.asarray(y)

    # 这里固定返回 250 Hz，因为 BCICIV2a 的采样率就是 250 Hz。
    sfreq = 250.0

    return X, y, metadata, sfreq


def integrate_band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """
    对指定频带做积分，得到频带功率。

    psd 形状是 (..., n_freqs)，返回结果形状是 psd 去掉最后一维后的形状。
    """

    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    if np.count_nonzero(band_mask) < 2:
        raise ValueError(
            f"频带 [{low_freq}, {high_freq}] Hz 内的频率采样点太少，无法稳定积分。"
        )
    return np.trapezoid(psd[..., band_mask], freqs[band_mask], axis=-1)


def extract_peak_frequency(
    psd: np.ndarray,
    freqs: np.ndarray,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """
    在指定频带内寻找峰值频率。

    这可以理解为：在 mu 或 beta 频带里，哪一个频率点的能量最高。
    对运动想象来说，它能补充“功率大小”之外的频谱形状信息。
    """

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
    从 C3 / C4 两个通道提取频域特征。

    这里对每个 trial 做 Welch 功率谱估计，然后构造以下特征：
    1. C3、C4 在 mu / beta 频带上的对数功率
    2. C3、C4 在 mu / beta 频带上的相对功率
    3. C3 与 C4 的左右差异（对数功率差、相对功率差）
    4. 每个频带内的峰值频率

    这些特征兼顾了：
    - 单通道能量强弱
    - 相对能量分布
    - 左右运动皮层的不对称性
    - 频谱峰位置
    """

    # 输入形状应为 (n_trials, 2, n_samples)。
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(
            f"期望输入形状为 (n_trials, 2, n_samples)，实际收到 {X.shape}。"
        )

    # Welch 会在每个 trial、每个通道上估计功率谱密度。
    # axis=-1 表示沿着时间轴做频谱分析。
    nperseg = min(256, X.shape[-1])
    freqs, psd = welch(
        X,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
    )

    eps = 1e-12

    # 先算 4~40 Hz 的总功率，后面用于计算相对功率，避免只看绝对值时受整体幅值影响太大。
    total_power = integrate_band_power(psd, freqs, 4.0, 40.0) + eps

    feature_columns: list[np.ndarray] = []
    feature_names: list[str] = []

    # C3 放在第 0 维，C4 放在第 1 维。
    c3_index = 0
    c4_index = 1

    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        band_power = integrate_band_power(psd, freqs, low_freq, high_freq) + eps
        relative_power = band_power / total_power
        peak_frequency = extract_peak_frequency(psd, freqs, low_freq, high_freq)

        log_power = np.log10(band_power)
        log_power_diff = log_power[:, c3_index] - log_power[:, c4_index]
        relative_power_diff = (
            relative_power[:, c3_index] - relative_power[:, c4_index]
        )

        # 依次把每一列特征压进列表里，最后再统一拼成二维特征矩阵。
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

    features = np.column_stack(feature_columns)
    return features, feature_names


def select_c3_c4_from_full_channels(X_all: np.ndarray) -> np.ndarray:
    """
    从 BCICIV2a 的 22 通道全通道数据里切出 C3 / C4。

    这里基于 BNCI2014-001 的固定通道顺序：
    C3 位于索引 7，C4 位于索引 11。
    为了让代码更可读，这里仍然通过通道名查索引，而不是直接写死数字。
    """

    if X_all.ndim != 3 or X_all.shape[1] != len(BNCI2014001_CHANNEL_NAMES):
        raise ValueError(
            "当前数据形状与 BCICIV2a 的 22 通道设置不一致，"
            f"收到的形状是 {X_all.shape}。"
        )

    c3_index = BNCI2014001_CHANNEL_NAMES.index("C3")
    c4_index = BNCI2014001_CHANNEL_NAMES.index("C4")
    return X_all[:, [c3_index, c4_index], :]


def extract_fbcsp_features(
    X: np.ndarray,
    labels_int: np.ndarray,
    sfreq: float,
    m: int = 2,
    k: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """
    复用现有 FBCSP.py 的 FilterBank / PairedMIBIF 思路，从全通道里提取 FBCSP 特征。

    这里不直接调用最终分类器，而是把 FBCSP 当成“监督式特征提取器”来使用：
    1. 先把全通道 EEG 划分到多个频带。
    2. 对每个类别做 One-Versus-Rest 的 CSP。
    3. 在每个 OVR 分支中，用 PairedMIBIF 选出信息量最大的 CSP 特征。
    4. 把所有 OVR 分支的被选特征拼起来，作为 FBCSP 特征矩阵。

    这样做的好处是：
    - FBCSP 能捕获多通道空间模式；
    - C3/C4 频域特征能保留直观的局部生理解释；
    - 两者拼接后通常比单一路径更有判别力。
    """

    try:
        from mne.decoding import CSP
        from FBCSP import FilterBank, PairedMIBIF
    except ImportError as exc:
        raise ImportError(
            "导入 FBCSP 相关模块失败，请确认 FBCSP.py 与依赖已经可用。"
        ) from exc

    filter_bank = FilterBank(sfreq=int(sfreq))
    X_fb = filter_bank.transform(X)

    n_bands = X_fb.shape[0]
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

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

    features = np.concatenate(feature_blocks, axis=1)
    return features, feature_names


def reduce_features_to_3d(
    features: np.ndarray,
    labels: np.ndarray,
    supervised: bool = False,
) -> tuple[np.ndarray, np.ndarray, "object"]:
    """
    先标准化特征，再用 UMAP 把特征压到 3 维。

    为什么要先标准化？
    - 因为不同特征的量纲不同，例如“对数功率”和“峰值频率”数值范围不一样。
    - 如果不标准化，UMAP 的距离计算会被大尺度特征主导。

    当 supervised=True 时：
    - UMAP 会额外利用类别标签，让低维空间更强调类间分离。
    - 这通常会让不同运动想象类别分得更开。
    """

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "没有检测到 umap-learn，请先在当前虚拟环境里安装：\n"
            "test_newPyEnv/.venv/bin/pip install umap-learn"
        ) from exc

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_neighbors = min(30, max(5, len(features_scaled) - 1))

    reducer_kwargs = dict(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )

    if supervised:
        reducer_kwargs["target_metric"] = "categorical"
        reducer_kwargs["target_weight"] = 0.35

    reducer = umap.UMAP(**reducer_kwargs)

    if supervised:
        encoded_labels = LabelEncoder().fit_transform(labels)
        embedding = reducer.fit_transform(features_scaled, y=encoded_labels)
    else:
        embedding = reducer.fit_transform(features_scaled)
    return embedding, features_scaled, reducer


def plot_3d_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    subject_id: int,
    save_path: Path,
    reduction_name: str,
    show: bool = False,
) -> None:
    """
    绘制 3D UMAP 散点图。

    不同任务类别用不同颜色表示，方便观察在低维空间中的聚类趋势。
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label in LABEL_TO_DISPLAY_NAME:
        mask = labels == label
        if not np.any(mask):
            continue

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=36,
            alpha=0.85,
            color=LABEL_TO_COLOR[label],
            label=LABEL_TO_DISPLAY_NAME[label],
            edgecolors="black",
            linewidths=0.2,
        )

    ax.set_title(
        f"BCICIV2a Subject {subject_id}: C3/C4 + FBCSP -> {reduction_name} 3D",
        pad=18,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


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
    metadata: "object",
) -> Path:
    """
    把提取好的特征和降维结果一起保存成 .npz 文件。

    这样你后续如果想继续做分类、聚类、或者换别的可视化方法，就不用重复提特征。
    """

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


def build_argument_parser() -> argparse.ArgumentParser:
    """定义命令行参数，方便你切换被试编号和输出目录。"""

    parser = argparse.ArgumentParser(
        description="读取 BCICIV2a 的一个被试，融合 C3/C4 频域特征与 FBCSP 特征，并做 supervised UMAP 3D 可视化。"
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="被试编号，BCICIV2a 通常为 1~9，默认值为 1。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="输出目录，保存图片和 .npz 特征文件。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否在保存图片后弹出图窗显示。",
    )
    parser.add_argument(
        "--supervised-umap",
        action="store_true",
        help="启用 supervised UMAP；默认关闭，此时使用普通 UMAP。",
    )
    return parser


def main() -> None:
    """脚本主入口。把所有步骤串起来执行。"""

    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取一个被试的数据。
    print("正在读取 BCICIV2a 被试的全通道数据...", flush=True)
    X_all, labels, metadata, sfreq = load_subject_data(
        subject_id=args.subject,
        channels=None,
    )

    print("正在从全通道数据中提取 C3 / C4 通道...", flush=True)
    X_c3_c4 = select_c3_c4_from_full_channels(X_all)

    labels_int = np.asarray([LABEL_TO_INT[label] for label in labels])

    # 2. 从 C3 / C4 上提取频域特征。
    print("正在提取 C3 / C4 频域特征...", flush=True)
    c3_c4_features, c3_c4_feature_names = extract_c3_c4_frequency_features(
        X_c3_c4,
        sfreq,
    )

    # 3. 从全通道上提取 FBCSP 特征。
    print("正在提取 FBCSP 空间滤波特征...", flush=True)
    fbcsp_features, fbcsp_feature_names = extract_fbcsp_features(
        X_all,
        labels_int,
        sfreq,
    )

    # 4. 拼接两路特征，形成强化后的混合特征表示。
    print("正在拼接 C3/C4 特征与 FBCSP 特征...", flush=True)
    features = np.concatenate([c3_c4_features, fbcsp_features], axis=1)
    feature_names = c3_c4_feature_names + fbcsp_feature_names

    # 5. 用 supervised UMAP 压到 3 维。
    reduction_name = "supervised UMAP" if args.supervised_umap else "UMAP"
    print(f"正在执行 {reduction_name} 三维降维，首次运行可能会稍等几秒...", flush=True)
    embedding, scaled_features, _ = reduce_features_to_3d(
        features,
        labels,
        supervised=args.supervised_umap,
    )

    # 6. 保存特征和降维结果，便于后续复用。
    print("正在保存特征结果...", flush=True)
    feature_package_path = save_feature_package(
        output_dir=args.output_dir,
        subject_id=args.subject,
        embedding_name=(
            "supervised_umap3d" if args.supervised_umap else "umap3d"
        ),
        raw_features=features,
        scaled_features=scaled_features,
        embedding=embedding,
        c3_c4_features=c3_c4_features,
        fbcsp_features=fbcsp_features,
        labels=labels,
        feature_names=feature_names,
        metadata=metadata,
    )

    # 7. 画 3D 散点图。
    print("正在绘制 3D 散点图...", flush=True)
    figure_stem = (
        "subject_"
        f"{args.subject:02d}_hybrid_fbcsp_"
        f"{'supervised_umap3d' if args.supervised_umap else 'umap3d'}"
    )
    figure_path = args.output_dir / f"{figure_stem}.png"
    plot_3d_embedding(
        embedding=embedding,
        labels=labels,
        subject_id=args.subject,
        save_path=figure_path,
        reduction_name=reduction_name,
        show=args.show,
    )

    # 6. 在终端打印关键结果，方便快速确认脚本有没有跑通。
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"被试编号: {args.subject}")
    print(f"全通道原始 trial 数据形状: {X_all.shape}")
    print(f"C3/C4 原始 trial 数据形状: {X_c3_c4.shape}")
    print(f"C3/C4 频域特征形状: {c3_c4_features.shape}")
    print(f"FBCSP 特征形状: {fbcsp_features.shape}")
    print(f"融合后特征矩阵形状: {features.shape}")
    print(f"{reduction_name} 3D 嵌入形状: {embedding.shape}")
    print("类别样本数:")
    for label, count in zip(unique_labels, counts, strict=True):
        print(f"  - {LABEL_TO_DISPLAY_NAME.get(label, label)}: {count}")
    print("特征名称:")
    for feature_name in feature_names:
        print(f"  - {feature_name}")
    print(f"特征包已保存到: {feature_package_path}")
    print(f"3D 散点图已保存到: {figure_path}")


if __name__ == "__main__":
    main()
