from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import pywt
except ImportError as exc:
    raise ImportError(
        "没有检测到 PyWavelets，请先安装：\n"
        "test_newPyEnv/.venv/bin/pip install PyWavelets"
    ) from exc


class WaveletEnergyFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    用 Morlet 小波提取时频能量与熵特征。

    实现思路：
    1. 对每个 trial、每个通道做连续小波变换(CWT)。
    2. 在 mu / beta 等频带内计算平均能量。
    3. 计算频带功率分布的 Shannon 熵，补充“能量是否集中”的信息。

    这类特征比静态 PSD 更强调时频局部变化，适合运动想象中短暂的节律抑制/增强现象。
    """

    def __init__(
        self,
        sfreq: float,
        freqs: np.ndarray | None = None,
        wavelet_name: str = "cmor1.5-1.0",
    ):
        self.sfreq = sfreq
        self.freqs = np.asarray(freqs if freqs is not None else np.arange(8, 31, 2))
        self.wavelet_name = wavelet_name
        self.bands = {
            "mu": (8.0, 12.0),
            "beta_low": (13.0, 20.0),
            "beta_high": (20.0, 30.0),
        }

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "WaveletEnergyFeatureExtractor":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        eps = 1e-10
        wavelet = pywt.ContinuousWavelet(self.wavelet_name)
        central_frequency = pywt.central_frequency(wavelet)
        scales = central_frequency * self.sfreq / self.freqs
        features: list[list[float]] = []

        for trial in X:
            trial_features: list[float] = []
            for channel_signal in trial:
                coefficients, _ = pywt.cwt(
                    channel_signal,
                    scales,
                    wavelet,
                    sampling_period=1.0 / self.sfreq,
                )
                power = np.abs(coefficients) ** 2

                for low_freq, high_freq in self.bands.values():
                    band_mask = (self.freqs >= low_freq) & (self.freqs <= high_freq)
                    band_power = power[band_mask]

                    mean_energy = float(np.mean(band_power))
                    normalized_power = band_power / (np.sum(band_power) + eps)
                    entropy = float(
                        -np.sum(normalized_power * np.log(normalized_power + eps))
                    )

                    trial_features.append(np.log10(mean_energy + eps))
                    trial_features.append(entropy)

            features.append(trial_features)

        return np.asarray(features, dtype=np.float32)

    def get_feature_names(self, channel_names: list[str]) -> list[str]:
        names: list[str] = []
        for channel_name in channel_names:
            for band_name in self.bands:
                names.append(f"{channel_name}_{band_name}_wavelet_log_energy")
                names.append(f"{channel_name}_{band_name}_wavelet_entropy")
        return names
