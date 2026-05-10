from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .constants import BNCI2014001_CHANNEL_NAMES, LABEL_TO_DISPLAY_NAME


def _design_bandpass(low: float, high: float, sfreq: float, order: int = 4):
    nyq = 0.5 * sfreq
    low_n = low / nyq
    high_n = high / nyq
    return scipy_signal.cheby2(order, 40, [low_n, high_n], btype="band", output="sos")


def _design_notch(freq: float, sfreq: float, q: float = 30):
    nyq = 0.5 * sfreq
    w0 = freq / nyq
    b, a = scipy_signal.iirnotch(w0, q)
    return b, a


def apply_bandpass(X: np.ndarray, sfreq: float, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Apply zero-phase bandpass filter to each trial."""
    sos = _design_bandpass(low, high, sfreq)
    out = np.zeros_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        out[i] = scipy_signal.sosfiltfilt(sos, X[i], axis=-1)
    return out


def apply_notch(X: np.ndarray, sfreq: float, freq: float = 50.0) -> np.ndarray:
    """Apply zero-phase notch filter to each trial."""
    b, a = _design_notch(freq, sfreq)
    out = np.zeros_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        out[i] = scipy_signal.filtfilt(b, a, X[i], axis=-1)
    return out


def apply_car(X: np.ndarray) -> np.ndarray:
    """Common Average Reference: subtract mean across channels per time point."""
    return X - X.mean(axis=1, keepdims=True)


def apply_zscore(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """Z-score normalize per channel-time point using training set statistics."""
    if mean is None or std is None:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def preprocess_eeg(
    X: np.ndarray,
    sfreq: float,
    bandpass: tuple[float, float] | None = (0.5, 40.0),
    notch: float | None = 50.0,
    apply_car_flag: bool = True,
    zscore: bool = False,
    zscore_mean: np.ndarray | None = None,
    zscore_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply standard EEG preprocessing steps.

    Args:
        X: (n_trials, n_channels, n_samples) raw EEG data
        sfreq: sampling frequency in Hz
        bandpass: (low, high) cutoff frequencies; None to skip
        notch: notch frequency in Hz; None to skip
        apply_car_flag: whether to apply CAR
        zscore: whether to z-score normalize
        zscore_mean, zscore_std: pre-computed normalization stats

    Returns:
        preprocessed X, and (mean, std) if zscore=True, else (None, None)
    """
    out = X.astype(np.float64)

    if bandpass is not None:
        out = apply_bandpass(out, sfreq, bandpass[0], bandpass[1])
    if notch is not None:
        out = apply_notch(out, sfreq, notch)
    if apply_car_flag:
        out = apply_car(out)

    if zscore:
        out, mean, std = apply_zscore(out, zscore_mean, zscore_std)
        return out, mean, std
    return out.astype(np.float32), None, None


def load_subject_epochs(
    subject_id: int,
    tmin: float = 0.5,
    tmax: float = 2.5,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, "object", float]:
    """Unified loading of epoched data for one subject via MOABB."""
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise ImportError(
            "MOABB is required.  Install with:\n"
            "  pip install moabb"
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
    bandpass: tuple[float, float] | None = (0.5, 40.0),
    notch: float | None = 50.0,
    apply_car: bool = True,
    zscore: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load train/test split for one subject with optional preprocessing.

    Args:
        subject_id: 1-9
        tmin, tmax: epoch time window in seconds (default 0.0-4.0s)
        channels: channel subset or None for all 22
        bandpass: (low, high) Hz; None to skip
        notch: notch filter frequency; None to skip
        apply_car: whether to apply common average reference
        zscore: whether to apply z-score normalization (per time-chan)

    Returns:
        X_train, X_test, y_train, y_test, sfreq
    """
    X, y, metadata, sfreq = load_subject_epochs(
        subject_id=subject_id, tmin=tmin, tmax=tmax, channels=channels,
    )
    is_train = metadata["session"].astype(str).str.contains("train").to_numpy()
    is_test = metadata["session"].astype(str).str.contains("test").to_numpy()
    X_train, X_test = X[is_train], X[is_test]
    y_train, y_test = y[is_train], y[is_test]

    # Apply common preprocessing
    if bandpass is not None or notch is not None or apply_car:
        n_train, n_test = len(X_train), len(X_test)
        all_X = np.concatenate([X_train, X_test], axis=0)
        all_X, _, _ = preprocess_eeg(all_X, sfreq, bandpass=bandpass, notch=notch, apply_car_flag=apply_car, zscore=False)
        X_train, X_test = all_X[:n_train], all_X[n_train:]

    if zscore:
        X_train, mean, std = apply_zscore(X_train)
        X_test, _, _ = apply_zscore(X_test, mean, std)

    return X_train, X_test, y_train, y_test, sfreq


def select_named_channels(X: np.ndarray, channel_names: list[str]) -> np.ndarray:
    indices = [BNCI2014001_CHANNEL_NAMES.index(name) for name in channel_names]
    return X[:, indices, :]


def select_c3_c4_from_full_channels(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3 or X.shape[1] != len(BNCI2014001_CHANNEL_NAMES):
        raise ValueError(
            "Data shape does not match BCICIV2a 22-channel layout. "
            f"Got shape {X.shape}."
        )
    c3_index = BNCI2014001_CHANNEL_NAMES.index("C3")
    c4_index = BNCI2014001_CHANNEL_NAMES.index("C4")
    return X[:, [c3_index, c4_index], :]
