"""
Cross-validation splitting utilities for BCICIV2a.

Provides:
- LOSO (Leave-One-Subject-Out): train on N-1 subjects, test on held-out subject
- LOSO cross-session: train on Session T of N-1 subjects, test on Session E of held-out
- LOSO same-session: train on Session T of N-1 subjects, test on Session T of held-out
"""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np

from .data import load_subject_epochs, preprocess_eeg, apply_zscore

logger = logging.getLogger(__name__)


def load_subject_sessions(
    subject_id: int,
    tmin: float = 0.5,
    tmax: float = 2.5,
    channels: list[str] | None = None,
    bandpass: tuple[float, float] | None = (0.5, 40.0),
    notch: float | None = 50.0,
    apply_car: bool = True,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], float]:
    """Load a single subject's data split by session.

    Args:
        subject_id: 1-9
        tmin, tmax: epoch time window in seconds
        channels: channel subset or None for all 22
        bandpass: (low, high) Hz; None to skip
        notch: notch filter frequency; None to skip
        apply_car: whether to apply common average reference

    Returns:
        (X_session_T, y_session_T), (X_session_E, y_session_E), sfreq
    """
    X, y, metadata, sfreq = load_subject_epochs(
        subject_id=subject_id, tmin=tmin, tmax=tmax, channels=channels,
    )

    is_train = metadata["session"].astype(str).str.contains("train").to_numpy()
    is_test = metadata["session"].astype(str).str.contains("test").to_numpy()

    X_T, y_T = X[is_train], y[is_train]
    X_E, y_E = X[is_test], y[is_test]

    # Apply bandpass / notch / CAR
    if bandpass is not None or notch is not None or apply_car:
        n_T, n_E = len(X_T), len(X_E)
        all_X = np.concatenate([X_T, X_E], axis=0)
        all_X, _, _ = preprocess_eeg(
            all_X, sfreq, bandpass=bandpass, notch=notch, apply_car_flag=apply_car, zscore=False,
        )
        X_T, X_E = all_X[:n_T], all_X[n_T:]

    return (X_T.astype(np.float32), y_T), (X_E.astype(np.float32), y_E), sfreq


def load_all_subjects_sessions(
    subject_ids: list[int],
    tmin: float = 0.5,
    tmax: float = 2.5,
    channels: list[str] | None = None,
    bandpass: tuple[float, float] | None = (0.5, 40.0),
    notch: float | None = 50.0,
    apply_car: bool = True,
) -> dict[int, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], float]]:
    """Load session-split data for all subjects.

    Returns:
        Dict mapping subject_id -> ((X_T, y_T), (X_E, y_E), sfreq)
    """
    all_data: dict[int, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], float]] = {}
    for sid in subject_ids:
        logger.info("Loading subject %d sessions...", sid)
        all_data[sid] = load_subject_sessions(
            subject_id=sid,
            tmin=tmin,
            tmax=tmax,
            channels=channels,
            bandpass=bandpass,
            notch=notch,
            apply_car=apply_car,
        )
    return all_data


def _check_class_balance(y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Warn if any class is missing from train or test set."""
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    if train_classes != test_classes:
        logger.warning(
            "Class mismatch: train=%s, test=%s. Missing classes: test-only=%s, train-only=%s",
            sorted(train_classes),
            sorted(test_classes),
            sorted(test_classes - train_classes),
            sorted(train_classes - test_classes),
        )


def build_loso_fold(
    all_data: dict[int, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], float]],
    held_out_subject: int,
    train_session: str = "T",
    test_session: str = "E",
    zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """Build a single LOSO train/test fold.

    Args:
        all_data: output of load_all_subjects_sessions()
        held_out_subject: subject to hold out for testing
        train_session: which session to use for training subjects ("T", "E", or "both")
        test_session: which session to use for the held-out subject ("T", "E", or "both")
        zscore: whether to z-score normalize (using training set statistics)

    Returns:
        X_train, y_train, X_test, y_test, held_out_subject, fold_meta
    """
    train_subjects = [s for s in all_data if s != held_out_subject]

    # --- Collect training data ---
    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []

    for sid in train_subjects:
        (X_T, y_T), (X_E, y_E), _sfreq = all_data[sid]

        def _pick(session_key: str) -> tuple[np.ndarray, np.ndarray]:
            if session_key == "T":
                return X_T, y_T
            elif session_key == "E":
                return X_E, y_E
            else:  # "both"
                return np.concatenate([X_T, X_E], axis=0), np.concatenate([y_T, y_E], axis=0)

        X_part, y_part = _pick(train_session)
        X_train_parts.append(X_part)
        y_train_parts.append(y_part)

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    # --- Collect test data ---
    (X_T_held, y_T_held), (X_E_held, y_E_held), _sfreq = all_data[held_out_subject]

    def _pick_test(session_key: str) -> tuple[np.ndarray, np.ndarray]:
        if session_key == "T":
            return X_T_held, y_T_held
        elif session_key == "E":
            return X_E_held, y_E_held
        else:  # "both"
            return (
                np.concatenate([X_T_held, X_E_held], axis=0),
                np.concatenate([y_T_held, y_E_held], axis=0),
            )

    X_test, y_test = _pick_test(test_session)

    _check_class_balance(y_train, y_test)

    # --- Z-score normalization ---
    if zscore:
        X_train, mean, std = apply_zscore(X_train)
        X_test, _, _ = apply_zscore(X_test, mean, std)

    fold_meta = {
        "held_out_subject": held_out_subject,
        "train_subjects": train_subjects,
        "train_session": train_session,
        "test_session": test_session,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return X_train, y_train, X_test, y_test, held_out_subject, fold_meta


def generate_loso_folds(
    subject_ids: list[int],
    cross_session: bool = True,
    tmin: float = 0.5,
    tmax: float = 2.5,
    channels: list[str] | None = None,
    bandpass: tuple[float, float] | None = (0.5, 40.0),
    notch: float | None = 50.0,
    apply_car: bool = True,
    zscore: bool = True,
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict], None, None]:
    """Generate LOSO folds for all subjects.

    Each fold yields (X_train, y_train, X_test, y_test, held_out_subject, fold_meta).

    Args:
        subject_ids: list of subject IDs (e.g., [1..9])
        cross_session: if True, train on Session T of N-1 subjects, test on Session E of held-out.
                       if False, train on Session T of N-1 subjects, test on Session T of held-out
                       (same-session LOSO, measures cross-subject generalization without session shift).
        tmin, tmax: epoch time window
        channels: channel subset
        bandpass, notch, apply_car: preprocessing flags
        zscore: whether to z-score per fold

    Yields:
        X_train, y_train, X_test, y_test, held_out_subject, fold_meta
    """
    all_data = load_all_subjects_sessions(
        subject_ids=subject_ids,
        tmin=tmin,
        tmax=tmax,
        channels=channels,
        bandpass=bandpass,
        notch=notch,
        apply_car=apply_car,
    )

    test_session = "E" if cross_session else "T"

    for held_out_subject in subject_ids:
        logger.info(
            "Building LOSO fold: held_out=%d, train_session=T, test_session=%s",
            held_out_subject,
            test_session,
        )
        yield build_loso_fold(
            all_data=all_data,
            held_out_subject=held_out_subject,
            train_session="T",
            test_session=test_session,
            zscore=zscore,
        )


def describe_folds(subject_ids: list[int], cross_session: bool = True) -> str:
    """Return a human-readable description of the LOSO fold structure."""
    test_session = "E" if cross_session else "T"
    lines = [
        f"LOSO {'cross-session' if cross_session else 'same-session'} evaluation",
        f"  Subjects: {subject_ids}",
        f"  Train: Session T from all subjects except the held-out one",
        f"  Test:  Session {test_session} from the held-out subject",
    ]
    for held_out in subject_ids:
        train_subs = [s for s in subject_ids if s != held_out]
        lines.append(f"  Fold held_out=S{held_out:02d}: train on S{sorted(train_subs)}, test on S{held_out:02d}")
    return "\n".join(lines)
