from __future__ import annotations

import numpy as np


def compute_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    po = float(np.mean(y_true == y_pred))
    n_classes = len(np.unique(y_true))
    pe = 1.0 / n_classes if n_classes > 0 else 0.0
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else 0.0


def compute_accuracy_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    accuracy = float(np.mean(y_true == y_pred))
    unique_labels = np.unique(y_true)
    n_classes = len(unique_labels)
    pe = 1.0 / n_classes if n_classes > 0 else 0.0
    kappa = (accuracy - pe) / (1.0 - pe) if pe < 1.0 else 0.0
    return accuracy, float(kappa)
