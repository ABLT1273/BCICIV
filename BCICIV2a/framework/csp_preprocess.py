"""CSP spatial filtering as preprocessing for deep learning models.

Applies OVR-CSP spatial filters to transform raw EEG channels into
discriminative spatial component time series, preserving the time
dimension so convolutional models can still learn temporal patterns.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin

from .constants import LABEL_TO_INT


class CSPFilterBank(BaseEstimator, TransformerMixin):
    """Multi-class CSP via OVR: learn spatial filters and project raw
    data to CSP virtual channels while preserving the time axis.

    Output shape: (n_trials, n_classes * n_components, n_times)
    """

    def __init__(self, n_components: int = 8, reg: float | None = None):
        self.n_components = n_components
        self.reg = reg
        self.filters_: list[np.ndarray] = []
        self._classes: list[int] = []

    def _compute_csp_filters(self, X: np.ndarray, y_binary: np.ndarray) -> np.ndarray:
        """Compute CSP spatial filters for a binary problem.

        Returns filters of shape (n_components, n_channels).
        """
        n_trials, n_channels, n_times = X.shape
        class_0 = X[y_binary == 0]
        class_1 = X[y_binary == 1]

        # Compute class-wise covariance matrices
        cov_0 = np.mean([np.cov(trial) for trial in class_0], axis=0)
        cov_1 = np.mean([np.cov(trial) for trial in class_1], axis=0)

        if self.reg is not None:
            cov_0 = cov_0 + self.reg * np.eye(n_channels)
            cov_1 = cov_1 + self.reg * np.eye(n_channels)

        # Eigen decomposition of cov_0^-1 * cov_1
        eigenvalues, eigenvectors = linalg.eigh(cov_1, cov_0 + cov_1)

        # Sort eigenvalues in descending order
        idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        sorted_eigenvectors = eigenvectors[:, idx]

        # Take top and bottom n_components/2
        half = self.n_components // 2
        top_filters = sorted_eigenvectors[:, :half].T  # (half, n_channels)
        bottom_filters = sorted_eigenvectors[:, -half:].T  # (half, n_channels)
        filters = np.concatenate([top_filters, bottom_filters], axis=0)
        return filters  # (n_components, n_channels)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPFilterBank":
        label_to_int = LABEL_TO_INT
        y_int = np.array([label_to_int[lbl] if isinstance(lbl, str) else int(lbl) for lbl in y])
        self._classes = sorted(set(y_int.tolist()))

        self.filters_ = []
        for cls in self._classes:
            y_binary = (y_int == cls).astype(int)
            filters = self._compute_csp_filters(X, y_binary)
            self.filters_.append(filters)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.filters_:
            raise RuntimeError("CSPFilterBank must be fitted before transform.")
        features = []
        for W in self.filters_:  # W: (n_components, n_channels)
            # W @ X: (n_components, n_channels) @ (n_trials, n_channels, n_times)
            proj = np.einsum("cn,bnt->bct", W, X)
            # log-variance normalization per component
            log_var_proj = np.log(np.var(proj, axis=-1, keepdims=True) + 1e-8)
            features.append(log_var_proj * proj)
        return np.concatenate(features, axis=1).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
