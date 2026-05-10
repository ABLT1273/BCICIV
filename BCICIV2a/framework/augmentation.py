"""EEG data augmentation transforms for PyTorch.

Usage in DataLoader:
    transform = Compose([SlidingWindow(crop_samples=500), GaussNoise(snr_db=15)])
    augmented_X = transform(X_batch)  # X_batch: (B, C, T)
"""

from __future__ import annotations

import numpy as np
import torch


class SlidingWindow:
    """Randomly crop a shorter window from each trial.

    Useful when loading longer epochs (e.g. 0.0-4.0s = 1000 samples at 250Hz)
    but training on shorter windows (e.g. 500 samples = 2s).
    """

    def __init__(self, crop_samples: int):
        self.crop_samples = crop_samples

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        # X: (n_channels, n_times) or (batch, channels, times)
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        total = X.shape[-1]
        if total <= self.crop_samples:
            result = X
        else:
            start = np.random.randint(0, total - self.crop_samples + 1)
            result = X[..., start:start + self.crop_samples]

        return result.squeeze(0) if squeeze_back else result


class GaussNoise:
    """Add Gaussian noise at a specified SNR (dB)."""

    def __init__(self, snr_db: float = 15.0):
        self.snr_db = snr_db

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        signal_power = X.std(dim=-1, keepdim=True).pow(2).mean()
        snr_linear = 10.0 ** (self.snr_db / 10.0)
        noise_std = torch.sqrt(signal_power / snr_linear)
        noise = torch.randn_like(X) * noise_std
        return X + noise


class ChannelDropout:
    """Randomly zero out a fraction of channels."""

    def __init__(self, drop_prob: float = 0.1):
        self.drop_prob = drop_prob

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        n_ch = X.shape[-2]
        mask = torch.rand(n_ch) > self.drop_prob
        result = X * mask[:, None].to(X.device)

        return result.squeeze(0) if squeeze_back else result


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            X = t(X)
        return X


def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Mixup augmentation at batch level.

    Args:
        X: (batch, channels, times) input
        y: (batch,) integer labels
        alpha: Beta distribution parameter

    Returns:
        mixed_X, y_a, y_b, lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = X.size(0)
    index = torch.randperm(batch_size, device=X.device)
    mixed_X = lam * X + (1 - lam) * X[index]
    return mixed_X, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_augmented_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    use_window_crop: int | None = None,
    noise_snr_db: float | None = 15.0,
    channel_drop_prob: float | None = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> "AugmentedDataLoader":
    """Factory for a DataLoader-like object with EEG augmentations."""
    transforms = []
    if use_window_crop is not None:
        transforms.append(SlidingWindow(use_window_crop))
    if noise_snr_db is not None:
        transforms.append(GaussNoise(noise_snr_db))
    if channel_drop_prob is not None:
        transforms.append(ChannelDropout(channel_drop_prob))

    return AugmentedDataLoader(
        X, y, batch_size=batch_size, shuffle=shuffle,
        transforms=Compose(transforms) if transforms else None,
        use_mixup=use_mixup, mixup_alpha=mixup_alpha,
    )


class AugmentedDataLoader:
    """DataLoader wrapper that applies EEG augmentations on-the-fly.

    Use like a regular DataLoader: for Xb, yb in loader: ...
    Mixup returns (Xb, y_a, y_b, lam) instead of (Xb, yb).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 16,
        shuffle: bool = True,
        transforms: Compose | None = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
    ):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.n_samples = len(self.X)
        self._order = list(range(self.n_samples))

    def __iter__(self):
        if self.shuffle:
            self._order = np.random.permutation(self.n_samples).tolist()
        for start in range(0, self.n_samples, self.batch_size):
            indices = self._order[start:start + self.batch_size]
            Xb = self.X[indices]
            yb = self.y[indices]
            if self.transforms is not None:
                Xb = self.transforms(Xb)
            if self.use_mixup:
                Xb, ya, yb2, lam = mixup_batch(Xb, yb, self.mixup_alpha)
                yield Xb, ya, yb2, lam
            else:
                yield Xb, yb

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
