"""
DeepConvNet and ShallowConvNet for EEG decoding.

Based on Schirrmeister et al. 2017:
"Deep learning with convolutional neural networks for EEG decoding and
visualization" (Human Brain Mapping).

Both models follow the same train/predict API as EEGNet for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from framework.data import apply_zscore as _normalize_eeg
from framework.devices import resolve_torch_device


# ---------------------------------------------------------------------------
# DeepConvNet
# ---------------------------------------------------------------------------

class DeepConvNet(nn.Module):
    """Deep Convolutional Network for EEG.

    4 conv-pool blocks with increasing filter counts.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10)),
            nn.Conv2d(25, 25, kernel_size=(n_channels, 1), groups=25),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(p=dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 10)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(p=dropout),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(p=dropout),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 10)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(p=dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            flattened_dim = x.flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# ShallowConvNet
# ---------------------------------------------------------------------------

class ShallowConvNet(nn.Module):
    """Shallow Convolutional Network for EEG.

    Single temporal + spatial convolution block, inspired by FBCSP.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv_time = nn.Conv2d(1, 40, kernel_size=(1, 25))
        self.conv_spat = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), groups=40)
        self.bn = nn.BatchNorm2d(40)

        # Square activation, then average pooling
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.conv_time(dummy)
            x = self.conv_spat(x)
            x = self.bn(x)
            x = x ** 2
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            flattened_dim = x.flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Training / Inference (shared API with EEGNet)
# ---------------------------------------------------------------------------

@dataclass
class DeepConvResult:
    model: nn.Module
    train_mean: np.ndarray
    train_std: np.ndarray
    label_values: np.ndarray
    best_val_accuracy: float


def _train_dl_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 15,
    label_smoothing: float = 0.1,
    device: str | None = None,
) -> DeepConvResult:
    device = resolve_torch_device(device)

    label_values = np.unique(y_train)
    label_to_index = {label: index for index, label in enumerate(label_values)}
    y_train_idx = np.asarray([label_to_index[label] for label in y_train], dtype=np.int64)
    y_val_idx = np.asarray([label_to_index[label] for label in y_val], dtype=np.int64)

    X_train_norm, train_mean, train_std = _normalize_eeg(X_train)
    X_val_norm, _, _ = _normalize_eeg(X_val, train_mean, train_std)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_norm),
        torch.from_numpy(y_train_idx),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_state_dict = None
    best_val_accuracy = -1.0
    no_improvement_epochs = 0

    X_val_tensor = torch.from_numpy(X_val_norm).to(device)
    y_val_tensor = torch.from_numpy(y_val_idx).to(device)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val_tensor)
            predictions = logits.argmax(dim=1)
            val_accuracy = (predictions == y_val_tensor).float().mean().item()

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            no_improvement_epochs += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model_name = type(model).__name__
            print(f"  {model_name} epoch {epoch+1:3d}/{epochs}: val_acc={val_accuracy:.4f}", flush=True)

        if no_improvement_epochs >= early_stop_patience:
            model_name = type(model).__name__
            print(f"  {model_name} early stopping at epoch {epoch+1} (best_val_acc={best_val_accuracy:.4f})", flush=True)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return DeepConvResult(
        model=model,
        train_mean=train_mean,
        train_std=train_std,
        label_values=label_values,
        best_val_accuracy=best_val_accuracy,
    )


def _predict_dl_model(
    result: DeepConvResult,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    device = resolve_torch_device(device)
    X_norm, _, _ = _normalize_eeg(X, result.train_mean, result.train_std)

    result.model.eval()
    result.model.to(device)
    with torch.no_grad():
        logits = result.model(torch.from_numpy(X_norm).to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()
    return result.label_values[predictions]


def train_deep_conv_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    device: str | None = None,
) -> DeepConvResult:
    model = DeepConvNet(
        n_channels=X_train.shape[1],
        n_samples=X_train.shape[2],
        n_classes=len(np.unique(y_train)),
    )
    return _train_dl_model(model, X_train, y_train, X_val, y_val, epochs=epochs, device=device)


def predict_deep_conv_net(
    result: DeepConvResult,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    return _predict_dl_model(result, X, device=device)


def train_shallow_conv_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    device: str | None = None,
) -> DeepConvResult:
    model = ShallowConvNet(
        n_channels=X_train.shape[1],
        n_samples=X_train.shape[2],
        n_classes=len(np.unique(y_train)),
    )
    return _train_dl_model(model, X_train, y_train, X_val, y_val, epochs=epochs, device=device)


def predict_shallow_conv_net(
    result: DeepConvResult,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    return _predict_dl_model(result, X, device=device)
