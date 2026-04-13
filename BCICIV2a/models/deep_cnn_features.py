from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class EEGNetFeatureExtractor(nn.Module):
    """
    一个更接近 EEGNet 思路的轻量网络。

    结构上包含：
    1. 时间卷积：先学节律/相位相关的时间滤波器
    2. 深度可分离空间卷积：在每个时间滤波器上学习跨通道空间模式
    3. 可分离卷积块：继续压缩时序信息并生成紧凑 embedding

    它不是为了追求 SOTA，而是为了：
    - 在 CPU 上也能较快训练；
    - 给当前实验提供一个“端到端深度特征”分支；
    - 尽量贴近 EEGNet 这种 EEG 任务里常见的 inductive bias；
    - 能导出倒数第二层 embedding，用来和 TRCA / Wavelet 特征做对比。
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        pointwise_filters: int = 32,
        embedding_dim: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                temporal_filters * depth_multiplier,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                temporal_filters * depth_multiplier,
                temporal_filters * depth_multiplier,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=temporal_filters * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(
                temporal_filters * depth_multiplier,
                pointwise_filters,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(pointwise_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flattened_dim = self._forward_backbone(dummy).flatten(1).shape[1]

        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, embedding_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._forward_backbone(x)
        embedding = self.embedding_layer(x)
        if return_features:
            return embedding
        return self.classifier(embedding)


@dataclass
class DeepFeatureResult:
    model: EEGNetFeatureExtractor
    train_mean: np.ndarray
    train_std: np.ndarray
    label_values: np.ndarray
    best_val_accuracy: float


def _normalize_eeg(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用训练集统计量做 z-score 归一化。

    这里沿着 trial 维度统计每个通道、每个时间点的均值与标准差，
    让 CNN 更容易收敛。
    """

    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    if std is None:
        std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def train_tiny_eeg_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 4,
    device: str | None = None,
) -> DeepFeatureResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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

    model = EEGNetFeatureExtractor(
        n_channels=X_train.shape[1],
        n_samples=X_train.shape[2],
        n_classes=len(label_values),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )
    criterion = nn.CrossEntropyLoss()

    best_state_dict = None
    best_val_accuracy = -1.0
    no_improvement_epochs = 0

    X_val_tensor = torch.from_numpy(X_val_norm).to(device)
    y_val_tensor = torch.from_numpy(y_val_idx).to(device)

    for _ in range(epochs):
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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            no_improvement_epochs += 1

        scheduler.step()
        if no_improvement_epochs >= early_stop_patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return DeepFeatureResult(
        model=model,
        train_mean=train_mean,
        train_std=train_std,
        label_values=label_values,
        best_val_accuracy=best_val_accuracy,
    )


def predict_tiny_eeg_cnn(
    result: DeepFeatureResult,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_norm, _, _ = _normalize_eeg(X, result.train_mean, result.train_std)

    result.model.eval()
    result.model.to(device)
    with torch.no_grad():
        logits = result.model(torch.from_numpy(X_norm).to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()
    return result.label_values[predictions]


def extract_tiny_eeg_cnn_features(
    result: DeepFeatureResult,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_norm, _, _ = _normalize_eeg(X, result.train_mean, result.train_std)

    result.model.eval()
    result.model.to(device)
    with torch.no_grad():
        features = result.model(
            torch.from_numpy(X_norm).to(device),
            return_features=True,
        )
    return features.cpu().numpy().astype(np.float32)
