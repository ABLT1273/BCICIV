"""DRSN adapter using vendored official DRSN-CS PyTorch source."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset


def _normalize_eeg(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    if mean is None or std is None:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((X - mean) / std).astype(np.float32), mean, std


def _to_index(y: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(label_values)}
    return np.array([mapping[v] for v in y], dtype=np.int64)


def _load_official_module():
    from models import official_drsn

    return official_drsn


def _build_backbone(num_classes: int) -> nn.Module:
    official = _load_official_module()
    backbone = official.rsnet18()
    # Replace the original classifier so the official feature extractor remains intact.
    backbone.fc = nn.Identity()
    return backbone


class DRSNClassifier(nn.Module):
    """EEG adapter around the official 1-channel DRSN backbone."""

    def __init__(self, in_channels: int, n_classes: int, embedding_dim: int = 64):
        super().__init__()
        self.channel_projection = nn.Conv1d(in_channels, 1, kernel_size=1, bias=False)
        self.backbone = _build_backbone(n_classes)
        self.embedding = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        if x.ndim == 4:
            x = x.squeeze(1)
        x = self.channel_projection(x)
        features = self.backbone.conv1(x)
        features = self.backbone.conv2_x(features)
        features = self.backbone.conv3_x(features)
        features = self.backbone.conv4_x(features)
        features = self.backbone.conv5_x(features)
        features = self.backbone.avg_pool(features).view(features.size(0), -1)
        emb = self.embedding(features)
        if return_features:
            return emb
        return self.classifier(emb)


@dataclass
class DRSNResult:
    model: DRSNClassifier
    train_mean: np.ndarray
    train_std: np.ndarray
    label_values: np.ndarray
    best_val_accuracy: float


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(len(loader), 1)


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    running = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            running += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return running / max(len(loader), 1), correct / max(total, 1)


def train_drsn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> DRSNResult:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    label_values = np.unique(y_train)

    X_train_norm, train_mean, train_std = _normalize_eeg(X_train)
    X_val_norm, _, _ = _normalize_eeg(X_val, train_mean, train_std)

    y_train_idx = _to_index(y_train, label_values)
    y_val_idx = _to_index(y_val, label_values)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train_idx)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_norm), torch.from_numpy(y_val_idx)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = DRSNClassifier(in_channels=X_train.shape[1], n_classes=len(label_values)).to(device_obj)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for _ in range(epochs):
        _train_epoch(model, train_loader, optimizer, criterion, device_obj)
        _, val_acc = _eval_epoch(model, val_loader, criterion, device_obj)
        best_val = max(best_val, val_acc)

    return DRSNResult(model=model, train_mean=train_mean, train_std=train_std, label_values=label_values, best_val_accuracy=best_val)


def predict_drsn(result: DRSNResult, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X_norm, _, _ = _normalize_eeg(X, result.train_mean, result.train_std)
    xt = torch.from_numpy(X_norm).to(device_obj)
    result.model.eval()
    with torch.no_grad():
        logits = result.model(xt)
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
    return result.label_values[pred_idx]


def extract_drsn_features(result: DRSNResult, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X_norm, _, _ = _normalize_eeg(X, result.train_mean, result.train_std)
    xt = torch.from_numpy(X_norm).to(device_obj)
    result.model.eval()
    with torch.no_grad():
        features = result.model(xt, return_features=True).cpu().numpy()
    return features.astype(np.float32)


def setup_drsn_pipeline(n_channels: int = 22, num_classes: int = 4, device: str | None = None) -> DRSNClassifier:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DRSNClassifier(in_channels=n_channels, n_classes=num_classes).to(device_obj)
    model.eval()
    return model


def run_drsn_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    n_train = len(X_train)
    split = max(1, int(n_train * 0.8))
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]
    if len(X_val) == 0:
        X_tr, X_val = X_train[:-1], X_train[-1:]
        y_tr, y_val = y_train[:-1], y_train[-1:]

    t0 = perf_counter()
    result = train_drsn(X_tr, y_tr, X_val, y_val, epochs=1)
    train_time = perf_counter() - t0

    t1 = perf_counter()
    y_pred = predict_drsn(result, X_test)
    features = extract_drsn_features(result, X_test)
    infer_time = perf_counter() - t1

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "kappa": float(cohen_kappa_score(y_test, y_pred)),
        "train_time": float(train_time),
        "inference_time": float(infer_time),
        "best_val_accuracy": float(result.best_val_accuracy),
    }
    return metrics, features
