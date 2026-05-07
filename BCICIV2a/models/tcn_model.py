"""TCN adapter using official pytorch-tcn backbone."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

from framework.devices import resolve_torch_device


class TCNClassifier(nn.Module):
    """Classification wrapper around official pytorch-tcn.TCN."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 5,
        dropout: float = 0.25,
        embedding_dim: int = 64,
    ):
        super().__init__()
        from models.official_tcn import TCN

        channels = num_channels or [32, 64, 128]
        self.backbone = TCN(
            num_inputs=in_channels,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
            use_norm="weight_norm",
            activation="relu",
            input_shape="NCL",
            output_projection=None,
            output_activation=None,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Linear(channels[-1], embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        if x.ndim == 4:
            x = x.squeeze(1)
        features = self.backbone(x)
        features = self.pool(features).squeeze(-1)
        emb = self.embedding(features)
        if return_features:
            return emb
        return self.classifier(emb)


@dataclass
class TCNResult:
    model: TCNClassifier
    train_mean: np.ndarray
    train_std: np.ndarray
    label_values: np.ndarray
    best_val_accuracy: float


def _normalize_eeg(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    if mean is None or std is None:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((X - mean) / std).astype(np.float32), mean, std


def _to_index(y: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(label_values)}
    return np.array([label_to_idx[v] for v in y], dtype=np.int64)


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


def train_tcn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 20,
    device: str | None = None,
) -> TCNResult:
    device_obj = resolve_torch_device(device)
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

    model = TCNClassifier(in_channels=X_train.shape[1], n_classes=len(label_values)).to(device_obj)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None
    epochs_no_improve = 0
    for epoch in range(epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device_obj)
        val_loss, val_acc = _eval_epoch(model, val_loader, criterion, device_obj)
        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  TCN epoch {epoch+1:3d}/{epochs}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", flush=True)

        if epochs_no_improve >= patience:
            print(f"  TCN early stopping at epoch {epoch+1} (best_val_acc={best_val:.4f})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TCNResult(model=model, train_mean=train_mean, train_std=train_std, label_values=label_values, best_val_accuracy=best_val)


def predict_tcn(result: TCNResult, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device_obj = resolve_torch_device(device)
    X_norm, _, _ = _normalize_eeg(X, result.train_mean, result.train_std)
    xt = torch.from_numpy(X_norm).to(device_obj)
    result.model.eval()
    with torch.no_grad():
        logits = result.model(xt)
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
    return result.label_values[pred_idx]


def extract_tcn_features(result: TCNResult, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device_obj = resolve_torch_device(device)
    X_norm, _, _ = _normalize_eeg(X, result.train_mean, result.train_std)
    xt = torch.from_numpy(X_norm).to(device_obj)
    result.model.eval()
    with torch.no_grad():
        features = result.model(xt, return_features=True).cpu().numpy()
    return features.astype(np.float32)


def setup_tcn_pipeline(n_channels: int = 22, num_classes: int = 4, device: str | None = None) -> TCNClassifier:
    device_obj = resolve_torch_device(device)
    model = TCNClassifier(in_channels=n_channels, n_classes=num_classes).to(device_obj)
    model.eval()
    return model


def run_tcn_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    checkpoint_dir: Path | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    n_train = len(X_train)
    split = max(1, int(n_train * 0.8))
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]
    if len(X_val) == 0:
        X_tr, X_val = X_train[:-1], X_train[-1:]
        y_tr, y_val = y_train[:-1], y_train[-1:]

    t0 = perf_counter()
    result = train_tcn(X_tr, y_tr, X_val, y_val, epochs=100)
    train_time = perf_counter() - t0

    t1 = perf_counter()
    y_pred = predict_tcn(result, X_test)
    features = extract_tcn_features(result, X_test)
    infer_time = perf_counter() - t1

    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "tcn_checkpoint.pth"
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in result.model.state_dict().items()
                },
                "train_mean": result.train_mean,
                "train_std": result.train_std,
                "label_values": result.label_values,
                "best_val_accuracy": result.best_val_accuracy,
            },
            checkpoint_path,
        )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "kappa": float(cohen_kappa_score(y_test, y_pred)),
        "train_time": float(train_time),
        "inference_time": float(infer_time),
        "best_val_accuracy": float(result.best_val_accuracy),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    return metrics, features
