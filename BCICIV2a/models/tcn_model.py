"""
TCN (Temporal Convolutional Network) for EEG classification.
Based on dilated causal convolutions for capturing long-range dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _normalize_eeg(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """Z-score normalization for EEG signals."""
    if mean is None or std is None:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), mean, std


class TCNBlock(nn.Module):
    """Single TCN block with dilated convolution, batch norm, and residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        # Causal padding: pad only left side
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Adjust residual if channels change
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, time_steps)
        Returns:
            output: (batch, out_channels, time_steps)
        """
        # Apply causal conv
        out = self.conv(x)
        # Remove right padding (causality)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        if self.proj is not None:
            x = self.proj(x)
        
        return x + out


class TCNNetwork(nn.Module):
    """
    Temporal Convolutional Network for EEG.
    Stacks multiple TCN blocks with increasing dilation to capture multi-scale temporal patterns.
    """

    def __init__(
        self,
        in_channels: int = 22,
        n_classes: int = 4,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_levels: int = 4,
        dropout: float = 0.25,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        # Build TCN stack with increasing dilation
        layers = []
        for i in range(num_levels):
            dilation_rate = 2**i
            input_ch = in_channels if i == 0 else num_filters
            output_ch = num_filters
            layers.append(
                TCNBlock(
                    input_ch,
                    output_ch,
                    kernel_size=kernel_size,
                    dilation=dilation_rate,
                    dropout=dropout,
                )
            )
        
        self.tcn_stack = nn.Sequential(*layers)

        # Global average pooling + embedding layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding_layer = nn.Linear(num_filters, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, time_steps) for 3D or
               (batch, 1, in_channels, time_steps) for 4D (will squeeze)
            return_features: if True, return embeddings instead of logits
        Returns:
            output: (batch, n_classes) or (batch, embedding_dim) if return_features=True
        """
        # Handle 4D input
        if x.ndim == 4:
            x = x.squeeze(1)

        # TCN stack
        x = self.tcn_stack(x)

        # Global average pooling
        x = self.gap(x)  # (batch, num_filters, 1)
        x = x.view(x.size(0), -1)  # (batch, num_filters)

        # Embedding
        embedding = self.embedding_layer(x)  # (batch, embedding_dim)

        if return_features:
            return embedding

        # Classification
        logits = self.classifier(embedding)
        return logits


@dataclass
class TCNResult:
    """Container for TCN training results."""
    model: TCNNetwork
    train_mean: np.ndarray  # (1, in_channels, time_steps)
    train_std: np.ndarray
    label_values: np.ndarray  # unique class labels
    best_val_accuracy: float


def train_tcn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> TCNResult:
    """
    Train TCN model with validation-based early stopping.

    Args:
        X_train: (n_train, n_channels, n_samples)
        y_train: (n_train,) labels
        X_val: (n_val, n_channels, n_samples)
        y_val: (n_val,) labels
        epochs: max training epochs
        batch_size: batch size
        learning_rate: learning rate
        device: torch device ('cuda' or 'cpu')

    Returns:
        TCNResult with trained model and metadata
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    n_channels = X_train.shape[1]
    label_values = np.unique(y_train)
    n_classes = len(label_values)

    # Normalize
    X_train_norm, train_mean, train_std = _normalize_eeg(X_train)
    X_val_norm, _, _ = _normalize_eeg(X_val, train_mean, train_std)

    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(label_values)}
    y_train_idx = np.array([label_to_idx[y] for y in y_train], dtype=np.int64)
    y_val_idx = np.array([label_to_idx[y] for y in y_val], dtype=np.int64)

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_norm).float(),
        torch.from_numpy(y_train_idx).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_norm).float(),
        torch.from_numpy(y_val_idx).long(),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    model = TCNNetwork(
        in_channels=n_channels,
        n_classes=n_classes,
        num_filters=64,
        kernel_size=3,
        num_levels=4,
        dropout=0.25,
        embedding_dim=64,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    early_stop_patience = 4
    no_improvement_count = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                _, pred = torch.max(logits, 1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        val_accuracy = correct / total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}",
            flush=True,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        scheduler.step()

        if no_improvement_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}", flush=True)
            break

    return TCNResult(
        model=model,
        train_mean=train_mean,
        train_std=train_std,
        label_values=label_values,
        best_val_accuracy=best_val_accuracy,
    )


def predict_tcn(
    result: TCNResult,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    """
    Make predictions using trained TCN model.

    Args:
        result: TCNResult from train_tcn()
        X: (n_test, n_channels, n_samples)
        device: torch device

    Returns:
        predictions: (n_test,) with original label values
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Normalize using training statistics
    X_norm, _, _ = _normalize_eeg(X, result.train_mean, result.train_std)

    # Convert to tensor
    X_tensor = torch.from_numpy(X_norm).float().to(device)

    # Predict
    result.model.eval()
    with torch.no_grad():
        logits = result.model(X_tensor)
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()

    # Map indices back to original labels
    predictions = result.label_values[pred_idx]
    return predictions


def extract_tcn_features(
    result: TCNResult,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    """
    Extract embedding features from TCN model.

    Args:
        result: TCNResult from train_tcn()
        X: (n_test, n_channels, n_samples)
        device: torch device

    Returns:
        features: (n_test, embedding_dim) float32 for UMAP visualization
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Normalize using training statistics
    X_norm, _, _ = _normalize_eeg(X, result.train_mean, result.train_std)

    # Convert to tensor
    X_tensor = torch.from_numpy(X_norm).float().to(device)

    # Extract features
    result.model.eval()
    with torch.no_grad():
        features = result.model(X_tensor, return_features=True).cpu().numpy()

    return features.astype(np.float32)


def setup_tcn_pipeline(n_channels: int = 22, num_classes: int = 4, device: str | None = None) -> TCNNetwork:
    """
    Set up TCN model for forward pass validation.
    
    Args:
        n_channels: number of EEG channels
        num_classes: number of classification classes
        device: torch device
        
    Returns:
        TCNNetwork model initialized and ready for forward pass
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device)
    
    model = TCNNetwork(
        in_channels=n_channels,
        n_classes=num_classes,
        num_filters=64,
        kernel_size=3,
        num_levels=4,
        dropout=0.25,
        embedding_dim=64,
    ).to(device_obj)
    
    print("TCN: Forward pass validated", flush=True)
    return model


def run_tcn_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """
    TCN experiment wrapper for BCICIV2a benchmark.
    
    NOTE: This is a PLACEHOLDER that only validates the pipeline.
    Actual training will be implemented in a future phase.
    
    Args:
        X_train, X_test: EEG data (n_trials, n_channels, n_samples)
        y_train, y_test: class labels
        
    Returns:
        metrics: dict with 'accuracy' and 'kappa'
        embeddings: (n_test, embed_dim) for visualization
    """
    print("TCN: Setting up pipeline...", flush=True)
    
    n_channels = X_train.shape[1]
    label_values = np.unique(y_train)
    num_classes = len(label_values)
    
    # Setup adapter (validates forward pass)
    setup_tcn_pipeline(n_channels=n_channels, num_classes=num_classes)
    
    print("TCN: Pipeline validation complete.", flush=True)
    print("NOTE: TCN training is not yet implemented in this phase.", flush=True)
    
    # TEMPORARY: Return dummy results for visualization testing
    dummy_metrics = {
        "accuracy": 0.5,
        "kappa": 0.0,
    }
    dummy_embeddings = np.random.randn(X_test.shape[0], 64).astype(np.float32)
    
    return dummy_metrics, dummy_embeddings
