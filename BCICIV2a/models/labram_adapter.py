"""
LaBraM-Large adapter for BCICIV2a classification.
Uses official TorchEEG implementation: https://torcheeg.readthedocs.io/
Constructs pipeline but does NOT execute training - only validates forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class LabramAdapterResult:
    """Container for LaBraM adapter results."""
    model: nn.Module
    train_mean: np.ndarray
    train_std: np.ndarray
    label_values: np.ndarray
    # Note: best_val_accuracy would only be set after actual training


def _normalize_eeg(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """Z-score normalization for EEG signals."""
    if mean is None or std is None:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), mean, std


class LabramAdapter:
    """
    Lightweight wrapper for TorchEEG's official LaBraM-Large model.
    
    IMPORTANT: This class validates the pipeline but does NOT execute training.
    It only performs forward pass verification to confirm compatibility.
    
    Key parameters (from TorchEEG):
    - chunk_size: 1600 (EEG signal length in samples)
    - patch_size: 200 (patch temporal length)
    - out_chans: 8 (temporal conv output channels)
    - embed_dim: 200 (embedding dimension)
    - depth: 12 (transformer layers)
    - num_heads: 10 (attention heads)
    """

    def __init__(
        self,
        num_classes: int = 4,
        chunk_size: int = 1600,
        patch_size: int = 200,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
    ):
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.model = None

    def initialize_model(self, device: str | None = None):
        """
        Initialize LaBraM model from TorchEEG.
        
        PREREQUISITE: User must install TorchEEG:
            pip install torcheeg
        """
        try:
            from torcheeg.models import LaBraM
        except ImportError as e:
            raise ImportError(
                "TorchEEG not found. Please install it with:\n"
                "  pip install torcheeg\n"
                "See: https://torcheeg.readthedocs.io/"
            ) from e

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_obj = torch.device(device)

        # Initialize LaBraM model
        # Using base_patch200_200 preset for BCICIV2a compatibility
        try:
            self.model = LaBraM.base_patch200_200(num_classes=self.num_classes)
        except Exception:
            # Fallback to manual initialization
            self.model = LaBraM(
                chunk_size=self.chunk_size,
                patch_size=self.patch_size,
                out_chans=8,
                num_classes=self.num_classes,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                use_mean_pooling=True,
                use_abs_pos_emb=True,
            )

        self.model = self.model.to(device_obj)
        return self.model

    def forward(self, X: np.ndarray, electrodes: list[str] | None = None, device: str | None = None):
        """
        Forward pass for validation (pipeline verification only).
        
        Args:
            X: (batch, 22_channels, chunk_size) or reshaped for TorchEEG
            electrodes: optional electrode names list
            device: torch device
            
        Returns:
            logits: (batch, num_classes) from model output
            
        NOTE: This only verifies the forward pass works correctly.
              Actual training is NOT performed.
        """
        if self.model is None:
            self.initialize_model(device)

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_obj = torch.device(device)

        # Reshape BCICIV2a format to TorchEEG format
        # Input: (batch, 22, ~1500-2000 samples)
        # Expected by LaBraM: (batch, num_channels, num_patches, patch_size)
        # where num_patches = chunk_size // patch_size = 1600 // 200 = 8

        if X.ndim == 3:
            batch_size, n_channels, n_samples = X.shape
            
            # Pad or truncate to chunk_size if needed
            if n_samples < self.chunk_size:
                pad_size = self.chunk_size - n_samples
                X_padded = np.pad(X, ((0, 0), (0, 0), (0, pad_size)), mode="constant")
            else:
                X_padded = X[:, :, : self.chunk_size]
            
            # Reshape into patches: (batch, n_channels, chunk_size) -> (batch, n_channels, num_patches, patch_size)
            num_patches = self.chunk_size // self.patch_size
            X_reshaped = X_padded.reshape(batch_size, n_channels, num_patches, self.patch_size)
        else:
            X_reshaped = X

        # Convert to tensor
        X_tensor = torch.from_numpy(X_reshaped).float().to(device_obj)

        # Forward pass (no gradients for validation)
        self.model.eval()
        with torch.no_grad():
            # Optional: provide electrodes info if available
            if electrodes is not None:
                logits = self.model(X_tensor, electrodes=electrodes)
            else:
                logits = self.model(X_tensor)

        return logits


def setup_labram_pipeline(
    n_channels: int = 22,
    num_classes: int = 4,
    device: str | None = None,
) -> LabramAdapter:
    """
    Set up LaBraM pipeline for forward pass validation.
    
    This function:
    1. Checks TorchEEG installation
    2. Initializes LaBraM-Large model
    3. Validates forward pass with dummy input
    
    Does NOT perform training.
    
    Args:
        n_channels: number of EEG channels (default 22 for BCICIV2a)
        num_classes: number of classification classes (default 4 for BCICIV2a)
        device: torch device ('cuda' or 'cpu')
        
    Returns:
        adapter: LabramAdapter instance ready for forward pass
        
    Example:
        adapter = setup_labram_pipeline()
        X_dummy = np.random.randn(2, 22, 1600)
        output = adapter.forward(X_dummy)  # Verify forward pass works
    """
    adapter = LabramAdapter(num_classes=num_classes)
    adapter.initialize_model(device)

    # Validate forward pass with dummy input
    print("LaBraM: Validating forward pass with dummy input...", flush=True)
    X_dummy = np.random.randn(2, n_channels, 1600).astype(np.float32)
    
    try:
        output = adapter.forward(X_dummy, device=device)
        print(f"LaBraM: Forward pass validated. Output shape: {output.shape}", flush=True)
        assert output.shape == (2, num_classes), f"Unexpected output shape: {output.shape}"
        print("LaBraM: Pipeline setup complete (no training performed)", flush=True)
    except Exception as e:
        raise RuntimeError(
            f"LaBraM forward pass validation failed: {e}\n"
            "Please ensure TorchEEG is correctly installed."
        ) from e

    return adapter


def run_labram_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """
    LaBraM experiment wrapper for BCICIV2a benchmark.
    
    IMPORTANT: This is a PLACEHOLDER that only validates the pipeline.
    Actual training will be implemented in a future phase.
    
    Current behavior:
    1. Initializes LaBraM model
    2. Validates forward pass
    3. Returns dummy metrics and random embeddings (for visualization testing)
    4. Raises NotImplementedError for actual training
    
    Args:
        X_train, X_test: EEG data (n_trials, n_channels, n_samples)
        y_train, y_test: class labels
        
    Returns:
        metrics: dict with 'accuracy' and 'kappa'
        embeddings: (n_test, embed_dim) for UMAP visualization
    """
    print("LaBraM: Setting up pipeline...", flush=True)
    
    n_channels = X_train.shape[1]
    label_values = np.unique(y_train)
    num_classes = len(label_values)

    # Setup adapter (validates forward pass)
    try:
        adapter = setup_labram_pipeline(
            n_channels=n_channels,
            num_classes=num_classes,
        )
    except ImportError as e:
        raise ImportError(
            f"Cannot initialize LaBraM: {e}\n"
            "Please install TorchEEG:\n"
            "  pip install torcheeg"
        ) from e

    print("LaBraM: Pipeline validation complete.", flush=True)
    print(
        "NOTE: LaBraM training is not yet implemented in this phase.\n"
        "Only forward pass has been validated for pipeline compatibility.",
        flush=True,
    )

    # TEMPORARY: Return dummy results for visualization testing
    # (This will be replaced with actual training + feature extraction in future phase)
    print("LaBraM: Returning dummy metrics and embeddings for visualization testing...", flush=True)
    
    dummy_metrics = {
        "accuracy": 0.5,  # Placeholder
        "kappa": 0.0,  # Placeholder
    }
    dummy_embeddings = np.random.randn(X_test.shape[0], 200).astype(np.float32)

    return dummy_metrics, dummy_embeddings
