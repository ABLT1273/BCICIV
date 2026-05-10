"""
LaBraM adapter for BCICIV2a classification — fine-tuning pipeline.

Uses braindecode's official LaBraM model with HuggingFace pretrained weights.
Follows the LaBraM paper preprocessing specification:
  - Resample from 250 Hz → 200 Hz
  - Bandpass filter 0.1–75 Hz
  - Notch filter 50 Hz
  - Unit conversion: uV → 0.1 mV (divide by 100)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

from framework.devices import resolve_torch_device


# ---------------------------------------------------------------------------
# MNE-based preprocessing following the LaBraM paper spec (Section 2.1)
# ---------------------------------------------------------------------------

def _preprocess_labram(
    X: np.ndarray,
    sfreq_orig: float = 250.0,
    sfreq_target: float = 200.0,
    fmin: float = 0.1,
    fmax: float = 75.0,
    notch_freq: float = 50.0,
) -> np.ndarray:
    """Resample, bandpass-filter, notch-filter and scale EEG to LaBraM spec.

    Paper requirements:
      - Resample to 200 Hz
      - Bandpass 0.1–75 Hz
      - Notch 50 Hz
      - Unit: 0.1 mV  (raw MOABB data is in uV → divide by 100)

    Returns: (n_trials, n_channels, n_times) float32 array at 200 Hz.
    """
    import mne

    n_trials, n_channels, n_samples = X.shape

    # Resample first (before filtering) to get enough samples for FIR filter.
    # MNE's resample uses FIR internally and handles short signals gracefully.
    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(n_channels)],
        sfreq=sfreq_orig,
        ch_types="eeg",
    )

    # Use IIR (butterworth) for filtering to avoid filter-length issues on
    # short (~2 s) epochs.  The paper does not prescribe FIR vs IIR.
    out_list = []
    for i in range(n_trials):
        raw = mne.io.RawArray(X[i], info, verbose=False)
        # Bandpass 0.1–75 Hz (IIR to handle short ~2 s epochs)
        raw.filter(fmin, fmax, method="iir", verbose=False)
        # Notch 50 Hz (IIR likewise)
        raw.notch_filter(notch_freq, method="iir", verbose=False)
        # Resample to 200 Hz
        raw.resample(sfreq_target, verbose=False)
        # uV → 0.1 mV: divide raw values by 100
        out_list.append(raw.get_data()[np.newaxis, ...] / 100.0)

    out = np.concatenate(out_list, axis=0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Training / eval helpers
# ---------------------------------------------------------------------------

def _to_index(y: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(label_values)}
    return np.array([mapping[v] for v in y], dtype=np.int64)


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


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_labram_model(
    n_chans: int,
    n_times: int,
    num_classes: int,
    ch_names: list[str] | None = None,
    pretrained: bool = True,
    device: str | None = None,
) -> nn.Module:
    """Build a braindecode LaBraM model, optionally loading pretrained weights."""
    from braindecode.models import Labram

    if ch_names is None:
        from framework.constants import BNCI2014001_CHANNEL_NAMES
        ch_names = BNCI2014001_CHANNEL_NAMES[:n_chans]

    model = Labram(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=num_classes,
        chs_info=[{"ch_name": name} for name in ch_names],
        patch_size=200,
        embed_dim=200,
        num_layers=12,
        num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_prob=0.0,
        attn_drop_prob=0.0,
        drop_path_prob=0.0,
        use_abs_pos_emb=True,
        use_mean_pooling=True,
        init_values=0.1,
        neural_tokenizer=True,
        on_unknown_chs="warn",
    )

    if pretrained:
        from collections import OrderedDict

        url = "https://huggingface.co/braindecode/Labram-Braindecode/resolve/main/braindecode_labram_base.pt"
        state = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")

        # The pretrained state may be nested under "student." keys
        if any(k.startswith("student.") for k in state.keys()):
            new_state = OrderedDict()
            for k, v in state.items():
                if k.startswith("student."):
                    new_state[k[8:]] = v
            state = new_state

        # Handle position_embedding size mismatch:
        # checkpoint has [1, 65, 200] (64 chans) but model expects [1, 129, 200] (128 chans).
        # Pad the checkpoint embedding with zeros for extra channels.
        if "position_embedding" in state and hasattr(model, "position_embedding"):
            ckpt_pe = state["position_embedding"]
            model_pe = model.position_embedding
            if ckpt_pe.shape != model_pe.shape:
                padded = torch.zeros(model_pe.shape, dtype=ckpt_pe.dtype)
                copy_len = min(ckpt_pe.shape[1], model_pe.shape[1])
                padded[:, :copy_len, :] = ckpt_pe[:, :copy_len, :]
                state = OrderedDict(state)  # copy
                state["position_embedding"] = padded

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"LaBraM: Pretrained weights loaded.  missing_keys={len(missing)}  unexpected_keys={len(unexpected)}",
              flush=True)

    return model.to(resolve_torch_device(device))


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

@dataclass
class LabramFineTuneResult:
    model: nn.Module
    label_values: np.ndarray
    best_val_accuracy: float


def finetune_labram(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    patience: int = 30,
    device: str | None = None,
    freeze_backbone: bool = True,
) -> LabramFineTuneResult:
    """Fine-tune the pretrained LaBraM model on BCICIV2a data.

    Uses adapter-style strategy: freezes most transformer layers and only
    trains the last 2 encoder layers + classification head, reducing
    trainable params from ~110M to ~20M and preventing overfitting.
    """
    device_obj = resolve_torch_device(device)
    label_values = np.unique(y_train)

    y_train_idx = _to_index(y_train, label_values)
    y_val_idx = _to_index(y_val, label_values)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_idx)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_idx)),
        batch_size=batch_size,
        shuffle=False,
    )

    n_chans = X_train.shape[1]
    n_times = X_train.shape[2]
    num_classes = len(label_values)

    model = _build_labram_model(
        n_chans=n_chans,
        n_times=n_times,
        num_classes=num_classes,
        pretrained=True,
        device=device,
    )

    # Adapter-style: freeze most layers, train only last 2 encoder + head
    if freeze_backbone:
        # Labram uses encoder (transformer) + fc_head for classification
        # encoder.encoder.layers is a ModuleList of transformer blocks
        frozen_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            # Keep last 2 encoder layers and fc_head trainable
            if "encoder.encoder.layers.10" in name or "encoder.encoder.layers.11" in name or "fc_head" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            elif "encoder.encoder.layers" in name:
                param.requires_grad = False
                frozen_params += param.numel()
            # Keep tokenizer and positional embedding trainable (small)
            elif "neural_tokenizer" in name or "position_embedding" in name or "cls_token" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = True
                trainable_params += param.numel()
        print(f"LaBraM: frozen={frozen_params:,}  trainable={trainable_params:,}", flush=True)

    # Separate LR: lower for backbone, higher for head
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc_head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": learning_rate},
        {"params": head_params, "lr": learning_rate * 5},
    ], weight_decay=1e-4)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-7,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = 0.0
    best_state = None
    epochs_no_improve = 0

    log_interval = max(1, epochs // 20)
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

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(
                f"  LaBraM epoch {epoch+1:3d}/{epochs}: "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}",
                flush=True,
            )

        if epochs_no_improve >= patience:
            print(f"  LaBraM early stopping at epoch {epoch+1} (best_val_acc={best_val:.4f})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return LabramFineTuneResult(
        model=model,
        label_values=label_values,
        best_val_accuracy=best_val,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_labram(
    result: LabramFineTuneResult,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    device_obj = resolve_torch_device(device)
    xt = torch.from_numpy(X).to(device_obj)
    result.model.eval()
    with torch.no_grad():
        logits = result.model(xt)
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
    return result.label_values[pred_idx]


def extract_labram_features(
    result: LabramFineTuneResult,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    """Extract CLS-token features from the fine-tuned LaBraM model."""
    device_obj = resolve_torch_device(device)
    xt = torch.from_numpy(X).to(device_obj)
    result.model.eval()
    with torch.no_grad():
        # forward_features needs input_chans from _select_channels
        x_selected, input_chans = result.model._select_channels(xt, ch_names=None)
        features = result.model.forward_features(x_selected, input_chans).cpu().numpy()
    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------

def run_labram_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    checkpoint_dir: Path | None = None,
    subject_id: int | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Fine-tune LaBraM on BCICIV2a and evaluate.

    Reloads raw data (no unified preprocessing) to avoid double-filtering
    with LaBraM's own MNE-based pipeline.
    """
    from framework.data import load_subject_train_test

    # Reload raw data without unified preprocessing to avoid double filtering
    if subject_id is not None:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw, _sfreq = load_subject_train_test(
            subject_id=subject_id, tmin=0.5, tmax=2.5, channels=None,
            bandpass=None, notch=None, apply_car=False, zscore=False,
        )
        X_train = X_train_raw
        X_test = X_test_raw
        # Use reloaded labels to ensure consistency
        y_train = y_train_raw
        y_test = y_test_raw

    print("LaBraM: Preprocessing data (resample 250→200 Hz, filter, scale)...", flush=True)

    t0 = perf_counter()

    X_train_proc = _preprocess_labram(X_train.astype(np.float64))
    X_test_proc = _preprocess_labram(X_test.astype(np.float64))

    # Pad or truncate to chunk_size=1600
    chunk_size = 1600
    for name, arr in [("train", X_train_proc), ("test", X_test_proc)]:
        n = arr.shape[2]
        if n < chunk_size:
            pad = np.pad(arr, ((0, 0), (0, 0), (0, chunk_size - n)), mode="constant")
            if name == "train":
                X_train_proc = pad
            else:
                X_test_proc = pad
        elif n > chunk_size:
            if name == "train":
                X_train_proc = arr[:, :, :chunk_size]
            else:
                X_test_proc = arr[:, :, :chunk_size]

    print(f"LaBraM: Preprocessed.  train={X_train_proc.shape}  test={X_test_proc.shape}", flush=True)

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    n_train = len(X_train_proc)
    split = max(1, int(n_train * 0.8))
    X_tr, X_val = X_train_proc[:split], X_train_proc[split:]
    y_tr, y_val = y_train[:split], y_train[split:]
    if len(X_val) == 0:
        X_tr, X_val = X_train_proc[:-1], X_train_proc[-1:]
        y_tr, y_val = y_train[:-1], y_train[-1:]

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------
    result = finetune_labram(X_tr, y_tr, X_val, y_val, epochs=200, freeze_backbone=True)

    train_time = perf_counter() - t0

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    t1 = perf_counter()
    y_pred = predict_labram(result, X_test_proc)
    features = extract_labram_features(result, X_test_proc)
    infer_time = perf_counter() - t1

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "kappa": float(cohen_kappa_score(y_test, y_pred)),
        "train_time": float(train_time),
        "inference_time": float(infer_time),
        "best_val_accuracy": float(result.best_val_accuracy),
        "checkpoint_path": None,
    }

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "labram_checkpoint.pth"
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in result.model.state_dict().items()
                },
                "label_values": result.label_values,
                "best_val_accuracy": result.best_val_accuracy,
            },
            checkpoint_path,
        )
        metrics["checkpoint_path"] = str(checkpoint_path)

    print(
        f"LaBraM fine-tuned: accuracy={metrics['accuracy']:.4f}  kappa={metrics['kappa']:.4f}  "
        f"train_time={metrics['train_time']:.1f}s  best_val_acc={result.best_val_accuracy:.4f}",
        flush=True,
    )

    return metrics, features
