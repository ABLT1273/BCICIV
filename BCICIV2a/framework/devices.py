from __future__ import annotations

import torch


def resolve_torch_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return device if isinstance(device, torch.device) else torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")