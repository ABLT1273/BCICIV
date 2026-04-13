from __future__ import annotations

from pathlib import Path

from .runtime import get_script_root


def get_results_root() -> Path:
    root = get_script_root() / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_result_group_dir(group_name: str) -> Path:
    path = get_results_root() / group_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_dir() -> Path:
    path = get_script_root() / "model"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_hybrid_results_dir() -> Path:
    return get_result_group_dir("dim_reduction_hybrid_fbcsp")


def get_advanced_results_dir() -> Path:
    return get_result_group_dir("benchmark_trca_wavelet_cnn")


def get_results_index_path() -> Path:
    return get_results_root() / "README.md"
