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


def get_hybrid_results_dir() -> Path:
    return get_result_group_dir("hybrid_reduction")


def get_advanced_results_dir() -> Path:
    return get_result_group_dir("advanced_benchmark")


def get_results_index_path() -> Path:
    return get_results_root() / "README.md"
