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


def get_paradigm_result_dir(paradigm: str, group_name: str) -> Path:
    """Return a results path scoped to a data-split paradigm.

    Paradigms:
        within_subject  – T→E, train on session T, test on session E
        loso            – Leave-One-Subject-Out cross-subject
    """
    path = get_results_root() / paradigm / group_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_dir() -> Path:
    path = get_script_root() / "model"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_param_dir() -> Path:
    path = get_script_root() / "model_param"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_index_path() -> Path:
    return get_results_root() / "README.md"
