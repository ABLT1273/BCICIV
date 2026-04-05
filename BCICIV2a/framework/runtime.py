from __future__ import annotations

import os
from pathlib import Path


def get_script_root() -> Path:
    """返回 BCICIV2a 目录。"""

    return Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """返回 test_newPyEnv 根目录。"""

    return Path(__file__).resolve().parents[3]


def prepare_runtime_environment() -> None:
    """
    统一处理 MNE / MOABB / matplotlib 的缓存与配置目录。

    这个函数专门给所有 BCICIV2a 实验脚本共用，避免每个脚本各自维护一套环境初始化逻辑。
    """

    script_root = get_script_root()
    project_root = get_project_root()
    runtime_root = script_root / ".runtime_cache"
    runtime_root.mkdir(parents=True, exist_ok=True)

    mne_home = runtime_root / ".mne"
    mpl_config = runtime_root / ".matplotlib"
    existing_data_candidates = [
        project_root / ".mne_data",
        project_root / "mne_data",
    ]
    mne_data = next(
        (candidate for candidate in existing_data_candidates if candidate.exists()),
        runtime_root / "mne_data",
    )

    for directory in (mne_home, mpl_config, mne_data):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(runtime_root)
    os.environ["MNE_HOME"] = str(mne_home)
    os.environ["MNE_DATA"] = str(mne_data)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(mne_data)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)

