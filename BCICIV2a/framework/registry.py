from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ParadigmSpec:
    key: str
    display_name: str
    description: str
    components: tuple[str, ...]
    default_result_group: str
    entry_script: str
    module: str


PARADIGM_REGISTRY = {
    "hybrid_fbcsp_umap": ParadigmSpec(
        key="hybrid_fbcsp_umap",
        display_name="Hybrid C3/C4 + FBCSP UMAP",
        description="融合 C3/C4 频域特征与 FBCSP 特征，再做 UMAP 降维。",
        components=(
            "framework.runtime.prepare_runtime_environment",
            "framework.data.load_subject_epochs",
            "C3/C4 frequency feature extractor",
            "FBCSP feature extractor",
            "UMAP / supervised UMAP reducer",
            "framework.plotting.plot_3d_embedding",
        ),
        default_result_group="dim_reduction_hybrid_fbcsp",
        entry_script="pre-precess.py",
        module="paradigms.hybrid_fbcsp_umap",
    ),
    "advanced_feature_benchmark": ParadigmSpec(
        key="advanced_feature_benchmark",
        display_name="TRCA / Wavelet / CNN Benchmark",
        description="统一比较 TRCA、小波特征和深度学习特征。",
        components=(
            "framework.runtime.prepare_runtime_environment",
            "framework.data.load_subject_train_test",
            "TRCA hybrid classifier",
            "PyWavelets feature extractor",
            "EEGNet-style CNN feature extractor",
            "framework.plotting.plot_metric_bar",
            "framework.plotting.plot_aggregate_metric_bar",
        ),
        default_result_group="benchmark_trca_wavelet_cnn",
        entry_script="pre-precess.py",
        module="paradigms.advanced_benchmark",
    ),
}


def load_paradigm_module(paradigm_key: str):
    """按注册表动态加载范式模块。"""

    spec = PARADIGM_REGISTRY[paradigm_key]
    return import_module(spec.module)
