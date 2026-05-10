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
    "nn_models_benchmark": ParadigmSpec(
        key="nn_models_benchmark",
        display_name="Neural Network Models Benchmark (TCN / ATCNet / DRSN / LaBraM)",
        description="统一基准测试四个深度学习模型。",
        components=(
            "framework.runtime.prepare_runtime_environment",
            "framework.data.load_subject_train_test",
            "TCN temporal convolutional network",
            "ATCNet attention temporal convolutional network",
            "DRSN dilated residual spatial network",
            "LaBraM transformer-based EEG model",
            "framework.plotting.plot_metric_bar",
            "framework.plotting.plot_aggregate_metric_bar",
        ),
        default_result_group="benchmark_nn_models",
        entry_script="pre-precess.py",
        module="paradigms.nn_models_benchmark",
    ),
    "loso_benchmark": ParadigmSpec(
        key="loso_benchmark",
        display_name="LOSO (Leave-One-Subject-Out) Benchmark",
        description="跨被试泛化评估: 每次留一个被试做测试, 其余被试 Session T 训练。支持 cross-session 和 same-session 两种模式。",
        components=(
            "framework.runtime.prepare_runtime_environment",
            "framework.cv_split.generate_loso_folds",
            "TCN / ATCNet / DRSN / LaBraM fold runners",
            "framework.plotting.plot_aggregate_metric_bar",
        ),
        default_result_group="loso_benchmark",
        entry_script="pre-precess.py",
        module="paradigms.loso_benchmark",
    ),
    "csp_lda_benchmark": ParadigmSpec(
        key="csp_lda_benchmark",
        display_name="CSP + LDA Baseline",
        description="经典 CSP 空间滤波 + LDA 分类器作为传统方法基线。同时提供 CSP+SVM 变体。",
        components=(
            "framework.runtime.prepare_runtime_environment",
            "framework.data.load_subject_train_test",
            "MNE CSP spatial filter",
            "sklearn LDA / SVM classifier",
        ),
        default_result_group="benchmark_csp_lda",
        entry_script="pre-precess.py",
        module="paradigms.csp_lda_benchmark",
    ),
    "fbcsp_dfbcsp_benchmark": ParadigmSpec(
        key="fbcsp_dfbcsp_benchmark",
        display_name="FBCSP + DFBCSP Benchmark",
        description="滤波器组 CSP 及其判别式变体 DFBCSP 的对比基准。",
        components=(
            "framework.runtime.prepare_runtime_environment",
            "framework.data.load_subject_train_test",
            "models.FBCSP (FilterBank + OVR_FBCSP_Ensemble)",
            "models.DFBCSP (OVR_DFBCSP_Ensemble)",
        ),
        default_result_group="benchmark_fbcsp_dfbcsp",
        entry_script="pre-precess.py",
        module="paradigms.fbcsp_dfbcsp_benchmark",
    ),
}


def load_paradigm_module(paradigm_key: str):
    """按注册表动态加载范式模块。"""

    spec = PARADIGM_REGISTRY[paradigm_key]
    return import_module(spec.module)
