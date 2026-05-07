"""
Neural Network Models Benchmark paradigm for BCICIV2a.

Unified benchmark for four deep learning models:
1. TCN (Temporal Convolutional Network)
2. ATCNet (Attention Temporal Convolutional Network)
3. DRSN (Dilated Residual Spatial Network)
4. LaBraM-Large (Transformer-based EEG model with TorchEEG)

Pipeline:
- Load subject data from BCICIV2a dataset
- Initialize each model with standard hyperparameters
- Execute training (when implemented) or forward pass validation
- Collect metrics (accuracy, kappa)
- Visualize comparative performance
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from framework.data import load_subject_train_test
from framework.paths import get_model_param_dir, get_result_group_dir
from framework.plotting import (
    plot_aggregate_metric_bar,
    plot_comparison_bar_subject_grid_from_data,
    plot_metric_bar,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    model_name: str
    accuracy: float
    kappa: float
    balanced_accuracy: float = 0.0
    confusion_matrix: list[list[int]] | None = None
    train_time: float = 0.0
    inference_time: float = 0.0
    checkpoint_path: str | None = None


@dataclass
class NNModelsBenchmarkConfig:
    subject_id: int = 1
    output_dir: Path | None = None
    all_subjects: bool = False
    show: bool = False
    tmin: float = 0.5
    tmax: float = 2.5
    shuffle_labels: bool = False


def _model_metrics_to_dict(metrics_list: list[ModelMetrics]) -> dict[str, dict[str, float]]:
    """Convert list of ModelMetrics to dict format compatible with plotting functions."""
    result: dict[str, dict[str, float]] = {}
    for m in metrics_list:
        result[m.model_name] = {
            "accuracy": m.accuracy,
            "kappa": m.kappa,
            "balanced_accuracy": m.balanced_accuracy,
            "train_time": m.train_time,
            "inference_time": m.inference_time,
        }
    return result


def summarize_all_subjects(
    all_results: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Compute mean ± std for accuracy, kappa across all subjects per model."""
    if not all_results:
        return {}
    summary: dict[str, dict[str, float]] = {}
    method_names = next(iter(all_results.values())).keys()

    for method_name in method_names:
        accuracies = np.asarray(
            [subject_result[method_name]["accuracy"] for subject_result in all_results.values()],
            dtype=np.float64,
        )
        balanced_accuracies = np.asarray(
            [subject_result[method_name].get("balanced_accuracy", 0.0) for subject_result in all_results.values()],
            dtype=np.float64,
        )
        kappas = np.asarray(
            [subject_result[method_name]["kappa"] for subject_result in all_results.values()],
            dtype=np.float64,
        )
        summary[method_name] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "balanced_accuracy_mean": float(np.mean(balanced_accuracies)),
            "balanced_accuracy_std": float(np.std(balanced_accuracies)),
            "kappa_mean": float(np.mean(kappas)),
            "kappa_std": float(np.std(kappas)),
        }

    return summary


def export_all_subjects_metrics_csv(
    all_results: dict[str, dict[str, dict[str, float]]],
    save_path: Path,
) -> None:
    """将全部被试、全部模型的指标保存为长表 CSV。"""
    fieldnames = [
        "subject_id",
        "model",
        "accuracy",
        "balanced_accuracy",
        "kappa",
        "train_time",
        "inference_time",
    ]

    rows: list[dict[str, object]] = []
    for subject_key, subject_result in all_results.items():
        # subject_key format: "subject_01"
        subject_id = int(subject_key.split("_")[1])
        for model_name, metrics in subject_result.items():
            rows.append(
                {
                    "subject_id": subject_id,
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
                    "kappa": metrics["kappa"],
                    "train_time": metrics.get("train_time", 0.0),
                    "inference_time": metrics.get("inference_time", 0.0),
                }
            )

    rows.sort(key=lambda item: (int(item["subject_id"]), str(item["model"])))

    with open(save_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_config_from_namespace(args: object) -> NNModelsBenchmarkConfig:
    """把统一 CLI 参数转成当前范式的配置对象。"""

    output_dir = args.output_dir if args.output_dir is not None else get_result_group_dir("benchmark_nn_models")
    return NNModelsBenchmarkConfig(
        subject_id=args.subject,
        output_dir=output_dir,
        all_subjects=args.all_subjects,
        show=args.show,
        shuffle_labels=getattr(args, "shuffle_labels", False),
    )


def compute_accuracy_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Compute accuracy and Cohen's kappa coefficient.
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        
    Returns:
        accuracy, kappa
    """
    accuracy = np.mean(y_true == y_pred)
    
    # Compute kappa using sklearn's formula
    n = len(y_true)
    po = accuracy  # Observed agreement
    
    # Expected agreement (assuming uniform class distribution for simplicity)
    unique_labels = np.unique(y_true)
    n_classes = len(unique_labels)
    pe = 1.0 / n_classes  # For uniform distribution
    
    kappa = (po - pe) / (1.0 - pe) if pe < 1.0 else 0.0
    
    return float(accuracy), float(kappa)


def run_tcn_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    checkpoint_dir: Path | None = None,
) -> ModelMetrics:
    """
    Run TCN model benchmark.
    
    Args:
        X_train, X_test: EEG data (n_trials, n_channels, n_samples)
        y_train, y_test: class labels
        
    Returns:
        ModelMetrics with TCN performance
    """
    logger.info("TCN: Initializing model...")
    try:
        from models.tcn_model import setup_tcn_pipeline, run_tcn_experiment
        
        logger.info("TCN: Running experiment...")
        metrics_dict, _ = run_tcn_experiment(
            X_train,
            X_test,
            y_train,
            y_test,
            checkpoint_dir=checkpoint_dir,
        )
        
        return ModelMetrics(
            model_name="TCN",
            accuracy=metrics_dict.get("accuracy", 0.0),
            kappa=metrics_dict.get("kappa", 0.0),
            balanced_accuracy=metrics_dict.get("balanced_accuracy", 0.0),
            confusion_matrix=metrics_dict.get("confusion_matrix", None),
            train_time=metrics_dict.get("train_time", 0.0),
            inference_time=metrics_dict.get("inference_time", 0.0),
            checkpoint_path=metrics_dict.get("checkpoint_path", None),
        )
    except Exception as e:
        logger.error(f"TCN benchmark failed: {e}")
        return ModelMetrics(
            model_name="TCN",
            accuracy=0.0,
            kappa=0.0,
        )


def run_atcnet_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    checkpoint_dir: Path | None = None,
) -> ModelMetrics:
    """
    Run ATCNet model benchmark.
    
    Args:
        X_train, X_test: EEG data (n_trials, n_channels, n_samples)
        y_train, y_test: class labels
        
    Returns:
        ModelMetrics with ATCNet performance
    """
    logger.info("ATCNet: Initializing model...")
    try:
        from models.atcnet_model import setup_atcnet_pipeline, run_atcnet_experiment
        
        logger.info("ATCNet: Running experiment...")
        metrics_dict, _ = run_atcnet_experiment(
            X_train,
            X_test,
            y_train,
            y_test,
            checkpoint_dir=checkpoint_dir,
        )
        
        return ModelMetrics(
            model_name="ATCNet",
            accuracy=metrics_dict.get("accuracy", 0.0),
            kappa=metrics_dict.get("kappa", 0.0),
            balanced_accuracy=metrics_dict.get("balanced_accuracy", 0.0),
            confusion_matrix=metrics_dict.get("confusion_matrix", None),
            train_time=metrics_dict.get("train_time", 0.0),
            inference_time=metrics_dict.get("inference_time", 0.0),
            checkpoint_path=metrics_dict.get("checkpoint_path", None),
        )
    except Exception as e:
        logger.error(f"ATCNet benchmark failed: {e}")
        return ModelMetrics(
            model_name="ATCNet",
            accuracy=0.0,
            kappa=0.0,
        )


def run_drsn_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    checkpoint_dir: Path | None = None,
) -> ModelMetrics:
    """
    Run DRSN model benchmark.
    
    Args:
        X_train, X_test: EEG data (n_trials, n_channels, n_samples)
        y_train, y_test: class labels
        
    Returns:
        ModelMetrics with DRSN performance
    """
    logger.info("DRSN: Initializing model...")
    try:
        from models.drsn_model import setup_drsn_pipeline, run_drsn_experiment
        
        logger.info("DRSN: Running experiment...")
        metrics_dict, _ = run_drsn_experiment(
            X_train,
            X_test,
            y_train,
            y_test,
            checkpoint_dir=checkpoint_dir,
        )
        
        return ModelMetrics(
            model_name="DRSN",
            accuracy=metrics_dict.get("accuracy", 0.0),
            kappa=metrics_dict.get("kappa", 0.0),
            balanced_accuracy=metrics_dict.get("balanced_accuracy", 0.0),
            confusion_matrix=metrics_dict.get("confusion_matrix", None),
            train_time=metrics_dict.get("train_time", 0.0),
            inference_time=metrics_dict.get("inference_time", 0.0),
            checkpoint_path=metrics_dict.get("checkpoint_path", None),
        )
    except Exception as e:
        logger.error(f"DRSN benchmark failed: {e}")
        return ModelMetrics(
            model_name="DRSN",
            accuracy=0.0,
            kappa=0.0,
        )


def run_labram_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    checkpoint_dir: Path | None = None,
) -> ModelMetrics:
    """
    Run LaBraM model benchmark.
    
    Args:
        X_train, X_test: EEG data (n_trials, n_channels, n_samples)
        y_train, y_test: class labels
        
    Returns:
        ModelMetrics with LaBraM performance
    """
    logger.info("LaBraM: Initializing model...")
    try:
        from models.labram_adapter import setup_labram_pipeline, run_labram_experiment
        
        logger.info("LaBraM: Running experiment...")
        metrics_dict, _ = run_labram_experiment(X_train, X_test, y_train, y_test, checkpoint_dir=checkpoint_dir)
        
        return ModelMetrics(
            model_name="LaBraM",
            accuracy=metrics_dict.get("accuracy", 0.0),
            kappa=metrics_dict.get("kappa", 0.0),
            balanced_accuracy=metrics_dict.get("balanced_accuracy", 0.0),
            confusion_matrix=metrics_dict.get("confusion_matrix", None),
            train_time=metrics_dict.get("train_time", 0.0),
            inference_time=metrics_dict.get("inference_time", 0.0),
            checkpoint_path=metrics_dict.get("checkpoint_path", None),
        )
    except Exception as e:
        logger.error(f"LaBraM benchmark failed: {e}")
        return ModelMetrics(
            model_name="LaBraM",
            accuracy=0.0,
            kappa=0.0,
        )


def run_all_models_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    subject_id: int | None = None,
    checkpoint_root: Path | None = None,
) -> list[ModelMetrics]:
    """
    Run all four models on the same subject data.
    
    Args:
        X_train, X_test: EEG data
        y_train, y_test: class labels
        subject_id: optional subject identifier
        
    Returns:
        List of ModelMetrics objects with results for all models
    """
    logger.info(f"Starting benchmark for subject {subject_id}...")
    
    results = []
    
    # Run each model
    for model_func, model_name in [
        (run_tcn_benchmark, "TCN"),
        (run_atcnet_benchmark, "ATCNet"),
        (run_drsn_benchmark, "DRSN"),
        (run_labram_benchmark, "LaBraM"),
    ]:
        logger.info(f"Running {model_name} benchmark...")
        try:
            checkpoint_dir = None
            if checkpoint_root is not None:
                checkpoint_dir = checkpoint_root / model_name.lower()
            if model_name == "TCN":
                metrics = run_tcn_benchmark(X_train, X_test, y_train, y_test, checkpoint_dir=checkpoint_dir)
            elif model_name == "ATCNet":
                metrics = run_atcnet_benchmark(X_train, X_test, y_train, y_test, checkpoint_dir=checkpoint_dir)
            elif model_name == "DRSN":
                metrics = run_drsn_benchmark(X_train, X_test, y_train, y_test, checkpoint_dir=checkpoint_dir)
            else:
                metrics = model_func(X_train, X_test, y_train, y_test, checkpoint_dir=checkpoint_dir)
            results.append(metrics)
            logger.info(f"{model_name}: Accuracy={metrics.accuracy:.4f}, Kappa={metrics.kappa:.4f}")
        except Exception as e:
            logger.error(f"Failed to run {model_name}: {e}")
            results.append(ModelMetrics(model_name=model_name, accuracy=0.0, kappa=0.0))
    
    return results


def save_benchmark_results(
    results: list[ModelMetrics],
    output_dir: Path,
    subject_id: int | None = None,
) -> None:
    """
    Save benchmark results to JSON.
    
    Args:
        results: list of ModelMetrics
        output_dir: directory to save results
        subject_id: optional subject identifier
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert metrics to dictionary
    results_dict = {
        "subject_id": subject_id,
        "models": [
            {
                "name": m.model_name,
                "accuracy": m.accuracy,
                "balanced_accuracy": m.balanced_accuracy,
                "kappa": m.kappa,
                "confusion_matrix": m.confusion_matrix,
                "train_time": m.train_time,
                "inference_time": m.inference_time,
                "checkpoint_path": m.checkpoint_path,
            }
            for m in results
        ],
    }
    
    # Save to file
    output_file = output_dir / f"nn_models_benchmark_subject_{subject_id}.json"
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def run_paradigm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    subject_id: int | None = None,
    output_base_dir: str | None = None,
) -> dict[str, Any]:
    """
    Main entry point for nn_models_benchmark paradigm.
    
    Args:
        X_train, X_test: EEG data (n_trials, n_channels, n_samples)
        y_train, y_test: class labels
        subject_id: optional subject identifier for logging
        output_base_dir: directory to save results
        
    Returns:
        Dictionary with benchmark results and summary
    """
    logger.info(f"Starting NN Models Benchmark Paradigm")
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_test shape: {X_test.shape}")
    logger.info(f"  y_train classes: {np.unique(y_train)}")
    
    # Run benchmark for all models
    results = run_all_models_benchmark(X_train, X_test, y_train, y_test, subject_id)
    
    # Save results if output directory is provided
    if output_base_dir is not None:
        output_dir = Path(output_base_dir) / "nn_models_benchmark"
        save_benchmark_results(results, output_dir, subject_id)
    
    # Compute summary
    accuracies = [r.accuracy for r in results]
    kappas = [r.kappa for r in results]
    
    summary = {
        "subject_id": subject_id,
        "benchmark_name": "nn_models_benchmark",
        "num_models": len(results),
        "models_tested": [r.model_name for r in results],
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_kappa": float(np.mean(kappas)),
        "best_model": max(results, key=lambda r: r.accuracy).model_name if results else None,
        "best_accuracy": float(max(accuracies)) if accuracies else 0.0,
        "results": [
            {
                "model_name": r.model_name,
                "accuracy": r.accuracy,
                "balanced_accuracy": r.balanced_accuracy,
                "kappa": r.kappa,
                "confusion_matrix": r.confusion_matrix,
            }
            for r in results
        ],
    }
    
    logger.info(f"Benchmark Summary:")
    logger.info(f"  Mean Accuracy: {summary['mean_accuracy']:.4f}")
    logger.info(f"  Mean Kappa: {summary['mean_kappa']:.4f}")
    logger.info(f"  Best Model: {summary['best_model']} ({summary['best_accuracy']:.4f})")
    
    return summary


def run_from_config(config: NNModelsBenchmarkConfig) -> dict[str, object]:
    """执行四模型基准并按被试保存结果。"""

    output_dir = config.output_dir or get_result_group_dir("benchmark_nn_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = list(range(1, 10)) if config.all_subjects else [config.subject_id]
    all_results: dict[str, list[ModelMetrics]] = {}

    for subject_id in subject_ids:
        logger.info("Loading subject %s train/test split...", subject_id)
        X_train, X_test, y_train, y_test, _ = load_subject_train_test(
            subject_id=subject_id,
            tmin=config.tmin,
            tmax=config.tmax,
            channels=None,
        )

        if config.shuffle_labels:
            rng = np.random.default_rng(42)
            y_train = rng.permutation(y_train)
            logger.info("Subject %s: training labels shuffled for negative control.", subject_id)

        logger.info("Subject %s: starting sequential benchmark.", subject_id)
        checkpoint_root = get_model_param_dir() / "benchmark_nn_models" / f"subject_{subject_id:02d}"
        subject_results = run_all_models_benchmark(
            X_train,
            X_test,
            y_train,
            y_test,
            subject_id,
            checkpoint_root=checkpoint_root,
        )
        all_results[f"subject_{subject_id:02d}"] = subject_results
        save_benchmark_results(subject_results, output_dir, subject_id)

        metrics_for_plot = {
            metric.model_name: {
                "accuracy": metric.accuracy,
                "kappa": metric.kappa,
            }
            for metric in subject_results
            if metric.checkpoint_path is not None
        }
        if metrics_for_plot:
            plot_metric_bar(
                results=metrics_for_plot,
                save_path=output_dir / f"subject_{subject_id:02d}_metrics_bar.png",
                title="BCICIV2a Neural Network Benchmark",
            )

    if config.all_subjects:
        # Convert list[ModelMetrics] → dict[str, dict] for downstream consumers
        all_results_as_dict: dict[str, dict[str, dict[str, float]]] = {
            subject_key: _model_metrics_to_dict(metrics)
            for subject_key, metrics in all_results.items()
        }

        # Long-form CSV
        export_all_subjects_metrics_csv(
            all_results=all_results_as_dict,
            save_path=output_dir / "all_subjects_metrics.csv",
        )

        # Summary statistics (mean ± std)
        summary = summarize_all_subjects(all_results_as_dict)

        # Combined summary JSON (matches benchmark_trca_wavelet_cnn format)
        json_payload: dict[str, object] = {
            "subjects": all_results_as_dict,
            "summary": summary,
        }
        with open(output_dir / "all_subjects_summary.json", "w", encoding="utf-8") as file:
            json.dump(json_payload, file, ensure_ascii=False, indent=2)

        # Also keep the legacy summary for backwards compatibility
        legacy_summary = {
            subject_key: [
                {
                    "name": metric.model_name,
                    "accuracy": metric.accuracy,
                    "kappa": metric.kappa,
                    "balanced_accuracy": metric.balanced_accuracy,
                    "train_time": metric.train_time,
                    "inference_time": metric.inference_time,
                }
                for metric in metrics
            ]
            for subject_key, metrics in all_results.items()
        }
        with open(output_dir / "nn_models_benchmark_summary.json", "w", encoding="utf-8") as file:
            json.dump(legacy_summary, file, ensure_ascii=False, indent=2)

        # Aggregate summary bar chart
        plot_aggregate_metric_bar(
            summary_results=summary,
            save_path=output_dir / "all_subjects_summary_bar.png",
        )

        # 3×3 comparison bar grid
        results_for_grid: dict[int, dict[str, dict[str, float]]] = {
            int(subject_key.split("_")[1]): all_results_as_dict[subject_key]
            for subject_key in sorted(all_results_as_dict.keys())
        }
        plot_comparison_bar_subject_grid_from_data(
            save_path=output_dir / "all_subjects_comparison_bar_grid.png",
            subject_ids=subject_ids,
            results_by_subject=results_for_grid,
            n_rows=3,
            n_cols=3,
        )

    return {
        "output_dir": output_dir,
        "subject_ids": subject_ids,
        "results": all_results,
    }
