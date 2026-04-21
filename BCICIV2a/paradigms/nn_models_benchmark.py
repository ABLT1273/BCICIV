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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
    train_time: float = 0.0
    inference_time: float = 0.0


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


def run_tcn_benchmark(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
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
        metrics_dict, _ = run_tcn_experiment(X_train, X_test, y_train, y_test)
        
        return ModelMetrics(
            model_name="TCN",
            accuracy=metrics_dict.get("accuracy", 0.0),
            kappa=metrics_dict.get("kappa", 0.0),
        )
    except Exception as e:
        logger.error(f"TCN benchmark failed: {e}")
        # Return dummy metrics
        return ModelMetrics(
            model_name="TCN",
            accuracy=0.0,
            kappa=0.0,
        )


def run_atcnet_benchmark(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
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
        metrics_dict, _ = run_atcnet_experiment(X_train, X_test, y_train, y_test)
        
        return ModelMetrics(
            model_name="ATCNet",
            accuracy=metrics_dict.get("accuracy", 0.0),
            kappa=metrics_dict.get("kappa", 0.0),
        )
    except Exception as e:
        logger.error(f"ATCNet benchmark failed: {e}")
        # Return dummy metrics
        return ModelMetrics(
            model_name="ATCNet",
            accuracy=0.0,
            kappa=0.0,
        )


def run_drsn_benchmark(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
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
        metrics_dict, _ = run_drsn_experiment(X_train, X_test, y_train, y_test)
        
        return ModelMetrics(
            model_name="DRSN",
            accuracy=metrics_dict.get("accuracy", 0.0),
            kappa=metrics_dict.get("kappa", 0.0),
        )
    except Exception as e:
        logger.error(f"DRSN benchmark failed: {e}")
        # Return dummy metrics
        return ModelMetrics(
            model_name="DRSN",
            accuracy=0.0,
            kappa=0.0,
        )


def run_labram_benchmark(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
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
        metrics_dict, _ = run_labram_experiment(X_train, X_test, y_train, y_test)
        
        return ModelMetrics(
            model_name="LaBraM",
            accuracy=metrics_dict.get("accuracy", 0.0),
            kappa=metrics_dict.get("kappa", 0.0),
        )
    except Exception as e:
        logger.error(f"LaBraM benchmark failed: {e}")
        # Return dummy metrics
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
            metrics = model_func(X_train, X_test, y_train, y_test)
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
                "kappa": m.kappa,
                "train_time": m.train_time,
                "inference_time": m.inference_time,
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
                "kappa": r.kappa,
            }
            for r in results
        ],
    }
    
    logger.info(f"Benchmark Summary:")
    logger.info(f"  Mean Accuracy: {summary['mean_accuracy']:.4f}")
    logger.info(f"  Mean Kappa: {summary['mean_kappa']:.4f}")
    logger.info(f"  Best Model: {summary['best_model']} ({summary['best_accuracy']:.4f})")
    
    return summary
