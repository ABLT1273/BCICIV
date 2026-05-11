"""
CSP + LDA Benchmark for BCICIV2a.

Compares classic CSP+LDA pipeline against deep learning models.
Uses One-Vs-Rest (OVR) strategy for 4-class classification.

Pipeline:
- Load subject data from BCICIV2a (Session T / Session E split)
- Apply CSP spatial filtering per class pair
- Train LDA classifier
- Report accuracy, kappa per subject and overall
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from mne.decoding import CSP

# Ensure BCICIV2a root is on sys.path for framework imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.data import load_subject_train_test
from framework.metrics import compute_kappa
from framework.paths import get_paradigm_result_dir
from framework.runtime import prepare_runtime_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CSPLDABenchmarkConfig:
    subject_id: int = 1
    output_dir: Path | None = None
    all_subjects: bool = False
    show: bool = False
    tmin: float = 0.5
    tmax: float = 2.5
    shuffle_labels: bool = False



def run_csp_lda_subject(
    subject_id: int,
    tmin: float = 0.5,
    tmax: float = 2.5,
    n_components: int = 4,
) -> dict:
    """
    Run CSP + LDA on a single subject.

    Args:
        subject_id: subject id (1-9)
        tmin/tmax: time window in seconds
        n_components: number of CSP components per class pair

    Returns:
        dict with accuracy, kappa, train_time, inference_time
    """
    # Load data
    X_train, X_test, y_train, y_test, sfreq = load_subject_train_test(
        subject_id=subject_id,
        tmin=tmin,
        tmax=tmax,
        channels=None,
    )

    # y values are strings like "left_hand", "right_hand", "feet", "tongue"
    # Map them to integers for CSP
    from framework.constants import LABEL_TO_INT
    y_train_int = np.array([LABEL_TO_INT[lbl] for lbl in y_train])
    y_test_int = np.array([LABEL_TO_INT[lbl] for lbl in y_test])

    # Build CSP + LDA pipeline
    # CSP with 'ledoit_wolf' regularization to handle rank-deficient data
    # (e.g. after CAR preprocessing reduces rank from 22→21)
    pipeline = Pipeline([
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("lda", OneVsRestClassifier(LinearDiscriminantAnalysis())),
    ])

    # Train
    t_start = time.perf_counter()
    pipeline.fit(X_train, y_train_int)
    t_train = time.perf_counter() - t_start

    # Inference
    t_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    t_infer = time.perf_counter() - t_start

    acc = float(np.mean(y_pred == y_test_int))
    kappa = compute_kappa(y_test_int, y_pred)

    logger.info(
        f"Subject {subject_id}: Acc={acc*100:.2f}%, Kappa={kappa:.4f}, "
        f"Train={t_train:.2f}s, Infer={t_infer:.3f}s"
    )

    return {
        "subject_id": subject_id,
        "accuracy": acc,
        "kappa": kappa,
        "train_time": t_train,
        "inference_time": t_infer,
        "n_train": len(y_train_int),
        "n_test": len(y_test_int),
    }


def run_csp_svm_subject(
    subject_id: int,
    tmin: float = 0.5,
    tmax: float = 2.5,
    n_components: int = 4,
) -> dict:
    """Run CSP + RBF-SVM on a single subject."""
    X_train, X_test, y_train, y_test, sfreq = load_subject_train_test(
        subject_id=subject_id, tmin=tmin, tmax=tmax, channels=None,
    )
    from framework.constants import LABEL_TO_INT
    y_train_int = np.array([LABEL_TO_INT[lbl] for lbl in y_train])
    y_test_int = np.array([LABEL_TO_INT[lbl] for lbl in y_test])

    pipeline = Pipeline([
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("svm", OneVsRestClassifier(SVC(kernel="rbf", C=1.0, gamma="scale", probability=False))),
    ])

    t_start = time.perf_counter()
    pipeline.fit(X_train, y_train_int)
    t_train = time.perf_counter() - t_start

    t_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    t_infer = time.perf_counter() - t_start

    acc = float(np.mean(y_pred == y_test_int))
    kappa = compute_kappa(y_test_int, y_pred)

    logger.info(
        f"Subject {subject_id} (CSP+SVM): Acc={acc*100:.2f}%, Kappa={kappa:.4f}, "
        f"Train={t_train:.2f}s, Infer={t_infer:.3f}s"
    )
    return {
        "subject_id": subject_id,
        "accuracy": acc,
        "kappa": kappa,
        "train_time": t_train,
        "inference_time": t_infer,
        "n_train": len(y_train_int),
        "n_test": len(y_test_int),
    }


def run_all_subjects(
    subject_ids: list[int] = None,
    output_dir: Path | None = None,
) -> list[dict]:
    """Run CSP+LDA on all subjects and save results."""
    if subject_ids is None:
        subject_ids = list(range(1, 10))

    if output_dir is None:
        output_dir = get_paradigm_result_dir("within_subject", "benchmark_csp_lda")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sid in subject_ids:
        try:
            result = run_csp_lda_subject(sid)
            results.append(result)
        except Exception as e:
            logger.error(f"Subject {sid} failed: {e}")
            results.append({
                "subject_id": sid,
                "accuracy": 0.0,
                "kappa": 0.0,
                "train_time": 0.0,
                "inference_time": 0.0,
                "n_train": 0,
                "n_test": 0,
                "error": str(e),
            })

    # Compute aggregate
    accs = [r["accuracy"] for r in results if r["accuracy"] > 0]
    kappas = [r["kappa"] for r in results if r["accuracy"] > 0]

    summary = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "kappa_mean": float(np.mean(kappas)),
        "kappa_std": float(np.std(kappas)),
        "n_subjects": len(subject_ids),
    }

    # Save per-subject CSV
    csv_path = output_dir / "all_subjects_csp_lda.csv"
    fieldnames = ["subject_id", "accuracy", "kappa", "train_time", "inference_time", "n_train", "n_test"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Save summary
    logger.info("=" * 60)
    logger.info("CSP+LDA Summary (BCIC IV 2a, all subjects)")
    logger.info(f"  Accuracy: {summary['accuracy_mean']*100:.2f}% ± {summary['accuracy_std']*100:.2f}%")
    logger.info(f"  Kappa:    {summary['kappa_mean']:.4f} ± {summary['kappa_std']:.4f}")
    logger.info(f"  Subjects: {summary['n_subjects']}")
    logger.info("=" * 60)

    print(f"\n=== Per-Subject Results ===")
    print(f"{'Subj':>5} {'Acc%':>8} {'Kappa':>8} {'Train(s)':>10} {'Infer(s)':>10}")
    print("-" * 45)
    for r in results:
        print(f"{r['subject_id']:>5} {r['accuracy']*100:>7.2f}% {r['kappa']:>8.4f} {r['train_time']:>10.2f} {r['inference_time']:>10.4f}")
    print("-" * 45)
    print(f"{'Mean':>5} {summary['accuracy_mean']*100:>7.2f}% {summary['kappa_mean']:>8.4f}")
    print(f"{'Std':>5} {summary['accuracy_std']*100:>7.2f}% {summary['kappa_std']:>8.4f}")

    return results


def build_config_from_namespace(args: object) -> CSPLDABenchmarkConfig:
    output_dir = args.output_dir if args.output_dir is not None else get_paradigm_result_dir("within_subject", "benchmark_csp_lda")
    return CSPLDABenchmarkConfig(
        subject_id=args.subject,
        output_dir=output_dir,
        all_subjects=args.all_subjects,
        show=args.show,
        shuffle_labels=getattr(args, "shuffle_labels", False),
    )


def run_from_config(config: CSPLDABenchmarkConfig) -> dict:
    subject_ids = list(range(1, 10)) if config.all_subjects else [config.subject_id]
    return {"results": run_all_subjects(subject_ids=subject_ids, output_dir=config.output_dir)}


if __name__ == "__main__":
    prepare_runtime_environment()
    run_all_subjects()
