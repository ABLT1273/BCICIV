"""
LOSO (Leave-One-Subject-Out) Benchmark for BCICIV2a.

Cross-subject generalization: train on session T of 8 subjects,
test on session E of the held-out subject.

Models: CSP+LDA, CSP+SVM, FBCSP, EEGNet
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.data import load_subject_epochs
from framework.constants import LABEL_TO_INT
from framework.paths import get_result_group_dir
from framework.runtime import prepare_runtime_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    po = np.mean(y_true == y_pred)
    n_classes = len(np.unique(y_true))
    pe = 1.0 / n_classes
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else 0.0


def load_loso_split(test_subject: int):
    """Load LOSO train/test split.

    Train: session T of all subjects except test_subject.
    Test: session E of test_subject.
    """
    all_X_train = []
    all_y_train = []
    X_test = None
    y_test = None

    for sid in range(1, 10):
        X, y, metadata, _sfreq = load_subject_epochs(subject_id=sid)
        is_train = metadata["session"].astype(str).str.contains("train").to_numpy()
        is_test = metadata["session"].astype(str).str.contains("test").to_numpy()

        if sid == test_subject:
            X_test = X[is_test]
            y_test = np.array([LABEL_TO_INT[lbl] for lbl in y[is_test]])
        else:
            all_X_train.append(X[is_train])
            all_y_train.append(np.array([LABEL_TO_INT[lbl] for lbl in y[is_train]]))

    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    return X_train, X_test, y_train, y_test, 250.0


# ---------------------------------------------------------------------------
# CSP+LDA
# ---------------------------------------------------------------------------

def run_csp_lda_loso(subject_id: int) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP

    logger.info("LOSO S%d: CSP+LDA loading data...", subject_id)
    X_train, X_test, y_train, y_test, sfreq = load_loso_split(subject_id)

    pipeline = Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("lda", OneVsRestClassifier(LinearDiscriminantAnalysis())),
    ])

    t_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    t_train = time.perf_counter() - t_start

    t_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    t_infer = time.perf_counter() - t_start

    acc = float(np.mean(y_pred == y_test))
    kappa = compute_kappa(y_test, y_pred)

    logger.info("LOSO S%d CSP+LDA: Acc=%.2f%%, Kappa=%.4f, Train=%.1fs",
                subject_id, acc * 100, kappa, t_train)
    return {"subject_id": subject_id, "method": "CSP+LDA", "accuracy": acc,
            "kappa": kappa, "train_time": t_train, "inference_time": t_infer}


# ---------------------------------------------------------------------------
# CSP+SVM
# ---------------------------------------------------------------------------

def run_csp_svm_loso(subject_id: int) -> dict:
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from mne.decoding import CSP

    logger.info("LOSO S%d: CSP+SVM loading data...", subject_id)
    X_train, X_test, y_train, y_test, sfreq = load_loso_split(subject_id)

    pipeline = Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("svm", OneVsRestClassifier(SVC(kernel="rbf", C=1.0, gamma="scale"))),
    ])

    t_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    t_train = time.perf_counter() - t_start

    t_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    t_infer = time.perf_counter() - t_start

    acc = float(np.mean(y_pred == y_test))
    kappa = compute_kappa(y_test, y_pred)

    logger.info("LOSO S%d CSP+SVM: Acc=%.2f%%, Kappa=%.4f, Train=%.1fs",
                subject_id, acc * 100, kappa, t_train)
    return {"subject_id": subject_id, "method": "CSP+SVM", "accuracy": acc,
            "kappa": kappa, "train_time": t_train, "inference_time": t_infer}


# ---------------------------------------------------------------------------
# FBCSP
# ---------------------------------------------------------------------------

def run_fbcsp_loso(subject_id: int) -> dict:
    from models.FBCSP import FilterBank, OVR_FBCSP_Ensemble

    logger.info("LOSO S%d: FBCSP loading data...", subject_id)
    X_train, X_test, y_train, y_test, sfreq = load_loso_split(subject_id)

    fb = FilterBank(sfreq=int(sfreq))

    t_start = time.perf_counter()
    X_train_fb = fb.transform(X_train)
    X_test_fb = fb.transform(X_test)
    ovr = OVR_FBCSP_Ensemble(classes=[1, 2, 3, 4], m=2, k=4)
    ovr.fit(X_train_fb, y_train)
    t_train = time.perf_counter() - t_start

    t_start = time.perf_counter()
    y_pred = ovr.predict(X_test_fb)
    t_infer = time.perf_counter() - t_start

    acc = float(np.mean(y_pred == y_test))
    kappa = compute_kappa(y_test, y_pred)

    logger.info("LOSO S%d FBCSP: Acc=%.2f%%, Kappa=%.4f, Train=%.1fs",
                subject_id, acc * 100, kappa, t_train)
    return {"subject_id": subject_id, "method": "FBCSP", "accuracy": acc,
            "kappa": kappa, "train_time": t_train, "inference_time": t_infer}


# ---------------------------------------------------------------------------
# EEGNet
# ---------------------------------------------------------------------------

def run_eegnet_loso(subject_id: int) -> dict:
    from sklearn.model_selection import train_test_split
    from models.deep_cnn_features import train_tiny_eeg_cnn, predict_tiny_eeg_cnn

    logger.info("LOSO S%d: EEGNet loading data...", subject_id)
    X_train, X_test, y_train, y_test, sfreq = load_loso_split(subject_id)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
    )

    t_start = time.perf_counter()
    result = train_tiny_eeg_cnn(X_tr, y_tr, X_val, y_val, epochs=50)
    t_train = time.perf_counter() - t_start

    t_start = time.perf_counter()
    y_pred = predict_tiny_eeg_cnn(result, X_test)
    t_infer = time.perf_counter() - t_start

    acc = float(np.mean(y_pred == y_test))
    kappa = compute_kappa(y_test, y_pred)

    logger.info("LOSO S%d EEGNet: Acc=%.2f%%, Kappa=%.4f, Train=%.1fs",
                subject_id, acc * 100, kappa, t_train)
    return {"subject_id": subject_id, "method": "EEGNet", "accuracy": acc,
            "kappa": kappa, "train_time": t_train, "inference_time": t_infer}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_loso(output_dir: Path | None = None) -> dict:
    if output_dir is None:
        output_dir = get_result_group_dir("benchmark_loso")
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        ("CSP+LDA", run_csp_lda_loso),
        ("CSP+SVM", run_csp_svm_loso),
        ("FBCSP", run_fbcsp_loso),
        ("EEGNet", run_eegnet_loso),
    ]

    all_results = []
    for sid in range(1, 10):
        print(f"\n===== LOSO Subject {sid} =====", flush=True)
        for method_name, method_fn in methods:
            try:
                all_results.append(method_fn(sid))
            except Exception as e:
                logger.error("LOSO %s Subject %d failed: %s", method_name, sid, e)

    # Save CSV
    csv_path = output_dir / "all_subjects_loso.csv"
    fieldnames = ["subject_id", "method", "accuracy", "kappa", "train_time", "inference_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            w.writerow(r)

    # Per-method summary
    summary = {}
    for method_name, _ in methods:
        method_results = [r for r in all_results if r["method"] == method_name]
        accs = np.array([r["accuracy"] for r in method_results])
        kappas = np.array([r["kappa"] for r in method_results])
        summary[method_name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "kappa_mean": float(np.mean(kappas)),
            "kappa_std": float(np.std(kappas)),
        }
        logger.info("%s LOSO: Acc=%.2f%% ± %.2f%%",
                    method_name, np.mean(accs) * 100, np.std(accs) * 100)

    with open(output_dir / "all_subjects_summary.json", "w") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2, ensure_ascii=False)

    # Print table
    print(f"\n{'Subj':>5}", end="")
    for method_name, _ in methods:
        print(f"  {method_name:>10}", end="")
    print()
    print("-" * (8 + 12 * len(methods)))
    for sid in range(1, 10):
        print(f"{sid:>5}", end="")
        for method_name, _ in methods:
            r = next((r for r in all_results if r["subject_id"] == sid and r["method"] == method_name), None)
            acc = r["accuracy"] * 100 if r else 0
            print(f"  {acc:>9.2f}%", end="")
        print()
    print("-" * (8 + 12 * len(methods)))
    print(f"{'Mean':>5}", end="")
    for method_name, _ in methods:
        m = summary[method_name]
        print(f"  {m['accuracy_mean']*100:>8.2f}%", end="")
    print()

    return {"output_dir": output_dir, "results": all_results, "summary": summary}


if __name__ == "__main__":
    prepare_runtime_environment()
    run_all_loso()
