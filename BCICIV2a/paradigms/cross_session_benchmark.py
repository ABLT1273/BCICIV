"""
Cross-session and within-session evaluation for BCICIV2a.

Evaluations:
1. E→T: train on session E, test on session T (reversed from standard)
2. T→T: train on runs 0-3 of session T, test on runs 4-5 of session T
   (within-session, no time gap)

Models: CSP+LDA, EEGNet
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
    pe = 1.0 / len(np.unique(y_true))
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else 0.0


def load_split(subject_id: int, train_session: str, test_session: str):
    """Load data for a specific train→test session split.

    train_session/test_session: 'T' or 'E'
    """
    X, y, metadata, _sfreq = load_subject_epochs(subject_id=subject_id)

    is_train = metadata["session"].astype(str).str.contains(
        "train" if train_session == "T" else "test"
    ).to_numpy()
    is_test = metadata["session"].astype(str).str.contains(
        "train" if test_session == "T" else "test"
    ).to_numpy()

    X_train = X[is_train]
    y_train = np.array([LABEL_TO_INT[lbl] for lbl in y[is_train]])
    X_test = X[is_test]
    y_test = np.array([LABEL_TO_INT[lbl] for lbl in y[is_test]])
    return X_train, X_test, y_train, y_test, _sfreq


def load_within_session_split(subject_id: int):
    """Within-session T split: train on runs 0-3, test on runs 4-5 of session T."""
    X, y, metadata, _sfreq = load_subject_epochs(subject_id=subject_id)

    is_T = metadata["session"].astype(str).str.contains("train").to_numpy()
    X_T = X[is_T]
    y_T = y[is_T]
    runs_T = metadata["run"].to_numpy()[is_T]

    train_mask = np.isin(runs_T, [0, 1, 2, 3])
    test_mask = np.isin(runs_T, [4, 5])

    X_train = X_T[train_mask]
    y_train = np.array([LABEL_TO_INT[lbl] for lbl in y_T[train_mask]])
    X_test = X_T[test_mask]
    y_test = np.array([LABEL_TO_INT[lbl] for lbl in y_T[test_mask]])
    return X_train, X_test, y_train, y_test, _sfreq


def run_csp_lda(X_train, X_test, y_train, y_test) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP

    pipeline = Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("lda", OneVsRestClassifier(LinearDiscriminantAnalysis())),
    ])
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    t_infer = time.perf_counter() - t0
    return {"accuracy": float(np.mean(y_pred == y_test)),
            "kappa": compute_kappa(y_test, y_pred),
            "train_time": t_train, "inference_time": t_infer}


def run_eegnet(X_train, X_test, y_train, y_test) -> dict:
    from sklearn.model_selection import train_test_split
    from models.deep_cnn_features import train_tiny_eeg_cnn, predict_tiny_eeg_cnn

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
    )
    t0 = time.perf_counter()
    result = train_tiny_eeg_cnn(X_tr, y_tr, X_val, y_val, epochs=50)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = predict_tiny_eeg_cnn(result, X_test)
    t_infer = time.perf_counter() - t0
    return {"accuracy": float(np.mean(y_pred == y_test)),
            "kappa": compute_kappa(y_test, y_pred),
            "train_time": t_train, "inference_time": t_infer}


def run_all(output_dir: Path | None = None):
    if output_dir is None:
        output_dir = get_result_group_dir("benchmark_cross_session")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations = [
        ("E→T", lambda sid: load_split(sid, "E", "T")),
        ("T→T", lambda sid: load_within_session_split(sid)),
    ]
    models = [
        ("CSP+LDA", run_csp_lda),
        ("EEGNet", run_eegnet),
    ]

    all_results = []
    for sid in range(1, 10):
        for eval_name, loader_fn in evaluations:
            try:
                X_train, X_test, y_train, y_test, sfreq = loader_fn(sid)
            except Exception as e:
                logger.error("%s S%d: load failed: %s", eval_name, sid, e)
                continue

            logger.info("%s S%d: train=%d test=%d", eval_name, sid, len(y_train), len(y_test))

            for model_name, model_fn in models:
                try:
                    r = model_fn(X_train, X_test, y_train, y_test)
                    r.update({"subject_id": sid, "evaluation": eval_name, "model": model_name,
                              "n_train": len(y_train), "n_test": len(y_test)})
                    all_results.append(r)
                    logger.info("  %s %s S%d: Acc=%.2f%%",
                                eval_name, model_name, sid, r["accuracy"] * 100)
                except Exception as e:
                    logger.error("%s %s S%d failed: %s", eval_name, model_name, sid, e)

    # Save CSV
    csv_path = output_dir / "all_subjects_cross_session.csv"
    fieldnames = ["subject_id", "evaluation", "model", "accuracy", "kappa",
                  "train_time", "inference_time", "n_train", "n_test"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            w.writerow(r)

    # Summary
    summary = {}
    for eval_name, _ in evaluations:
        for model_name, _ in models:
            key = f"{model_name} ({eval_name})"
            subset = [r for r in all_results
                      if r["model"] == model_name and r["evaluation"] == eval_name]
            if subset:
                accs = [r["accuracy"] for r in subset]
                kappas = [r["kappa"] for r in subset]
                summary[key] = {"accuracy_mean": float(np.mean(accs)),
                                "accuracy_std": float(np.std(accs)),
                                "kappa_mean": float(np.mean(kappas)),
                                "kappa_std": float(np.std(kappas))}

    with open(output_dir / "all_subjects_summary.json", "w") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n=== Cross-Session Summary ===")
    print(f"{'Evaluation':>25}  {'Acc':>8}  {'Std':>8}  {'Kappa':>7}")
    print("-" * 55)
    for key, m in sorted(summary.items()):
        print(f"{key:>25}  {m['accuracy_mean']*100:>7.2f}%  {m['accuracy_std']*100:>6.2f}%  {m['kappa_mean']:>7.4f}")
    print("-" * 55)

    return {"output_dir": output_dir, "results": all_results, "summary": summary}


if __name__ == "__main__":
    prepare_runtime_environment()
    run_all()
