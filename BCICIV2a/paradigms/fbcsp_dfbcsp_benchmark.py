"""
FBCSP + DFBCSP Benchmark for BCICIV2a.

Runs Filter Bank CSP and Discriminative FBCSP on all 9 subjects.
Saves per-subject results and aggregate summary.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.data import load_subject_train_test
from framework.constants import LABEL_TO_INT
from framework.metrics import compute_kappa
from framework.paths import get_paradigm_result_dir
from framework.runtime import prepare_runtime_environment
from models.FBCSP import FilterBank, OVR_FBCSP_Ensemble
from models.DFBCSP import OVR_DFBCSP_Ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FBCSPDFBCSPBenchmarkConfig:
    subject_id: int = 1
    output_dir: Path | None = None
    all_subjects: bool = False
    show: bool = False
    tmin: float = 0.5
    tmax: float = 2.5
    shuffle_labels: bool = False



def run_fbcsp_subject(subject_id: int) -> dict:
    X_train, X_test, y_train, y_test, sfreq = load_subject_train_test(
        subject_id=subject_id, tmin=0.5, tmax=2.5, channels=None,
    )
    y_train_int = np.array([LABEL_TO_INT[lbl] for lbl in y_train])
    y_test_int = np.array([LABEL_TO_INT[lbl] for lbl in y_test])

    fb = FilterBank(sfreq=int(sfreq))

    t_start = time.perf_counter()
    X_train_fb = fb.transform(X_train)
    X_test_fb = fb.transform(X_test)
    ovr = OVR_FBCSP_Ensemble(classes=[1, 2, 3, 4], m=2, k=4)
    ovr.fit(X_train_fb, y_train_int)
    t_train = time.perf_counter() - t_start

    t_start = time.perf_counter()
    y_pred = ovr.predict(X_test_fb)
    t_infer = time.perf_counter() - t_start

    acc = float(np.mean(y_pred == y_test_int))
    kappa = compute_kappa(y_test_int, y_pred)

    logger.info(f"FBCSP S{subject_id}: Acc={acc*100:.2f}%, Kappa={kappa:.4f}, Train={t_train:.1f}s")
    return {
        "subject_id": subject_id,
        "method": "FBCSP",
        "accuracy": acc,
        "kappa": kappa,
        "train_time": t_train,
        "inference_time": t_infer,
        "n_train": len(y_train_int),
        "n_test": len(y_test_int),
    }


def run_dfbcsp_subject(subject_id: int) -> dict:
    X_train, X_test, y_train, y_test, sfreq = load_subject_train_test(
        subject_id=subject_id, tmin=0.5, tmax=2.5, channels=None,
    )
    y_train_int = np.array([LABEL_TO_INT[lbl] for lbl in y_train])
    y_test_int = np.array([LABEL_TO_INT[lbl] for lbl in y_test])

    fb = FilterBank(sfreq=int(sfreq))

    t_start = time.perf_counter()
    X_train_fb = fb.transform(X_train)
    X_test_fb = fb.transform(X_test)
    ovr = OVR_DFBCSP_Ensemble(classes=[1, 2, 3, 4], m=2, k=4, n_select=4)
    ovr.fit(X_train_fb, y_train_int)
    t_train = time.perf_counter() - t_start

    t_start = time.perf_counter()
    y_pred = ovr.predict(X_test_fb)
    t_infer = time.perf_counter() - t_start

    acc = float(np.mean(y_pred == y_test_int))
    kappa = compute_kappa(y_test_int, y_pred)

    logger.info(f"DFBCSP S{subject_id}: Acc={acc*100:.2f}%, Kappa={kappa:.4f}, Train={t_train:.1f}s")
    return {
        "subject_id": subject_id,
        "method": "DFBCSP",
        "accuracy": acc,
        "kappa": kappa,
        "train_time": t_train,
        "inference_time": t_infer,
        "n_train": len(y_train_int),
        "n_test": len(y_test_int),
    }


def run_all_subjects(output_dir: Path | None = None) -> dict:
    if output_dir is None:
        output_dir = get_paradigm_result_dir("within_subject", "benchmark_fbcsp_dfbcsp")
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = list(range(1, 10))
    all_results = []

    for sid in subject_ids:
        print(f"\n===== Subject {sid} =====", flush=True)
        try:
            all_results.append(run_fbcsp_subject(sid))
        except Exception as e:
            logger.error(f"FBCSP Subject {sid} failed: {e}")
            all_results.append({"subject_id": sid, "method": "FBCSP", "accuracy": 0.0, "kappa": 0.0, "train_time": 0, "inference_time": 0, "n_train": 0, "n_test": 0, "error": str(e)})
        try:
            all_results.append(run_dfbcsp_subject(sid))
        except Exception as e:
            logger.error(f"DFBCSP Subject {sid} failed: {e}")
            all_results.append({"subject_id": sid, "method": "DFBCSP", "accuracy": 0.0, "kappa": 0.0, "train_time": 0, "inference_time": 0, "n_train": 0, "n_test": 0, "error": str(e)})

    # Save CSV
    csv_path = output_dir / "all_subjects_fbcsp_dfbcsp.csv"
    fieldnames = ["subject_id", "method", "accuracy", "kappa", "train_time", "inference_time", "n_train", "n_test"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            w.writerow(r)

    # Compute summary per method
    for method in ["FBCSP", "DFBCSP"]:
        method_results = [r for r in all_results if r["method"] == method and r.get("accuracy", 0) > 0]
        if not method_results:
            continue
        accs = [r["accuracy"] for r in method_results]
        kappas = [r["kappa"] for r in method_results]
        logger.info("=" * 60)
        logger.info(f"{method} Summary (BCIC IV 2a, all subjects)")
        logger.info(f"  Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
        logger.info(f"  Kappa:    {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
        logger.info(f"  Best:  S{np.argmax(accs)+1} ({np.max(accs)*100:.2f}%)")
        logger.info(f"  Worst: S{np.argmin(accs)+1} ({np.min(accs)*100:.2f}%)")
        logger.info("=" * 60)

    # Save summary JSON
    summary = {}
    for method in ["FBCSP", "DFBCSP"]:
        method_results = [r for r in all_results if r["method"] == method]
        accs = np.array([r["accuracy"] for r in method_results])
        kappas = np.array([r["kappa"] for r in method_results])
        summary[method] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)),
            "kappa_mean": float(np.mean(kappas)),
            "kappa_std": float(np.std(kappas, ddof=1)),
        }

    with open(output_dir / "all_subjects_summary.json", "w") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2, ensure_ascii=False)

    # Print table
    print(f"\n{'Subj':>5} {'FBCSP Acc':>10} {'FBCSP Kappa':>12} {'DFBCSP Acc':>11} {'DFBCSP Kappa':>13}")
    print("-" * 55)
    for sid in subject_ids:
        fb = next((r for r in all_results if r["subject_id"] == sid and r["method"] == "FBCSP"), None)
        df = next((r for r in all_results if r["subject_id"] == sid and r["method"] == "DFBCSP"), None)
        fb_acc = fb["accuracy"] * 100 if fb else 0
        fb_k = fb["kappa"] if fb else 0
        df_acc = df["accuracy"] * 100 if df else 0
        df_k = df["kappa"] if df else 0
        print(f"{sid:>5} {fb_acc:>9.2f}% {fb_k:>12.4f} {df_acc:>10.2f}% {df_k:>13.4f}")

    fb_mean = summary["FBCSP"]["accuracy_mean"] * 100
    df_mean = summary["DFBCSP"]["accuracy_mean"] * 100
    fb_std = summary["FBCSP"]["accuracy_std"] * 100
    df_std = summary["DFBCSP"]["accuracy_std"] * 100
    print("-" * 55)
    print(f"{'Mean':>5} {fb_mean:>9.2f}% {'±'+str(round(fb_std,2)):>11} {df_mean:>10.2f}% {'±'+str(round(df_std,2)):>12}")

    return {"output_dir": output_dir, "results": all_results, "summary": summary}


def build_config_from_namespace(args: object) -> FBCSPDFBCSPBenchmarkConfig:
    output_dir = args.output_dir if args.output_dir is not None else get_paradigm_result_dir("within_subject", "benchmark_fbcsp_dfbcsp")
    return FBCSPDFBCSPBenchmarkConfig(
        subject_id=args.subject,
        output_dir=output_dir,
        all_subjects=args.all_subjects,
        show=args.show,
        shuffle_labels=getattr(args, "shuffle_labels", False),
    )


def run_from_config(config: FBCSPDFBCSPBenchmarkConfig) -> dict:
    return run_all_subjects(output_dir=config.output_dir)


if __name__ == "__main__":
    prepare_runtime_environment()
    run_all_subjects()
