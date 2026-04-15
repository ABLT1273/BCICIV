from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models.deep_cnn_features import (
    extract_tiny_eeg_cnn_features,
    predict_tiny_eeg_cnn,
    train_tiny_eeg_cnn,
)
from framework.data import load_subject_train_test, select_named_channels
from framework.paths import get_advanced_results_dir
from framework.plotting import (
    plot_comparison_bar_subject_grid_from_data,
    plot_aggregate_metric_bar,
    plot_umap_subject_method_grid_from_data,
)
from models.trca_module import TRCAHybridClassifier
from models.wavelet_features import WaveletEnergyFeatureExtractor


@dataclass
class AdvancedBenchmarkConfig:
    subject_id: int = 1
    output_dir: Path | None = None
    all_subjects: bool = False
    show: bool = False
    tmin: float = 0.5
    tmax: float = 2.5


def reduce_for_visualization(features: np.ndarray) -> np.ndarray:
    """统一用普通 UMAP 把测试集特征压到 3 维做可视化。"""

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "没有检测到 umap-learn，请先安装：\n"
            "test_newPyEnv/.venv/bin/pip install umap-learn"
        ) from exc

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(20, max(5, len(features_scaled) - 1)),
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )
    return reducer.fit_transform(features_scaled)


def run_trca_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    print("正在运行 TRCA 实验...", flush=True)
    classifier = TRCAHybridClassifier(
        n_components=3,
        svm_c=2.5,
        svm_gamma="scale",
        template_weight=0.4,
    )
    classifier.fit(X_train, y_train)
    test_features = classifier.transform(X_test)
    predictions = classifier.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "kappa": float(cohen_kappa_score(y_test, predictions)),
    }
    print(classification_report(y_test, predictions), flush=True)
    return metrics, test_features


def run_wavelet_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
) -> tuple[dict[str, float], np.ndarray]:
    print("正在运行 Wavelet 实验...", flush=True)

    selected_channels = ["C3", "CZ", "C4"]
    X_train_center = select_named_channels(X_train, selected_channels)
    X_test_center = select_named_channels(X_test, selected_channels)

    extractor = WaveletEnergyFeatureExtractor(sfreq=sfreq)
    train_features = extractor.transform(X_train_center)
    test_features = extractor.transform(X_test_center)

    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=2.0, gamma="scale")),
        ]
    )
    classifier.fit(train_features, y_train)
    predictions = classifier.predict(test_features)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "kappa": float(cohen_kappa_score(y_test, predictions)),
    }
    print(classification_report(y_test, predictions), flush=True)
    return metrics, test_features


def run_cnn_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    print("正在运行 CNN 实验...", flush=True)
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )
    result = train_tiny_eeg_cnn(
        X_train_sub,
        y_train_sub,
        X_val_sub,
        y_val_sub,
        epochs=12,
        batch_size=64,
        learning_rate=1e-3,
    )
    predictions = predict_tiny_eeg_cnn(result, X_test)
    deep_features = extract_tiny_eeg_cnn_features(result, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "kappa": float(cohen_kappa_score(y_test, predictions)),
        "best_val_accuracy": float(result.best_val_accuracy),
    }
    print(classification_report(y_test, predictions), flush=True)
    return metrics, deep_features


def run_subject_experiment(
    subject_id: int,
    output_dir: Path,
    show: bool = False,
    tmin: float = 0.5,
    tmax: float = 2.5,
) -> dict[str, object]:
    print("正在读取 BCICIV2a 训练/测试数据...", flush=True)
    X_train, X_test, y_train, y_test, sfreq = load_subject_train_test(
        subject_id=subject_id,
        tmin=tmin,
        tmax=tmax,
        channels=None,
    )
    print(
        f"被试 {subject_id}: train={X_train.shape}, test={X_test.shape}, sfreq={sfreq}",
        flush=True,
    )

    results: dict[str, dict[str, float]] = {}
    feature_sets: dict[str, np.ndarray] = {}

    trca_metrics, trca_features = run_trca_experiment(X_train, X_test, y_train, y_test)
    results["TRCA"] = trca_metrics
    feature_sets["TRCA"] = trca_features

    wavelet_metrics, wavelet_features = run_wavelet_experiment(
        X_train,
        X_test,
        y_train,
        y_test,
        sfreq,
    )
    results["Wavelet"] = wavelet_metrics
    feature_sets["Wavelet"] = wavelet_features

    cnn_metrics, cnn_features = run_cnn_experiment(X_train, X_test, y_train, y_test)
    results["CNN"] = cnn_metrics
    feature_sets["CNN"] = cnn_features

    print("正在计算 UMAP 嵌入（总图内存管路）...", flush=True)

    umap_embeddings: dict[str, np.ndarray] = {}
    for method_name, features in feature_sets.items():
        umap_embeddings[method_name.lower()] = reduce_for_visualization(features)

    print("实验完成。结果摘要：", flush=True)
    for method_name, metrics in results.items():
        print(
            f"  - {method_name}: accuracy={metrics['accuracy']:.4f}, "
            f"kappa={metrics['kappa']:.4f}",
            flush=True,
        )

    return {
        "results": results,
        "umap_embeddings": umap_embeddings,
        "labels": y_test,
    }


def summarize_all_subjects(
    all_results: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    method_names = next(iter(all_results.values())).keys()

    for method_name in method_names:
        accuracies = np.asarray(
            [subject_result[method_name]["accuracy"] for subject_result in all_results.values()],
            dtype=np.float64,
        )
        kappas = np.asarray(
            [subject_result[method_name]["kappa"] for subject_result in all_results.values()],
            dtype=np.float64,
        )
        summary[method_name] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "kappa_mean": float(np.mean(kappas)),
            "kappa_std": float(np.std(kappas)),
        }

    return summary


def export_all_subjects_metrics_csv(
    all_results: dict[str, dict[str, dict[str, float]]],
    save_path: Path,
) -> None:
    """将所有被试、所有方法的指标保存为长表 CSV。"""

    fieldnames = [
        "subject_id",
        "method",
        "accuracy",
        "kappa",
        "best_val_accuracy",
    ]

    rows: list[dict[str, object]] = []
    for subject_key, subject_result in all_results.items():
        subject_id = int(subject_key.split("_")[-1])
        for method_name, metrics in subject_result.items():
            rows.append(
                {
                    "subject_id": subject_id,
                    "method": method_name,
                    "accuracy": metrics["accuracy"],
                    "kappa": metrics["kappa"],
                    "best_val_accuracy": metrics.get("best_val_accuracy", ""),
                }
            )

    rows.sort(key=lambda item: (int(item["subject_id"]), str(item["method"])))

    with open(save_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_config_from_namespace(args: object) -> AdvancedBenchmarkConfig:
    """把统一 CLI 参数转成当前范式的配置对象。"""

    output_dir = args.output_dir if args.output_dir is not None else get_advanced_results_dir()
    return AdvancedBenchmarkConfig(
        subject_id=args.subject,
        output_dir=output_dir,
        all_subjects=args.all_subjects,
        show=args.show,
    )


def run_from_config(config: AdvancedBenchmarkConfig) -> dict[str, object]:
    """执行 TRCA / Wavelet / CNN 综合对比实验。"""

    output_dir = config.output_dir or get_advanced_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = list(range(1, 10)) if config.all_subjects else [config.subject_id]
    all_results: dict[str, dict[str, dict[str, float]]] = {}
    embeddings_by_subject: dict[int, dict[str, np.ndarray]] = {}
    labels_by_subject: dict[int, np.ndarray] = {}

    for subject_id in subject_ids:
        print(f"\n===== 开始处理被试 {subject_id} =====", flush=True)
        subject_output = run_subject_experiment(
            subject_id=subject_id,
            output_dir=output_dir,
            show=config.show if len(subject_ids) == 1 else False,
            tmin=config.tmin,
            tmax=config.tmax,
        )
        all_results[f"subject_{subject_id:02d}"] = subject_output["results"]
        embeddings_by_subject[subject_id] = subject_output["umap_embeddings"]
        labels_by_subject[subject_id] = np.asarray(subject_output["labels"])

    export_all_subjects_metrics_csv(
        all_results=all_results,
        save_path=output_dir / "all_subjects_metrics.csv",
    )

    summary = None
    if config.all_subjects:
        summary = summarize_all_subjects(all_results)
        with open(
            output_dir / "all_subjects_summary.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "subjects": all_results,
                    "summary": summary,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        plot_aggregate_metric_bar(
            summary_results=summary,
            save_path=output_dir / "all_subjects_summary_bar.png",
        )

        plot_umap_subject_method_grid_from_data(
            save_path=output_dir / "all_subjects_umap3d_grid.png",
            subject_ids=subject_ids,
            method_names=["trca", "wavelet", "cnn"],
            embeddings_by_subject=embeddings_by_subject,
            labels_by_subject=labels_by_subject,
            method_display_names=["TRCA", "Wavelet", "CNN"],
        )

        plot_comparison_bar_subject_grid_from_data(
            save_path=output_dir / "all_subjects_comparison_bar_grid.png",
            subject_ids=subject_ids,
            results_by_subject={
                subject_id: all_results[f"subject_{subject_id:02d}"]
                for subject_id in subject_ids
            },
            n_rows=3,
            n_cols=3,
        )

    if not config.all_subjects:
        subject_id = subject_ids[0]
        plot_umap_subject_method_grid_from_data(
            save_path=output_dir / f"subject_{subject_id:02d}_umap3d_grid.png",
            subject_ids=[subject_id],
            method_names=["trca", "wavelet", "cnn"],
            embeddings_by_subject={subject_id: embeddings_by_subject[subject_id]},
            labels_by_subject={subject_id: labels_by_subject[subject_id]},
            method_display_names=["TRCA", "Wavelet", "CNN"],
        )

        plot_comparison_bar_subject_grid_from_data(
            save_path=output_dir / f"subject_{subject_id:02d}_comparison_bar_grid.png",
            subject_ids=[subject_id],
            results_by_subject={subject_id: all_results[f"subject_{subject_id:02d}"]},
            n_rows=1,
            n_cols=1,
        )

        print("\n全部被试平均结果：", flush=True)
        for method_name, metrics in summary.items():
            print(
                f"  - {method_name}: "
                f"accuracy={metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.4f}, "
                f"kappa={metrics['kappa_mean']:.4f}±{metrics['kappa_std']:.4f}",
                flush=True,
            )

    return {
        "output_dir": output_dir,
        "subject_ids": subject_ids,
        "results": all_results,
        "summary": summary,
    }
