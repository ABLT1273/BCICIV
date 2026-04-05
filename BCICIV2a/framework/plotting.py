from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import LABEL_TO_COLOR, LABEL_TO_DISPLAY_NAME


def plot_3d_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
    show: bool = False,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label in LABEL_TO_DISPLAY_NAME:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=36,
            alpha=0.85,
            color=LABEL_TO_COLOR[label],
            label=LABEL_TO_DISPLAY_NAME[label],
            edgecolors="black",
            linewidths=0.2,
        )

    ax.set_title(title, pad=18)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_metric_bar(results: dict[str, dict[str, float]], save_path: Path) -> None:
    methods = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in methods]
    kappas = [results[name]["kappa"] for name in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, accuracies, width, label="Accuracy")
    ax.bar(x + width / 2, kappas, width, label="Kappa")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("BCICIV2a Feature Experiments")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_metric_bar(
    summary_results: dict[str, dict[str, float]],
    save_path: Path,
) -> None:
    methods = list(summary_results.keys())
    accuracies = [summary_results[name]["accuracy_mean"] for name in methods]
    kappas = [summary_results[name]["kappa_mean"] for name in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, accuracies, width, label="Mean Accuracy")
    ax.bar(x + width / 2, kappas, width, label="Mean Kappa")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("BCICIV2a All-Subject Average Results")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

