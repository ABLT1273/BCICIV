from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def _scatter_embedding_on_axis(
    ax: object,
    embedding: np.ndarray,
    labels: np.ndarray,
) -> None:
    for label in LABEL_TO_DISPLAY_NAME:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=12,
            alpha=0.85,
            color=LABEL_TO_COLOR[label],
            edgecolors="black",
            linewidths=0.15,
        )


def plot_umap_subject_method_grid_from_data(
    save_path: Path,
    subject_ids: list[int],
    method_names: list[str],
    embeddings_by_subject: dict[int, dict[str, np.ndarray]],
    labels_by_subject: dict[int, np.ndarray],
    method_display_names: list[str] | None = None,
) -> None:
    """直接使用内存中的 embedding 画出所有被试 x 方法的 UMAP 3D 总图。"""

    if method_display_names is None:
        method_display_names = method_names

    n_rows = len(subject_ids)
    n_cols = len(method_names)
    fig = plt.figure(figsize=(5.6 * n_cols, 4.2 * n_rows))
    axes = np.empty((n_rows, n_cols), dtype=object)

    for row_index, subject_id in enumerate(subject_ids):
        subject_embeddings = embeddings_by_subject[subject_id]
        subject_labels = labels_by_subject[subject_id]
        for col_index, method_name in enumerate(method_names):
            ax = fig.add_subplot(n_rows, n_cols, row_index * n_cols + col_index + 1, projection="3d")
            axes[row_index, col_index] = ax
            _scatter_embedding_on_axis(
                ax=ax,
                embedding=subject_embeddings[method_name],
                labels=subject_labels,
            )
            ax.set_xlabel("UMAP-1")
            ax.set_zlabel("UMAP-3")
            if col_index == 0:
                ax.set_ylabel(
                    f"Subject {subject_id}",
                    rotation=0,
                    labelpad=34,
                )
            else:
                ax.set_ylabel("UMAP-2")

    for col_index, title in enumerate(method_display_names):
        axes[0, col_index].set_title(title)

    # 仅在右上角子图放置一次图例，避免 27 个子图重复图例造成遮挡。
    legend_ax = axes[0, -1]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=LABEL_TO_COLOR[label],
            markeredgecolor="black",
            markeredgewidth=0.2,
            markersize=5,
            label=LABEL_TO_DISPLAY_NAME[label],
        )
        for label in LABEL_TO_DISPLAY_NAME
    ]
    legend_ax.legend(handles=handles, loc="upper right", fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar_subject_grid_from_data(
    save_path: Path,
    subject_ids: list[int],
    results_by_subject: dict[int, dict[str, dict[str, float]]],
    n_rows: int = 3,
    n_cols: int = 3,
) -> None:
    """直接使用内存中的指标结果画出 3x3 被试 comparison bar 总图。"""

    expected = n_rows * n_cols
    if len(subject_ids) != expected:
        raise ValueError(
            f"subject_ids 数量必须为 {expected} 才能拼成 {n_rows}x{n_cols}，当前为 {len(subject_ids)}。"
        )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.4 * n_rows))
    axes = np.atleast_2d(axes)

    for index, subject_id in enumerate(subject_ids):
        row_index = index // n_cols
        col_index = index % n_cols
        ax = axes[row_index, col_index]
        results = results_by_subject[subject_id]

        methods = list(results.keys())
        accuracies = [results[name]["accuracy"] for name in methods]
        kappas = [results[name]["kappa"] for name in methods]

        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width / 2, accuracies, width, label="Accuracy")
        ax.bar(x + width / 2, kappas, width, label="Kappa")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Subject {subject_id}")
        if col_index == 0:
            ax.set_ylabel("Score")

    # 仅在左上角子图放一次图例，减少遮挡。
    axes[0, 0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_umap_subject_method_grid(
    output_dir: Path,
    save_path: Path,
    subject_ids: list[int],
    method_names: list[str],
    method_display_names: list[str] | None = None,
) -> None:
    """将所有被试与模型的 UMAP 单图拼接成 9x3 大图。"""

    if method_display_names is None:
        method_display_names = method_names

    n_rows = len(subject_ids)
    n_cols = len(method_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.8 * n_rows))

    axes = np.atleast_2d(axes)

    for row_index, subject_id in enumerate(subject_ids):
        for col_index, method_name in enumerate(method_names):
            ax = axes[row_index, col_index]
            image_path = output_dir / f"subject_{subject_id:02d}_{method_name.lower()}_umap3d.png"
            image = mpimg.imread(image_path)
            ax.imshow(image)
            ax.axis("off")

    for col_index, title in enumerate(method_display_names):
        axes[0, col_index].set_title(title)

    for row_index, subject_id in enumerate(subject_ids):
        axes[row_index, 0].set_ylabel(
            f"Subject {subject_id}",
            rotation=0,
            labelpad=56,
            va="center",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar_subject_grid(
    output_dir: Path,
    save_path: Path,
    subject_ids: list[int],
    n_rows: int = 3,
    n_cols: int = 3,
) -> None:
    """将各被试 comparison bar 单图拼接成 3x3 大图。"""

    expected = n_rows * n_cols
    if len(subject_ids) != expected:
        raise ValueError(
            f"subject_ids 数量必须为 {expected} 才能拼成 {n_rows}x{n_cols}，当前为 {len(subject_ids)}。"
        )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.8 * n_rows))
    axes = np.atleast_2d(axes)

    for index, subject_id in enumerate(subject_ids):
        row_index = index // n_cols
        col_index = index % n_cols
        ax = axes[row_index, col_index]
        image_path = output_dir / f"subject_{subject_id:02d}_comparison_bar.png"
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.set_title(f"Subject {subject_id}")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

