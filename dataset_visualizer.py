#!/usr/bin/env python3
"""
Dataset Visualization Utility
=============================

Generates visual summaries for all datasets currently used in the project.

Outputs:
    - Per-dataset subfolders in `visuals/`
    - Class distribution bar charts (train/test)
    - Representative sample grids for image datasets
    - Mean images per class for image datasets
    - Scatter plots for synthetic datasets (train/test)

Usage:
    python dataset_visualizer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# Use a non-interactive backend (important for headless environments)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
VISUALS_DIR = PROJECT_ROOT / "visuals"

DATASETS: Dict[str, Dict] = {
    "mnist_1_vs_7": {
        "type": "mnist",
        "classes": [1, 7],
        "class_names": {
            1: "Digit 1",
            7: "Digit 7",
        },
    },
    "mnist_6_vs_8": {
        "type": "mnist",
        "classes": [6, 8],
        "class_names": {
            6: "Digit 6",
            8: "Digit 8",
        },
    },
    "fmnist_sandals_vs_boots": {
        "type": "fmnist",
        "classes": [5, 9],
        "class_names": {
            5: "Sandal (5)",
            9: "Ankle Boot (9)",
        },
    },
    "fmnist_shirts_vs_coats": {
        "type": "fmnist",
        "classes": [6, 8],
        "class_names": {
            6: "Shirt (6)",
            8: "Coat (8)",
        },
    },
    "synthetic_a": {
        "type": "synthetic",
        "variant": "A",
        "class_names": {
            0: "Spiral A",
            1: "Spiral B",
        },
    },
    "synthetic_b": {
        "type": "synthetic",
        "variant": "B",
        "class_names": {
            0: "Tight Cluster",
            1: "Wide Cluster",
        },
    },
    "synthetic_c": {
        "type": "synthetic",
        "variant": "C",
        "class_names": {
            0: "Positive Sine",
            1: "Negative Sine",
        },
    },
}

# Label name helpers for image datasets
MNIST_LABEL_NAMES = {digit: f"Digit {digit}" for digit in range(10)}
FMNIST_LABEL_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

IMAGE_SIDE = 28  # MNIST & Fashion-MNIST image side length
RNG = np.random.default_rng(seed=42)
COLOR_MAP = ListedColormap(["#1f77b4", "#d62728"])


# -----------------------------------------------------------------------------
# Data loading utilities
# -----------------------------------------------------------------------------

def load_mnist_data(classes: List[int]) -> Dict[str, np.ndarray]:
    """Load extracted MNIST data for the specified digit pair."""
    data_dir = PROJECT_ROOT / "extracted_datasets" / "extracted_data_mnist"

    def _read_split(prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df_class_a = pd.read_csv(data_dir / f"{prefix}_{classes[0]}.csv")
        df_class_b = pd.read_csv(data_dir / f"{prefix}_{classes[1]}.csv")
        df = pd.concat([df_class_a, df_class_b], ignore_index=True)
        labels_raw = df["label"].to_numpy()
        features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        labels_binary = (labels_raw == classes[1]).astype(np.int32)
        return features, labels_binary, labels_raw

    x_train, y_train, y_train_raw = _read_split("train")
    x_test, y_test, y_test_raw = _read_split("test")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_raw": y_train_raw,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_raw": y_test_raw,
    }


def load_fmnist_data(classes: List[int]) -> Dict[str, np.ndarray]:
    """Load extracted Fashion-MNIST data for the specified class pair."""
    data_dir = PROJECT_ROOT / "extracted_datasets" / "extracted_data_fmnist"

    def _read_split(prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df_class_a = pd.read_csv(data_dir / f"{prefix}{classes[0]}.csv")
        df_class_b = pd.read_csv(data_dir / f"{prefix}{classes[1]}.csv")
        df = pd.concat([df_class_a, df_class_b], ignore_index=True)
        labels_raw = df["label"].to_numpy()
        features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        labels_binary = (labels_raw == classes[1]).astype(np.int32)
        return features, labels_binary, labels_raw

    x_train, y_train, y_train_raw = _read_split("train")
    x_test, y_test, y_test_raw = _read_split("test")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_raw": y_train_raw,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_raw": y_test_raw,
    }


def generate_synthetic_data(variant: str, n_train: int = 1000, n_test: int = 1000) -> Dict[str, np.ndarray]:
    """Generate synthetic datasets following project definitions."""
    rng = np.random.default_rng(seed=42)

    if variant == "A":
        theta_train = np.sqrt(rng.random(n_train)) * 2 * np.pi
        theta_test = np.sqrt(rng.random(n_test)) * 2 * np.pi

        r_a = 2 * theta_train + np.pi
        r_b = -2 * theta_train - np.pi

        x1_train = np.column_stack((r_a * np.cos(theta_train), r_a * np.sin(theta_train)))
        x2_train = np.column_stack((r_b * np.cos(theta_train), r_b * np.sin(theta_train)))

        r_a_test = 2 * theta_test + np.pi
        r_b_test = -2 * theta_test - np.pi

        x1_test = np.column_stack((r_a_test * np.cos(theta_test), r_a_test * np.sin(theta_test)))
        x2_test = np.column_stack((r_b_test * np.cos(theta_test), r_b_test * np.sin(theta_test)))

        x_train = np.vstack((x1_train, x2_train))
        y_train = np.hstack((np.zeros(n_train, dtype=np.int32), np.ones(n_train, dtype=np.int32)))
        x_test = np.vstack((x1_test, x2_test))
        y_test = np.hstack((np.zeros(n_test, dtype=np.int32), np.ones(n_test, dtype=np.int32)))

    elif variant == "B":
        x1_train, _ = make_blobs(
            n_samples=n_train,
            centers=[[2, 2]],
            cluster_std=0.8,
            random_state=42,
        )
        x2_train, _ = make_blobs(
            n_samples=n_train,
            centers=[[2, 2]],
            cluster_std=1.5,
            random_state=43,
        )
        x1_test, _ = make_blobs(
            n_samples=n_test,
            centers=[[2, 2]],
            cluster_std=0.8,
            random_state=44,
        )
        x2_test, _ = make_blobs(
            n_samples=n_test,
            centers=[[2, 2]],
            cluster_std=1.5,
            random_state=45,
        )

        x_train = np.vstack((x1_train, x2_train))
        y_train = np.hstack((np.zeros(n_train, dtype=np.int32), np.ones(n_train, dtype=np.int32)))
        x_test = np.vstack((x1_test, x2_test))
        y_test = np.hstack((np.zeros(n_test, dtype=np.int32), np.ones(n_test, dtype=np.int32)))

    elif variant == "C":
        theta_train = np.linspace(0, np.pi, n_train)
        theta_test = np.linspace(0, np.pi, n_test)

        x1_train = np.column_stack((theta_train, np.sin(theta_train)))
        x2_train = np.column_stack((theta_train, -np.sin(theta_train)))
        x1_test = np.column_stack((theta_test, np.sin(theta_test)))
        x2_test = np.column_stack((theta_test, -np.sin(theta_test)))

        noise_train = rng.normal(scale=0.1, size=(2 * n_train, 2))
        noise_test = rng.normal(scale=0.1, size=(2 * n_test, 2))

        x_train = np.vstack((x1_train, x2_train)) + noise_train
        y_train = np.hstack((np.zeros(n_train, dtype=np.int32), np.ones(n_train, dtype=np.int32)))
        x_test = np.vstack((x1_test, x2_test)) + noise_test
        y_test = np.hstack((np.zeros(n_test, dtype=np.int32), np.ones(n_test, dtype=np.int32)))

    else:
        raise ValueError(f"Unknown synthetic variant '{variant}'.")

    train_idx = rng.permutation(len(x_train))
    test_idx = rng.permutation(len(x_test))

    return {
        "x_train": x_train[train_idx],
        "y_train": y_train[train_idx],
        "x_test": x_test[test_idx],
        "y_test": y_test[test_idx],
    }


def load_dataset(dataset_key: str, config: Dict) -> Dict[str, np.ndarray]:
    """Dispatch dataset loading based on configuration."""
    dataset_type = config["type"]

    if dataset_type == "mnist":
        return load_mnist_data(config["classes"])
    if dataset_type == "fmnist":
        return load_fmnist_data(config["classes"])
    if dataset_type == "synthetic":
        return generate_synthetic_data(config["variant"])

    raise ValueError(f"Unsupported dataset type '{dataset_type}' for '{dataset_key}'.")


# -----------------------------------------------------------------------------
# Visualization utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_class_distribution(
    labels: np.ndarray,
    title: str,
    output_path: Path,
    class_name_map: Dict[int, str],
) -> None:
    """Plot a class distribution bar chart."""
    unique, counts = np.unique(labels, return_counts=True)
    label_names = [class_name_map.get(int(lbl), str(lbl)) for lbl in unique]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(label_names, counts, color="#4f85e5")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mean_images(
    features: np.ndarray,
    raw_labels: np.ndarray,
    classes: List[int],
    class_name_map: Dict[int, str],
    title: str,
    output_path: Path,
) -> None:
    """Plot mean image per class."""
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]

    for ax, label in zip(axes, classes):
        mask = raw_labels == label
        if not np.any(mask):
            ax.axis("off")
            ax.set_title(f"No samples for {class_name_map.get(label, label)}")
            continue

        mean_image = features[mask].mean(axis=0).reshape(IMAGE_SIDE, IMAGE_SIDE)
        ax.imshow(mean_image, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{class_name_map.get(label, label)}\n{mask.sum()} samples")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sample_grid(
    features: np.ndarray,
    raw_labels: np.ndarray,
    classes: List[int],
    class_name_map: Dict[int, str],
    title: str,
    output_path: Path,
    samples_per_class: int = 6,
) -> None:
    """Plot a grid of sample images per class."""
    n_rows = len(classes)
    fig, axes = plt.subplots(
        n_rows,
        samples_per_class,
        figsize=(samples_per_class * 2, n_rows * 2),
    )

    axes = np.array(axes, dtype=object)
    axes = axes.reshape(n_rows, samples_per_class)

    for row_idx, label in enumerate(classes):
        indices = np.where(raw_labels == label)[0]
        if indices.size == 0:
            for col_idx in range(samples_per_class):
                axes[row_idx, col_idx].axis("off")
            continue

        if indices.size < samples_per_class:
            chosen = RNG.choice(indices, size=samples_per_class, replace=True)
        else:
            chosen = RNG.choice(indices, size=samples_per_class, replace=False)

        for col_idx, idx in enumerate(chosen):
            image = features[idx].reshape(IMAGE_SIDE, IMAGE_SIDE)
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(class_name_map.get(label, str(label)), rotation=90, labelpad=10)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scatter(
    features: np.ndarray,
    labels: np.ndarray,
    class_name_map: Dict[int, str],
    title: str,
    output_path: Path,
) -> None:
    """Scatter plot for synthetic datasets."""
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        features[:, 0],
        features[:, 1],
        c=labels,
        cmap=COLOR_MAP,
        s=10,
        alpha=0.7,
        edgecolor="none",
    )

    legend_handles = []
    for class_id in sorted(np.unique(labels)):
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=class_name_map.get(int(class_id), f"Class {class_id}"),
                markerfacecolor=COLOR_MAP(class_id / max(len(class_name_map) - 1, 1)),
                markersize=8,
            )
        )

    ax.legend(handles=legend_handles, loc="best", frameon=False)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def visualize_dataset(dataset_key: str, config: Dict) -> None:
    """Generate visualizations for a single dataset."""
    print(f"▶ Visualizing {dataset_key} ...")

    data = load_dataset(dataset_key, config)
    dataset_dir = ensure_dir(VISUALS_DIR / dataset_key)

    if config["type"] in {"mnist", "fmnist"}:
        class_names = config["class_names"]
        label_name_reference = MNIST_LABEL_NAMES if config["type"] == "mnist" else FMNIST_LABEL_NAMES

        # Class distributions
        plot_class_distribution(
            data["y_train_raw"],
            f"{dataset_key} – Train distribution",
            dataset_dir / "train_class_distribution.png",
            label_name_reference,
        )
        plot_class_distribution(
            data["y_test_raw"],
            f"{dataset_key} – Test distribution",
            dataset_dir / "test_class_distribution.png",
            label_name_reference,
        )

        # Mean images per class
        plot_mean_images(
            data["x_train"],
            data["y_train_raw"],
            classes=config["classes"],
            class_name_map=label_name_reference,
            title=f"{dataset_key} – Mean images (train)",
            output_path=dataset_dir / "train_mean_images.png",
        )
        plot_mean_images(
            data["x_test"],
            data["y_test_raw"],
            classes=config["classes"],
            class_name_map=label_name_reference,
            title=f"{dataset_key} – Mean images (test)",
            output_path=dataset_dir / "test_mean_images.png",
        )

        # Sample grids (train set)
        plot_sample_grid(
            data["x_train"],
            data["y_train_raw"],
            classes=config["classes"],
            class_name_map=class_names,
            title=f"{dataset_key} – Sample grid (train)",
            output_path=dataset_dir / "train_sample_grid.png",
        )

        # Sample grids (test set)
        plot_sample_grid(
            data["x_test"],
            data["y_test_raw"],
            classes=config["classes"],
            class_name_map=class_names,
            title=f"{dataset_key} – Sample grid (test)",
            output_path=dataset_dir / "test_sample_grid.png",
        )

    elif config["type"] == "synthetic":
        class_names = config["class_names"]

        plot_class_distribution(
            data["y_train"],
            f"{dataset_key} – Train distribution",
            dataset_dir / "train_class_distribution.png",
            class_names,
        )
        plot_class_distribution(
            data["y_test"],
            f"{dataset_key} – Test distribution",
            dataset_dir / "test_class_distribution.png",
            class_names,
        )

        plot_scatter(
            data["x_train"],
            data["y_train"],
            class_names,
            title=f"{dataset_key} – Train scatter",
            output_path=dataset_dir / "train_scatter.png",
        )
        plot_scatter(
            data["x_test"],
            data["y_test"],
            class_names,
            title=f"{dataset_key} – Test scatter",
            output_path=dataset_dir / "test_scatter.png",
        )

    else:
        raise ValueError(f"Unsupported dataset type '{config['type']}'.")

    print(f"✓ Saved visuals to {dataset_dir}")


def run(dataset_filter: List[str] | None = None) -> None:
    """Run visualization across datasets, optionally filtering by keys."""
    VISUALS_DIR.mkdir(exist_ok=True)
    targets = dataset_filter or list(DATASETS.keys())

    missing = [name for name in targets if name not in DATASETS]
    if missing:
        names = ", ".join(missing)
        raise KeyError(f"Unknown dataset keys requested: {names}")

    for dataset_key in targets:
        visualize_dataset(dataset_key, DATASETS[dataset_key])

    print("\nAll requested dataset visuals generated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize project datasets.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional list of dataset keys to visualize. Defaults to all datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.datasets)


if __name__ == "__main__":
    main()


