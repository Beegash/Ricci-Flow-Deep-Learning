#!/usr/bin/env python3
"""End-to-end experiment runner for Ricci Flow DNN vs KNN analyses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset

from dataset_loaders import available_datasets, load_dataset
from main import (
    MLP,
    MLPConfig,
    capture_layer_outputs,
    evaluate,
    set_seed,
    to_tensor,
    train,
)
from ricci_analysis import ricci_for_k


def _save_npz(path: Path, splits: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **splits)


def _train_dnn(
    dataset_dir: Path,
    splits: Dict[str, np.ndarray],
    *,
    depth: int,
    width: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], Path, Path]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MLPConfig(in_dim=int(splits["X_train"].shape[1]), depth=depth, width=width, dropout=dropout)
    model = MLP(cfg).to(device)

    train_ds = TensorDataset(to_tensor(splits["X_train"], device), torch.tensor(splits["y_train"], device=device))
    val_ds = TensorDataset(to_tensor(splits["X_val"], device), torch.tensor(splits["y_val"], device=device))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        early_patience=patience,
    )

    metrics = {
        "train": evaluate(model, splits["X_train"], splits["y_train"], device),
        "val": evaluate(model, splits["X_val"], splits["y_val"], device),
        "test": evaluate(model, splits["X_test"], splits["y_test"], device),
    }

    model_path = dataset_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    activations = capture_layer_outputs(model, splits["X_test"], device)
    act_path = dataset_dir / "layer_outputs_test.npz"
    np.savez_compressed(
        act_path,
        **activations,
        X_test=splits["X_test"],
        y_test=splits["y_test"],
    )

    return metrics, activations, model_path, act_path


def _evaluate_knn(
    splits: Dict[str, np.ndarray],
    k_values: Iterable[int],
) -> Dict[str, object]:
    results: List[Dict[str, float]] = []
    best = None
    best_val = -np.inf
    for k in sorted(set(int(k) for k in k_values)):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(splits["X_train"], splits["y_train"])
        val_pred = clf.predict(splits["X_val"])
        test_pred = clf.predict(splits["X_test"])
        val_acc = accuracy_score(splits["y_val"], val_pred)
        test_acc = accuracy_score(splits["y_test"], test_pred)
        record = {
            "k": int(k),
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
        }
        results.append(record)
        if val_acc > best_val:
            best_val = val_acc
            best = record
    summary = {
        "grid": results,
        "best": best,
    }
    return summary


def _prepare_layers_for_ricci(
    activations: Dict[str, np.ndarray],
    *,
    sample_size: int | None,
    seed: int,
) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    keys = sorted(k for k in activations.keys() if k.startswith("hidden_"))
    total = activations[keys[0]].shape[0] if keys else 0
    if sample_size is not None and total > sample_size:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=sample_size, replace=False))
    else:
        idx = np.arange(total)
    layers = [activations[k][idx].astype(np.float32) for k in keys]
    return layers, keys, idx


def _run_ricci_sweep(
    layers: List[np.ndarray],
    k_values: Iterable[int],
    *,
    sample_pairs: int,
    allow_disconnected: bool,
    disconnected_penalty: float,
    prefer_exact: bool,
    show_progress: bool,
) -> Dict[str, object]:
    sweep: Dict[int, Dict[str, object]] = {}
    best_k = None
    best_rho = float("inf")
    for k in sorted(set(int(k) for k in k_values)):
        try:
            res = ricci_for_k(
                layers,
                k,
                sample_pairs,
                allow_disconnected,
                disconnected_penalty,
                prefer_exact=prefer_exact,
                progress=show_progress,
            )
        except Exception as e:
            res = {"error": str(e)}
        sweep[k] = {
            "rho": float(res.get("rho", float("nan"))) if isinstance(res, dict) else float("nan"),
            "z": float(res.get("z", float("nan"))) if isinstance(res, dict) else float("nan"),
            "Ric": [float(x) for x in res.get("Ric", [])] if isinstance(res, dict) else [],
            "g": [float(x) for x in res.get("g", [])] if isinstance(res, dict) else [],
            **({"error": res.get("error")} if isinstance(res, dict) and "error" in res else {}),
        }
        rho = sweep[k]["rho"]
        if not np.isnan(rho) and rho < best_rho:
            best_rho = rho
            best_k = k
    return {
        "results": sweep,
        "best_k": best_k,
        "best_rho": best_rho if best_k is not None else None,
    }


def run_experiments(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    experiment_summaries = {}

    for dataset_name in args.datasets:
        print(f"\n=== Dataset: {dataset_name} ===")
        # Stage: data
        if not args.skip_data:
            splits, meta = load_dataset(dataset_name, seed=args.seed)
            dataset_dir = out_root / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            npz_path = dataset_dir / "dataset.npz"
            _save_npz(npz_path, splits)
            print(f"Saved splits to {npz_path}")
        else:
            dataset_dir = out_root / dataset_name
            npz_path = dataset_dir / "dataset.npz"
            if not npz_path.exists():
                raise SystemExit(f"--skip-data is set but {npz_path} not found")
            with np.load(npz_path) as f:
                splits = {k: f[k] for k in ["X_train","y_train","X_val","y_val","X_test","y_test"]}
            meta = {}

        # Stage: dnn
        if not args.skip_dnn:
            dnn_metrics, activations, model_path, act_path = _train_dnn(
                dataset_dir,
                splits,
                depth=args.depth,
                width=args.width,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                seed=args.seed,
            )
            print(f"DNN metrics: test_acc={dnn_metrics['test']['acc']:.4f} auc={dnn_metrics['test']['auc']:.4f}")
        else:
            model_path = dataset_dir / "model.pt"
            act_path = dataset_dir / "layer_outputs_test.npz"
            if not act_path.exists():
                raise SystemExit(f"--skip-dnn is set but {act_path} not found")
            with np.load(act_path) as f:
                activations = {k: f[k] for k in f.files if k.startswith("hidden_") or k in ("X_test","y_test")}
            dnn_metrics = {}

        # Stage: knn
        if not args.skip_knn:
            knn_summary = _evaluate_knn(splits, args.knn_k)
            if knn_summary["best"] is not None:
                print(f"Best KNN k={knn_summary['best']['k']} val_acc={knn_summary['best']['val_acc']:.4f}")
        else:
            knn_summary = {"grid": [], "best": None}

        # Stage: ricci
        if not args.skip_ricci:
            layers, hidden_keys, ricci_idx = _prepare_layers_for_ricci(
                activations,
                sample_size=args.ricci_sample_size,
                seed=args.seed,
            )
            ricci_summary = _run_ricci_sweep(
                layers,
                args.ricci_k,
                sample_pairs=args.ricci_sample_pairs,
                allow_disconnected=args.ricci_allow_disconnected,
                disconnected_penalty=args.ricci_disconnected_penalty,
                prefer_exact=args.ricci_prefer_exact,
                show_progress=bool(args.ricci_progress),
            )
            if ricci_summary["best_k"] is not None:
                print(f"Ricci best k={ricci_summary['best_k']} rho={ricci_summary['best_rho']:.4f}")
        else:
            hidden_keys = []
            ricci_idx = []
            ricci_summary = {"results": {}, "best_k": None, "best_rho": None}

        experiment_summaries[dataset_name] = {
            "meta": meta,
            "paths": {
                "npz": str(npz_path),
                "model": str(model_path),
                "activations": str(act_path),
            },
            "dnn": dnn_metrics,
            "knn": knn_summary,
            "ricci": {
                "config": {
                    "sample_indices_count": int(len(ricci_idx)),
                    "sample_pairs": int(args.ricci_sample_pairs),
                    "allow_disconnected": bool(args.ricci_allow_disconnected),
                    "disconnected_penalty": float(args.ricci_disconnected_penalty),
                    "prefer_exact": bool(args.ricci_prefer_exact),
                },
                "summary": ricci_summary,
                "hidden_keys": hidden_keys,
            },
        }

    summary_path = out_root / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(experiment_summaries, f, indent=2)
    print(f"\nSaved experiment summary to {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Ricci Flow experiments across multiple datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "mnist_0_vs_8",
            "fashion_top_vs_dress",
            "cifar_cat_vs_dog",
            "breast_cancer",
            "annulus_vs_disk",
            "torus_vs_sphere",
        ],
        help=f"Datasets to run. Available: {', '.join(available_datasets())}",
    )
    parser.add_argument("--out-root", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--knn-k", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

    parser.add_argument("--ricci-k", type=int, nargs="+", default=[10, 15, 20, 30, 40, 50])
    parser.add_argument("--ricci-sample-size", type=int, default=1500,
                        help="Number of test samples to keep for Ricci analysis (per dataset)")
    parser.add_argument("--ricci-sample-pairs", type=int, default=20000)
    parser.add_argument("--ricci-allow-disconnected", action="store_true")
    parser.add_argument("--ricci-disconnected-penalty", type=float, default=1e6)
    parser.add_argument("--ricci-prefer-exact", action="store_true",
                        help="Use exact shortest paths instead of sampling when feasible")
    parser.add_argument("--ricci-progress", action="store_true",
                        help="Show per-layer progress bars during Ricci sweep")

    # Stage skipping flags for partial runs
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset split regeneration; reuse saved npz")
    parser.add_argument("--skip-dnn", action="store_true", help="Skip DNN training; reuse saved activations")
    parser.add_argument("--skip-knn", action="store_true", help="Skip KNN sweep")
    parser.add_argument("--skip-ricci", action="store_true", help="Skip Ricci sweep")

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    run_experiments(parser.parse_args())
