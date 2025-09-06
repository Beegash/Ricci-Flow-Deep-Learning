#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal DNN trainer for binary classification + layer activations export.
Designed for the Ricci Flow vs DNN geometry project.

Usage examples:
  # Train on your own arrays saved as an .npz
  python main.py --data data/my_binary_dataset.npz --epochs 30 --depth 5 --width 50 \
      --batch-size 128 --lr 1e-3 --out run1

  # Quick demo using sklearn's make_moons (synthetic); ignores --data
  python main.py --demo --epochs 20 --depth 5 --width 50 --out demo_run

The .npz is expected to contain: X_train, y_train, X_val, y_val, X_test, y_test
Shapes: X_* -> (n_samples, n_features), y_* -> (n_samples,) with labels in {0,1}
"""

import argparse
import os
import json
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

try:
    from sklearn.datasets import make_moons
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# ------------------------
# Utilities
# ------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------------
# Model
# ------------------------
@dataclass
class MLPConfig:
    in_dim: int
    depth: int = 5
    width: int = 50
    dropout: float = 0.0


class MLP(nn.Module):
    """Simple MLP with ReLU and a single-logit output for binary classification.
    Layer names are registered to make activation capture easy.
    """
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = cfg.in_dim
        self.hidden: nn.ModuleList = nn.ModuleList()

        for i in range(cfg.depth):
            lin = nn.Linear(last_dim, cfg.width)
            self.hidden.append(lin)
            layers.append(lin)
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(p=cfg.dropout))
            last_dim = cfg.width

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, 1)  # single logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.head(x)
        return x  # raw logit

    def layer_names(self) -> List[str]:
        names = []
        for i, layer in enumerate(self.body):
            # keep only Linear layers as "hidden_i"
            if isinstance(layer, nn.Linear):
                names.append(f"hidden_{len(names)+1}")
        names.append("logit")
        return names


# ------------------------
# Data loading
# ------------------------

def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    expected = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [k for k in expected if k not in data]
    if missing:
        raise ValueError(f"Missing arrays in NPZ: {missing}. Expected keys: {expected}")
    return {k: data[k] for k in expected}


def make_demo(n_samples: int = 4000, noise: float = 0.25, test_size: float = 0.25, val_size: float = 0.25):
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn not available; install it or provide --data .npz")
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    # splits: test, val, train
    n_test = int(len(X) * test_size)
    n_val = int(len(X) * val_size)
    X_test, y_test = X[:n_test], y[:n_test]
    X_val, y_val = X[n_test:n_test+n_val], y[n_test:n_test+n_val]
    X_train, y_train = X[n_test+n_val:], y[n_test+n_val:]
    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "X_val": X_val.astype(np.float32),
        "y_val": y_val.astype(np.int64),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
    }


# ------------------------
# Training / Eval
# ------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    early_patience: int = 8,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    patience = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {ep}/{epochs} [train]", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        avg_train = total_loss / len(train_loader.dataset)

        # validation with progress bar
        model.eval()
        val_loss = 0.0
        vbar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {ep}/{epochs} [val]", leave=False)
        with torch.no_grad():
            for xb, yb in vbar:
                xb = xb.to(device)
                yb = yb.to(device).float().unsqueeze(1)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                vbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        avg_val = val_loss / len(val_loader.dataset)

        print(f"Epoch {ep:03d} | train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        # early stopping
        if avg_val < best_val - 1e-6:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_patience:
                print(f"Early stopping at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device):
    model.eval()
    X_t = to_tensor(X, device)
    logits = model(X_t).cpu().numpy().ravel()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    acc = accuracy_score(y, preds)
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = float("nan")
    return {"acc": float(acc), "auc": float(auc)}


# ------------------------
# Activation capture
# ------------------------

def capture_layer_outputs(model: MLP, X: np.ndarray, device: torch.device) -> Dict[str, np.ndarray]:
    """Return dict of layer_name -> activations (numpy) for given X.
    Captures outputs *after* each Linear layer's ReLU (i.e., the activated features),
    and the final raw logit.
    """
    model.eval()
    activations: Dict[str, np.ndarray] = {}

    hooks = []
    linear_idx = 0

    def make_hook(name):
        def hook(module, inp, out):
            # out is the linear output; we want ReLU output which follows next in self.body
            activations[name] = out.detach().cpu().numpy()
        return hook

    # Register hooks on Linear layers; we will apply ReLU manually when exporting
    for layer in model.body:
        if isinstance(layer, nn.Linear):
            linear_idx += 1
            hooks.append(layer.register_forward_hook(make_hook(f"hidden_{linear_idx}_prerelu")))

    X_t = to_tensor(X, device)

    with torch.no_grad():
        logits = model(X_t)

    # Apply ReLU to pre-outputs to mirror forward pass
    relu = nn.ReLU()
    out_dict: Dict[str, np.ndarray] = {}
    for k in sorted(activations.keys(), key=lambda s: int(s.split("_")[1])):
        arr = activations[k]
        arr = relu(torch.tensor(arr)).numpy()
        out_dict[k.replace("_prerelu", "")] = arr

    out_dict["logit"] = logits.cpu().numpy()

    # cleanup hooks
    for h in hooks:
        h.remove()

    return out_dict


# ------------------------
# Main
# ------------------------

def main():
    p = argparse.ArgumentParser(description="Train a simple DNN and export layer activations")
    p.add_argument("--data", type=str, default=None, help="Path to .npz with X/y splits")
    p.add_argument("--demo", action="store_true", help="Use sklearn.make_moons synthetic data")
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--width", type=int, default=50)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="run")
    args = p.parse_args()

    set_seed(args.seed)
    print(f"[setup] device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')} out={os.path.abspath(args.out)}")

    if args.demo:
        data = make_demo()
    elif args.data is not None:
        data = load_npz(args.data)
    else:
        raise SystemExit("Provide --data path to .npz or use --demo to generate synthetic data.")

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    out_dir = os.path.abspath(args.out)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = MLPConfig(in_dim=X_train.shape[1], depth=args.depth, width=args.width, dropout=args.dropout)
    model = MLP(cfg).to(device)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(to_tensor(X_train, device), torch.tensor(y_train, device=device)),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(to_tensor(X_val, device), torch.tensor(y_val, device=device)),
        batch_size=args.batch_size, shuffle=False
    )

    # Train
    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_patience=args.patience,
    )

    # Evaluate
    test_metrics = evaluate(model, X_test, y_test, device)
    val_metrics = evaluate(model, X_val, y_val, device)
    train_metrics = evaluate(model, X_train, y_train, device)

    print("\nFinal metrics:")
    print(json.dumps({
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }, indent=2))

    # Save model + metrics
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"train": train_metrics, "val": val_metrics, "test": test_metrics}, f, indent=2)

    # Export activations for X_test (we will use these for KNN graphs & Forman-Ricci later)
    acts = capture_layer_outputs(model, X_test, device)
    # Also export raw X_test and y_test for later analysis cohesion
    np.savez_compressed(
        os.path.join(out_dir, "layer_outputs_test.npz"),
        **acts,
        X_test=X_test,
        y_test=y_test,
    )
    print(f"Saved layer activations to {os.path.join(out_dir, 'layer_outputs_test.npz')}")


if __name__ == "__main__":
    main()
