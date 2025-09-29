#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
from pathlib import Path


def build_command(suite: dict, out_root: Path, ricci_progress: bool) -> str:
    parts = [
        "python", "run_experiments.py",
        "--out-root", str(out_root),
        "--datasets", *suite["datasets"],
        "--epochs", str(suite.get("epochs", 20)),
        "--depth", str(suite.get("depth", 5)),
        "--width", str(suite.get("width", 50)),
        "--dropout", str(suite.get("dropout", 0.0)),
        "--batch-size", str(suite.get("batch_size", 128)),
        "--lr", str(suite.get("lr", 1e-3)),
        "--weight-decay", str(suite.get("weight_decay", 0.0)),
        "--patience", str(suite.get("patience", 8)),
        "--knn-k", *[str(k) for k in suite.get("knn_k", [5,7,9])],
        "--ricci-k", *[str(k) for k in suite.get("ricci_k", [10,20])],
        "--ricci-sample-size", str(suite.get("ricci_sample_size", 1000)),
        "--ricci-sample-pairs", str(suite.get("ricci_sample_pairs", 10000)),
    ]
    if suite.get("ricci_allow_disconnected", False):
        parts.append("--ricci-allow-disconnected")
    if suite.get("ricci_prefer_exact", False):
        parts.append("--ricci-prefer-exact")
    if ricci_progress:
        parts.append("--ricci-progress")
    return " ".join(shlex.quote(p) for p in parts)


def main():
    ap = argparse.ArgumentParser(description="Run predefined experiment test suites from JSON config")
    ap.add_argument("--config", type=str, default=str(Path(__file__).with_name("test_cases.json")))
    ap.add_argument("--out-root", type=str, default="experiments_suites")
    ap.add_argument("--select", type=str, nargs="*", default=None,
                    help="Optional list of suite names to run (defaults to all)")
    ap.add_argument("--ricci-progress", action="store_true", help="Show progress bars during Ricci stage")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    data = json.loads(cfg_path.read_text())
    suites = data.get("suites", [])

    selected_names = set(args.select) if args.select else None
    for suite in suites:
        name = suite.get("name", "suite")
        if selected_names and name not in selected_names:
            continue
        out_root = Path(args.out_root) / name
        out_root.mkdir(parents=True, exist_ok=True)
        cmd = build_command(suite, out_root, args.ricci_progress)
        print(f"\n[run] {name}\n$ {cmd}")
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()


