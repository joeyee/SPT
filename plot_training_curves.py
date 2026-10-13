#!/usr/bin/env python3
"""Plot training loss/RMSE curves, including QK-normalization comparisons."""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("csv", nargs="+")
    p.add_argument("--labels", nargs="+"); p.add_argument("--output", required=True); args = p.parse_args()
    labels = args.labels or [Path(path).parent.name for path in args.csv]
    if len(labels) != len(args.csv): raise ValueError("--labels count must match CSV count")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for path, label in zip(args.csv, labels):
        frame = pd.read_csv(path); axes[0].plot(frame.epoch, frame.loss, label=label)
        axes[1].plot(frame.epoch, frame.rmse, label=label)
    axes[0].set(xlabel="Epoch", ylabel="Training loss"); axes[1].set(xlabel="Epoch", ylabel="Validation RMSE (m)")
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")


if __name__ == "__main__": main()
