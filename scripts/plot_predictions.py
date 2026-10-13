#!/usr/bin/env python3
"""Recreate the measured-versus-estimated Hs scatter figure."""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--csv", required=True); p.add_argument("--output", required=True)
    args = p.parse_args(); frame = pd.read_csv(args.csv)
    truth = frame["true_hs"].to_numpy(float); pred = frame["pred_hs"].to_numpy(float)
    rmse = np.sqrt(np.mean((pred - truth) ** 2)); cc = np.corrcoef(truth, pred)[0, 1]
    low, high = min(truth.min(), pred.min()), max(truth.max(), pred.max())
    fig, ax = plt.subplots(figsize=(5.2, 4.6)); ax.scatter(truth, pred, s=15, alpha=.65, edgecolors="none")
    ax.plot([low, high], [low, high], "k--", lw=1); ax.set(xlabel=r"Buoy $H_s$ (m)", ylabel=r"SPT $\hat{H}_s$ (m)")
    ax.text(.04, .95, f"CC={cc:.3f}\nRMSE={rmse:.3f} m", transform=ax.transAxes, va="top")
    ax.grid(alpha=.2); fig.tight_layout(); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")


if __name__ == "__main__": main()
