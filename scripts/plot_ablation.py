#!/usr/bin/env python3
"""Plot RMSE across Top-N or model-scale ablation CSV rows."""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--csv", required=True)
    p.add_argument("--x", required=True, help="Column used as categorical x axis")
    p.add_argument("--output", required=True); args = p.parse_args(); frame = pd.read_csv(args.csv)
    metrics = [column for column in ("rmse", "rmse_0.5_2.5", "rmse_2.5_4.0", "rmse_gt4") if column in frame]
    ax = frame.plot(x=args.x, y=metrics, marker="o", figsize=(6.5, 4.2))
    ax.set_ylabel("RMSE (m)"); ax.grid(alpha=.2); ax.figure.tight_layout()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output, dpi=300, bbox_inches="tight")


if __name__ == "__main__": main()
