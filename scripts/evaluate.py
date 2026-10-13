#!/usr/bin/env python3
"""Evaluate an SPT checkpoint and export per-sample predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import SpectralPointDataset
from src.metrics import regression_metrics
from src.model import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True); p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(); device = torch.device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device); model.eval()
    dataset = SpectralPointDataset(args.data, model.config.num_points, "kxkywp", augment=False)
    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    truth, prediction, names = [], [], []
    with torch.no_grad():
        for batch in loader:
            prediction.extend(model(batch["point_cloud"].to(device)).cpu().tolist())
            truth.extend(batch["wave_height"].tolist()); names.extend(batch["npy_file"])
    result = regression_metrics(truth, prediction); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("npy_file", "true_hs", "pred_hs", "error"))
        writer.writerows((n, t, p, p - t) for n, t, p in zip(names, truth, prediction))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
