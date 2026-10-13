#!/usr/bin/env python3
"""Train the paper configuration or a controlled SPT ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from spt.data import FEATURE_SETS, SpectralPointDataset
from spt.metrics import regression_metrics
from spt.model import SPT, SPTConfig


def evaluate(model, loader, device):
    model.eval(); truth, prediction, names = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["point_cloud"].to(device)).cpu()
            prediction.extend(pred.tolist()); truth.extend(batch["wave_height"].tolist())
            names.extend(batch["npy_file"])
    return np.asarray(truth), np.asarray(prediction), names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-pt", required=True); p.add_argument("--val-pt", required=True)
    p.add_argument("--out-dir", required=True); p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--seed", type=int, default=1981)
    p.add_argument("--num-points", type=int, default=512); p.add_argument("--feature-set", choices=FEATURE_SETS, default="kxkywp")
    p.add_argument("--embed-dim", type=int, default=128); p.add_argument("--heads", type=int, default=8)
    p.add_argument("--blocks", type=int, default=2); p.add_argument("--attention", choices=("offset", "residual"), default="offset")
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    indices = FEATURE_SETS[args.feature_set]
    cfg = SPTConfig(input_dim=len(indices), embed_dim=args.embed_dim, heads=args.heads,
                    blocks=args.blocks, attention_type=args.attention, qk_norm=args.qk_norm,
                    num_points=args.num_points, feature_indices=indices)
    train_set = SpectralPointDataset(args.train_pt, args.num_points, args.feature_set, augment=True)
    val_set = SpectralPointDataset(args.val_pt, args.num_points, args.feature_set, augment=False)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device); model = SPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.SmoothL1Loss(); best = None
    config = {**vars(args), **cfg.to_dict()}; (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    metrics_file = out / "metrics.csv"
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in train_loader:
            prediction = model(batch["point_cloud"].to(device))
            loss = criterion(prediction, batch["wave_height"].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            truth, prediction, names = evaluate(model, val_loader, device)
            result = regression_metrics(truth, prediction)
            row = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"],
                   "rmse": result["rmse"], "mae": result["mae"], "bias": result["bias"], "cc": result["cc"]}
            exists = metrics_file.exists()
            with metrics_file.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                if not exists: writer.writeheader()
                writer.writerow(row)
            print(row, flush=True)
            if best is None or result["rmse"] < best["metrics"]["rmse"]:
                best = {"epoch": epoch, "metrics": result}
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": result,
                            "config": cfg.to_dict()}, out / "spt_best.pt")
                with (out / "val_predictions.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle); writer.writerow(("npy_file", "true_hs", "pred_hs", "error"))
                    writer.writerows((n, t, pr, pr - t) for n, t, pr in zip(names, truth, prediction))
                (out / "best_record.json").write_text(json.dumps(best, indent=2) + "\n")


if __name__ == "__main__":
    main()
