#!/usr/bin/env python3
"""Compute per-sample feature WNR and dispersion-shell alignment."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch

from spt.model import load_checkpoint
from spt.physics import dispersion_mask, feature_wnr, similarity_scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(); device = torch.device(args.device); model, _ = load_checkpoint(args.checkpoint, device); model.eval()
    records = torch.load(args.data, map_location="cpu", weights_only=False); rows = []
    with torch.no_grad():
        for record in records:
            full = np.asarray(record["filter_cloud_points"], np.float32)[:model.config.num_points]
            selected = torch.from_numpy(full[:, model.config.feature_indices]).unsqueeze(0).to(device)
            features = model.forward_features(selected); prediction = float(model.regressor(features["global"])[0, 0])
            scores = similarity_scores(features["local"], features["global_max"])[0].cpu().numpy()
            mask, distance = dispersion_mask(full)
            rows.append({"npy_file": Path(str(record.get("seq_file_name", ""))).name,
                         "true_hs": float(record["wave_height"]), "pred_hs": prediction,
                         "feature_wnr": feature_wnr(scores, mask), "on_shell_points": int(mask.sum()),
                         "on_shell_fraction": float(mask.mean()), "median_shell_distance": float(np.median(distance))})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"saved {len(rows)} rows to {output}")


if __name__ == "__main__": main()
