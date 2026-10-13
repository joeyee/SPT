#!/usr/bin/env python3
"""Leakage-safe event split with Hs and classical-SNR residual strata."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch


def event_key(path: str) -> str:
    """Group all observation zones acquired at the same timestamp."""
    return Path(path).stem.split("_x_y_data_")[0]


def hs_bin(height: float) -> str:
    return "low" if height < 2.5 else "moderate" if height <= 4.0 else "high"


def acquisition_month(path: str) -> str:
    match = re.search(r"(?:^|/)(\d{6})/(?:normal/)?", path)
    if match:
        return match.group(1)
    timestamp = re.search(r"(\d{4})-(\d{2})-\d{2}", Path(path).name)
    return "".join(timestamp.groups()) if timestamp else "unknown"


def fit_snr_residuals(records: list[dict], csv_path: str, quantile: float):
    frame = pd.read_csv(csv_path)
    if "variant" in frame:
        frame = frame[frame["variant"].astype(str) == "source"]
    allowed = {str(r["seq_file_name"]) for r in records}
    frame = frame[frame["npy_file"].astype(str).isin(allowed)].copy()
    frame = frame[np.isfinite(frame["sqrt_borge_snr"]) & np.isfinite(frame["true_hs"])]
    x, y = frame["sqrt_borge_snr"].to_numpy(float), frame["true_hs"].to_numpy(float)
    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    frame["residual"] = intercept + slope * x - y
    mapping = dict(zip(frame["npy_file"].astype(str), frame["residual"].astype(float)))
    return mapping, {"intercept": float(intercept), "slope": float(slope), "n": len(frame),
                     "quantile": quantile}


def make_events(records: list[dict], residuals: dict[str, float], quantile: float):
    groups = defaultdict(list)
    for record in records:
        groups[event_key(str(record["seq_file_name"]))].append(record)
    event_extremes = {}
    for key, members in groups.items():
        available = [residuals[str(r["seq_file_name"])] for r in members
                     if str(r["seq_file_name"]) in residuals]
        if available:
            event_extremes[key] = max(available, key=abs)
    magnitudes = [abs(value) for value in event_extremes.values()]
    threshold = float(np.quantile(magnitudes, quantile)) if magnitudes else math.inf
    events = []
    for key, members in sorted(groups.items()):
        height = float(np.median([float(r["wave_height"]) for r in members]))
        if key not in event_extremes:
            category = "missing"
        else:
            extreme = event_extremes[key]
            category = "normal" if abs(extreme) < threshold else ("pred_high" if extreme > 0 else "pred_low")
        events.append({"key": key, "records": members, "height": height,
                       "hs_bin": hs_bin(height), "residual_category": category,
                       "month": acquisition_month(str(members[0]["seq_file_name"])),
                       "stratum": f"{hs_bin(height)}|{category}"})
    return events, threshold


def choose_validation(events: list[dict], val_fraction: float, seed: int, searches: int):
    groups = defaultdict(list)
    for event in events:
        groups[event["stratum"]].append(event)
    target = round(len(events) * val_fraction)
    raw = {key: len(value) * val_fraction for key, value in groups.items()}
    allocations = {key: math.floor(value) for key, value in raw.items()}
    remainder_order = sorted(
        ((raw[key] - allocations[key], key, len(groups[key])) for key in groups),
        reverse=True,
    )
    for _, key, _ in remainder_order:
        if sum(allocations.values()) >= target:
            break
        allocations[key] += 1

    best = None
    for offset in range(searches):
        rng = random.Random(seed + offset)
        selected = []
        for label in sorted(groups):
            members = groups[label]
            shuffled = list(members)
            rng.shuffle(shuffled)
            selected.extend(shuffled[: allocations[label]])
        score = 0.0
        for field, weight in (("hs_bin", 4.0), ("residual_category", 4.0),
                              ("month", 1.0), ("stratum", 2.0)):
            overall = Counter(e[field] for e in events)
            chosen = Counter(e[field] for e in selected)
            score += weight * sum(
                abs(chosen[key] / len(selected) - count / len(events))
                for key, count in overall.items()
            )
        candidate = (score, seed + offset, {e["key"] for e in selected})
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best, allocations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True, help="One or more real-data .pt files")
    parser.add_argument("--snr-csv", required=True, help="CSV with npy_file, true_hs, sqrt_borge_snr")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=1 / 3)
    parser.add_argument("--outlier-quantile", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=1981)
    parser.add_argument("--searches", type=int, default=2000)
    args = parser.parse_args()

    records = []
    for path in args.records:
        records.extend(torch.load(path, map_location="cpu", weights_only=False))
    residuals, fit = fit_snr_residuals(records, args.snr_csv, args.outlier_quantile)
    events, threshold = make_events(records, residuals, args.outlier_quantile)
    (score, chosen_seed, val_keys), allocations = choose_validation(
        events, args.val_fraction, args.seed, args.searches
    )
    train = [r for e in events if e["key"] not in val_keys for r in e["records"]]
    val = [r for e in events if e["key"] in val_keys for r in e["records"]]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(train, out / "real_train.pt")
    torch.save(val, out / "real_val.pt")
    rows = [{"event_key": e["key"], "split": "val" if e["key"] in val_keys else "train",
             "hs": e["height"], "hs_bin": e["hs_bin"],
             "snr_residual_category": e["residual_category"], "month": e["month"],
             "sample_count": len(e["records"])}
            for e in events]
    with (out / "event_split.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {"real_events": len(events), "train_samples": len(train), "val_samples": len(val),
               "snr_fit": fit, "outlier_threshold": threshold, "selected_seed": chosen_seed,
               "balance_score": score, "allocations": allocations}
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
