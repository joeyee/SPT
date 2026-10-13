#!/usr/bin/env python3
"""Fit Hs = intercept + slope * sqrt(spectral SNR) on train only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spt.metrics import regression_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True); p.add_argument("--val-csv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    train, val = pd.read_csv(args.train_csv), pd.read_csv(args.val_csv)
    required = {"true_hs", "sqrt_borge_snr"}
    for name, frame in (("train", train), ("val", val)):
        missing = required - set(frame.columns)
        if missing: raise ValueError(f"{name} CSV missing {sorted(missing)}")
    design = np.column_stack((np.ones(len(train)), train["sqrt_borge_snr"].to_numpy(float)))
    intercept, slope = np.linalg.lstsq(design, train["true_hs"].to_numpy(float), rcond=None)[0]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    result = {"intercept": float(intercept), "slope": float(slope), "train": {}, "val": {}}
    for name, frame in (("train", train), ("val", val)):
        frame = frame.copy(); frame["pred_hs"] = intercept + slope * frame["sqrt_borge_snr"]
        result[name] = regression_metrics(frame["true_hs"], frame["pred_hs"])
        frame.to_csv(out / f"{name}_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
