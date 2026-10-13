"""Metrics and sea-state bins reported in the paper."""

from __future__ import annotations

import numpy as np


SEA_STATE_BINS = ("low", "moderate", "high")


def regression_metrics(truth, prediction) -> dict:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("truth and prediction must be equal-length 1-D arrays")
    error = prediction - truth
    result = {
        "n": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "cc": float(np.corrcoef(truth, prediction)[0, 1]) if len(truth) > 1 else float("nan"),
        "bins": {},
    }
    masks = {
        "low": (truth >= 0.5) & (truth < 2.5),
        "moderate": (truth >= 2.5) & (truth <= 4.0),
        "high": truth > 4.0,
    }
    for name in SEA_STATE_BINS:
        mask = masks[name]
        result["bins"][name] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else float("nan"),
        }
    return result
