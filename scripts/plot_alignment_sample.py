#!/usr/bin/env python3
"""Plot one sample's omega-ky power and SPT contribution-score planes."""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from spt.model import load_checkpoint
from spt.physics import dispersion_mask, similarity_scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True); p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", required=True); p.add_argument("--device", default="cpu")
    args = p.parse_args(); device = torch.device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device); model.eval()
    record = torch.load(args.data, map_location="cpu", weights_only=False)[args.index]
    points = np.asarray(record["filter_cloud_points"], np.float32)[:model.config.num_points]
    selected = torch.from_numpy(points[:, model.config.feature_indices]).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(selected)
        scores = similarity_scores(features["local"], features["global_max"])[0].cpu().numpy()
        prediction = float(model.regressor(features["global"])[0, 0])
    on_shell, _ = dispersion_mask(points)
    kx, ky, omega, power = points[:, 1], points[:, 2], points[:, 3], points[:, 4]
    scale = 8 + 50 * (power - power.min()) / (np.ptp(power) + 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4), sharex=True, sharey=True)
    first = axes[0].scatter(ky, omega, c=np.log10(np.maximum(power, 1e-30)), s=scale, cmap="viridis")
    fig.colorbar(first, ax=axes[0], label="log10 power")
    second = axes[1].scatter(ky, omega, c=scores, s=scale, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].scatter(ky[on_shell], omega[on_shell], s=scale[on_shell] + 10,
                    facecolors="none", edgecolors="black", linewidths=.45, label="on shell")
    fig.colorbar(second, ax=axes[1], label="cosine similarity")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Selected spectral power"); axes[1].set_title("SPT contribution score")
    for ax in axes: ax.set_xlabel(r"$k_y$ (rad m$^{-1}$)"); ax.grid(alpha=.15)
    axes[0].set_ylabel(r"$\omega$ (rad s$^{-1}$)")
    fig.suptitle(f"true Hs={float(record['wave_height']):.2f} m, predicted={prediction:.2f} m; marker=dispersion shell")
    fig.tight_layout(); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")


if __name__ == "__main__": main()
