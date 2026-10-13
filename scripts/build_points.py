#!/usr/bin/env python3
"""Convert Shengshan radar cubes into SPT point-record files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from spt.spectral import RadarGrid, extract_spectral_points, infer_zone_id, load_radar_npy


def clamp_frames(cube: np.ndarray, multiple: float = 10.0) -> np.ndarray:
    mean = cube.mean(axis=(0, 1), keepdims=True)
    return np.clip(cube, 0.0, mean * multiple)


def metadata_value(meta: dict, *keys: str, default=float("nan")):
    for key in keys:
        value = meta.get(key)
        if value is not None:
            return value
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", required=True, help="Text file containing one .npy path per line")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--num-points", type=int, default=1024, help="Store enough points for N ablations")
    parser.add_argument("--dx", type=float, default=7.5)
    parser.add_argument("--dy", type=float, default=7.5)
    parser.add_argument("--dt", type=float, default=2.5)
    parser.add_argument("--cutoff", type=float, default=0.018)
    parser.add_argument("--no-clamp", action="store_true", help="Disable paper's per-frame [0, 10*mean] clamp")
    args = parser.parse_args()

    paths = [Path(line.strip()) for line in Path(args.file_list).read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    grid = RadarGrid(dx=args.dx, dy=args.dy, dt=args.dt, low_frequency_cutoff=args.cutoff)
    records = []
    for index, path in enumerate(paths, 1):
        cube, meta = load_radar_npy(path)
        if not args.no_clamp:
            cube = clamp_frames(cube)
        record = {
            "seq_file_name": str(path.resolve()),
            "filter_cloud_points": extract_spectral_points(
                cube, infer_zone_id(path, meta), grid, args.num_points
            ),
            "wave_height": float(metadata_value(meta, "height", "wave_height")),
            "wave_period": float(metadata_value(meta, "period", "wave_period")),
            "wave_dir": float(metadata_value(meta, "dir", "wave_dir")),
            "wind_speed": float(metadata_value(meta, "wind_speed")),
            "wind_dir": float(metadata_value(meta, "wind_dir")),
        }
        records.append(record)
        if index % 50 == 0:
            print(f"processed {index}/{len(paths)}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output)
    print(f"saved {len(records)} records to {output}")


if __name__ == "__main__":
    main()
