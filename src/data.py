"""Portable CPU-backed dataset for SPT point-cloud records."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_SETS = {
    "kxkywp": (1, 2, 3, 4),
    "zid_kxkywp": (0, 1, 2, 3, 4),
    "wp": (3, 4),
    "kxkyw": (1, 2, 3),
}


class SpectralPointDataset(Dataset):
    def __init__(
        self,
        records: str | Path | Sequence[dict],
        num_points: int = 512,
        feature_set: str = "kxkywp",
        augment: bool = False,
        noise_std: float = 0.01,
        power_scale: float = 0.05,
    ) -> None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set}; choose from {sorted(FEATURE_SETS)}")
        self.records = (
            torch.load(records, map_location="cpu", weights_only=False)
            if isinstance(records, (str, Path)) else list(records)
        )
        self.num_points = num_points
        self.indices = torch.tensor(FEATURE_SETS[feature_set], dtype=torch.long)
        self.augment = augment
        self.noise_std = noise_std
        self.power_scale = power_scale

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        points = torch.as_tensor(np.asarray(record["filter_cloud_points"]), dtype=torch.float32)
        if len(points) < self.num_points:
            raise ValueError(f"record {index} has {len(points)} points; need {self.num_points}")
        points = points[: self.num_points].index_select(-1, self.indices).clone()
        if self.augment:
            points += torch.randn_like(points) * self.noise_std
            points[:, -1] *= 1 + (torch.rand(()) * 2 - 1) * self.power_scale
        return {
            "point_cloud": points,
            "wave_height": torch.tensor(float(record["wave_height"]), dtype=torch.float32),
            "npy_file": str(record.get("seq_file_name", "")),
            "wind_speed": torch.tensor(float(record.get("wind_speed", float("nan"))), dtype=torch.float32),
            "wind_dir": torch.tensor(float(record.get("wind_dir", float("nan"))), dtype=torch.float32),
        }
