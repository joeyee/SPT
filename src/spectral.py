"""3-D spectrum construction and deterministic Top-N point selection."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RadarGrid:
    nx: int = 128
    ny: int = 128
    nt: int = 64
    dx: float = 7.5
    dy: float = 7.5
    dt: float = 2.5
    low_frequency_cutoff: float = 0.018


def power_spectrum(cube: np.ndarray, grid: RadarGrid) -> np.ndarray:
    """Return power per (kx, ky, omega) bin in m^2.

    This follows the research code normalization: PSD is divided by spectral
    increments and grid size, then multiplied back by the increments to obtain
    per-bin power.
    """
    cube = np.asarray(cube)
    expected = (grid.nx, grid.ny, grid.nt)
    if cube.shape != expected:
        raise ValueError(f"expected radar cube {expected}, got {cube.shape}")
    fft = np.fft.fftn(cube)
    scale = float((grid.nx * grid.ny * grid.nt) ** 2)
    return np.abs(fft) ** 2 / scale


def extract_spectral_points(
    cube: np.ndarray,
    zone_id: int,
    grid: RadarGrid | None = None,
    num_points: int = 1024,
) -> np.ndarray:
    """Extract power-ranked points ``[zone_id, kx, ky, omega, power]``.

    Only non-negative observed frequencies are eligible. Frequencies at or
    below the configured cutoff are removed before ranking. Ties are resolved
    deterministically by the flattened array index.
    """
    grid = grid or RadarGrid()
    if num_points < 1:
        raise ValueError("num_points must be positive")
    spectrum = np.fft.fftshift(power_spectrum(cube, grid), axes=(0, 1))
    kx = np.fft.fftshift(np.fft.fftfreq(grid.nx, grid.dx) * 2 * np.pi)
    ky = np.fft.fftshift(np.fft.fftfreq(grid.ny, grid.dy) * 2 * np.pi)
    omega = np.fft.fftfreq(grid.nt, grid.dt)[: grid.nt // 2] * 2 * np.pi
    positive = spectrum[:, :, : grid.nt // 2].copy()
    positive[:, :, omega <= grid.low_frequency_cutoff] = 0.0
    flat = positive.reshape(-1)
    n = min(num_points, flat.size)
    candidate = np.argpartition(flat, flat.size - n)[-n:]
    order = np.lexsort((candidate, -flat[candidate]))
    chosen = candidate[order]
    ix, iy, iw = np.unravel_index(chosen, positive.shape)
    return np.column_stack(
        (np.full(n, int(zone_id)), kx[ix], ky[iy], omega[iw], flat[chosen])
    ).astype(np.float32)


def load_radar_npy(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load the Shengshan ``.npy`` dictionary and reconstruct data if needed."""
    obj = np.load(path, allow_pickle=True).item()
    cube = obj.get("data")
    if cube is None:
        fft = obj.get("fft")
        if fft is None:
            raise ValueError(f"{path}: expected 'data' or 'fft'")
        cube = np.fft.ifftn(fft).real
    return np.asarray(cube), obj


def infer_zone_id(path: str | Path, metadata: dict | None = None) -> int:
    if metadata and metadata.get("zone_class") is not None:
        return int(np.asarray(metadata["zone_class"]).reshape(-1)[0])
    match = re.search(r"_x_y_data_(\d+)-", Path(path).name)
    if not match:
        raise ValueError(f"cannot infer zone id from {path}")
    return int(match.group(1))
