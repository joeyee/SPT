"""Physical-consistency diagnostics for SPT features."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def intrinsic_omega(k, depth: float = 39.0, harmonic_order: int = 0, gravity: float = 9.81):
    multiplier = harmonic_order + 1.0
    k = np.asarray(k, dtype=float)
    return multiplier * np.sqrt(gravity * k / multiplier * np.tanh(k * depth / multiplier))


def fold_frequency(omega, nyquist: float):
    folded = np.mod(omega, 2 * nyquist)
    return np.where(folded <= nyquist, folded, 2 * nyquist - folded)


def dispersion_mask(
    points: np.ndarray,
    depth: float = 39.0,
    dt: float = 2.5,
    tolerance: float | None = None,
    harmonic_orders: tuple[int, ...] = (0, 1),
    current: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Classify full-format ``[zid,kx,ky,w,p]`` points as on/off shell."""
    points = np.asarray(points)
    kx, ky, observed = points[:, 1], points[:, 2], np.abs(points[:, 3])
    k = np.hypot(kx, ky)
    nyquist = np.pi / dt
    dw = 2 * np.pi / (64 * dt)
    tolerance = dw / 2 if tolerance is None else tolerance
    distance = np.full(len(points), np.inf)
    doppler = kx * current[0] + ky * current[1]
    for order in harmonic_orders:
        modeled = fold_frequency(intrinsic_omega(k, depth, order) + doppler, nyquist)
        distance = np.minimum(distance, np.abs(observed - modeled))
    return distance <= tolerance, distance


def similarity_scores(local_features: torch.Tensor, pooled_features: torch.Tensor) -> torch.Tensor:
    local = F.normalize(local_features, dim=-1)
    pooled = F.normalize(pooled_features, dim=-1).unsqueeze(1)
    return (local * pooled).sum(dim=-1)


def feature_wnr(scores, on_shell, eps: float = 1e-12) -> float:
    """Positive-cosine feature WNR used for the paper's alignment analysis."""
    scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
    on_shell = np.asarray(on_shell, dtype=bool)
    return float(scores[on_shell].sum() / (scores[~on_shell].sum() + eps))


def borge_snr(
    cube: np.ndarray,
    dx: float = 7.5,
    dy: float = 7.5,
    dt: float = 2.5,
    depth: float = 39.0,
    current: tuple[float, float] = (0.0, 0.0),
    alpha: float = 0.1,
    frequency_start_bin: int = 2,
    shell_half_width_bins: float = 1.0,
) -> dict[str, float]:
    """Classical Borge-style spectral SNR used as a validation baseline.

    The caller may supply a separately estimated surface current. The paper's
    final analysis used iterative Senet current estimates; zero current is an
    explicit, useful diagnostic but is not identical to that reported result.
    """
    ny, nx, nt = cube.shape
    kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
    omega = np.fft.fftfreq(nt, dt) * 2 * np.pi
    dkx, dky, dw = abs(kx[1] - kx[0]), abs(ky[1] - ky[0]), abs(omega[1] - omega[0])
    power = np.abs(np.fft.fftn(cube)) ** 2 / (dkx * dky * dw * (nx * ny * nt) ** 2)
    power[:, :, 0] = 0.0
    kx_grid = kx[None, :, None]
    ky_grid = ky[:, None, None]
    w_grid = omega[None, None, :]
    magnitude = np.hypot(kx_grid, ky_grid)
    doppler = kx_grid * current[0] + ky_grid * current[1]
    valid = w_grid > dw
    masks = []
    for order in (0, 1):
        shell = intrinsic_omega(magnitude, depth, order) + doppler
        masks.append(valid & (np.abs(w_grid - shell) <= shell_half_width_bins * dw))
    fundamental = masks[0]
    harmonic = masks[1] & ~fundamental
    background = np.maximum(power - power * fundamental - power * harmonic, 0.0)
    positive_stop = nt // 2
    integrated_wave = (power * fundamental)[:, :, frequency_start_bin:positive_stop].sum(axis=2)
    maximum = float(integrated_wave.max())
    strong = integrated_wave >= alpha * maximum if maximum > 0 else np.zeros_like(integrated_wave, bool)
    wave_energy = float(integrated_wave[strong].sum())
    background_energy = float(background[:, :, frequency_start_bin:positive_stop].sum())
    ratio = wave_energy / (background_energy + 1e-30)
    return {"snr": ratio, "sqrt_snr": float(np.sqrt(max(ratio, 0))),
            "wave_energy": wave_energy, "background_energy": background_energy}
