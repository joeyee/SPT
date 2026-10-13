# Spectral Point Transformer (SPT)

[![Paper](https://img.shields.io/badge/IEEE%20JOE-10.1109%2FJOE.2026.3724015-blue)](https://doi.org/10.1109/JOE.2026.3724015)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-blue)](pyproject.toml)

Official PyTorch implementation of **“Spectral Point Transformer for
Significant Wave Height Estimation From Sea Clutter”**, by Yi Zhou, Li Wang,
Hang Su, and Tian Wang, published in the *IEEE Journal of Oceanic Engineering*.

- IEEE Xplore: <https://ieeexplore.ieee.org/document/11718839>
- DOI: <https://doi.org/10.1109/JOE.2026.3724015>
- Repository: <https://github.com/joeyee/SPT>

SPT estimates significant wave height, denoted by $H_s$, from a sparse set of
high-power points in the three-dimensional wavenumber–frequency spectrum of
X-band marine-radar sea clutter. The paper configuration retains the strongest
512 spectral points and represents each point as `[kx, ky, omega, power]`.

## Highlights

- Sparse spectral representation: only a small fraction of the full 3-D
  spectrum is used for regression.
- Lightweight transformer: MLP token embedding, QK-normalized offset
  attention, cross-layer feature concatenation, and permutation-invariant
  max/mean pooling.
- Physics-oriented interpretation: learned point contributions can be compared
  with finite-depth dispersion shells through the feature-domain WNR analysis.
- Paper-supporting utilities: Top-N extraction, training, evaluation, classical
  spectral-SNR baseline, ablations, and plotting scripts.

## Repository layout

```text
src/                         Core model, data, spectrum, metrics, and physics
scripts/                     Preprocessing, splitting, training, evaluation,
                             physical analysis, and figure generation
configs/paper.json           Architecture and optimization configuration
tests/                       Unit tests independent of the protected dataset
```

## Installation

```bash
git clone https://github.com/joeyee/SPT.git
cd SPT

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest -q
```

SPT requires Python 3.9 or newer and PyTorch 2.1 or newer. The implementation
does not require `timm`.

## Quick start

The following example checks the public model interface using synthetic input.
It does not produce a scientifically meaningful wave-height estimate.

```python
import torch

from src.model import SPT, SPTConfig

config = SPTConfig(
    input_dim=4,
    num_points=512,
    embed_dim=128,
    heads=8,
    blocks=2,
    attention_type="offset",
    qk_norm=True,
)
model = SPT(config).eval()

# [batch, spectral_points, (kx, ky, omega, power)]
points = torch.randn(2, 512, 4)
with torch.no_grad():
    estimated_hs = model(points)

print(estimated_hs.shape)  # torch.Size([2])
```


## Workflow with an authorized dataset

The command-line tools accept paths supplied by the user and contain no
machine-specific data locations.

```bash
# Convert compatible 128 x 128 x 64 radar cubes to Top-1024 records.
python scripts/build_points.py \
  --file-list /path/to/authorized_file_list.txt \
  --output /path/to/all_points.pt

# Build an event-level split. The SNR CSV must contain
# npy_file,true_hs,sqrt_borge_snr.
python scripts/split_events.py \
  --records /path/to/all_points.pt \
  --snr-csv /path/to/snr.csv \
  --out-dir /path/to/split

# Train the published SPT architecture on data you are authorized to use.
python scripts/train.py \
  --train-pt /path/to/train.pt \
  --val-pt /path/to/val.pt \
  --out-dir outputs/spt

# Evaluate a checkpoint produced by the preceding command.
python scripts/evaluate.py \
  --checkpoint outputs/spt/spt_best.pt \
  --data /path/to/val.pt \
  --out-dir outputs/evaluation

# Analyze learned-feature alignment with dispersion shells.
python scripts/analyze_alignment.py \
  --checkpoint outputs/spt/spt_best.pt \
  --data /path/to/val.pt \
  --output outputs/evaluation/alignment.csv
```

## Reported paper configuration and results

The principal configuration is Top-512 `[kx, ky, omega, power]`, embedding
dimension 128, eight attention heads, two offset-attention blocks, and QK
normalization. It contains approximately 0.70 million trainable parameters.

The paper reports the following results on its event-level validation split:

| Metric | Reported value |
|---|---:|
| Pearson correlation coefficient | 0.97 |
| Overall RMSE | 0.23 m |
| RMSE, $0.5 \leq H_s < 2.5$ m | 0.19 m |
| RMSE, $2.5 \leq H_s \leq 4.0$ m | 0.32 m |
| RMSE, $H_s > 4.0$ m | 0.39 m |

Aggregate machine-readable records are provided under
[`artifacts/reported_results`](artifacts/reported_results). These values are
reported experimental results; exact numerical reproduction requires the
protected research data and is therefore outside the scope of this public
release.

## Data and model availability

The Shengshan raw marine-radar data are protected by data-ownership and usage
agreements and are **not distributed** in this repository. 

No pretrained checkpoint or model weights are released. The repository
provides the model definition and training/evaluation code so that users can
train checkpoints on their own authorized data.

## Limitations

- The paper evaluates SPT on a single coastal radar dataset; transfer to other
  sites, radar geometries, and environmental conditions requires validation.
- High-sea-state samples are limited, so performance in extreme conditions is
  less certain.
- The optimal number of spectral points depends on radar resolution, spectral
  dynamic range, and the target dataset.
- The released zero-current dispersion diagnostic is not a substitute for the
  complete current-corrected analysis used in the paper.
- SPT is research software and must not be used as the sole basis for
  navigation or safety-critical decisions.

## Citation

```bibtex
@article{zhou2026spt,
  author  = {Yi Zhou and Li Wang and Hang Su and Tian Wang},
  title   = {Spectral Point Transformer for Significant Wave Height Estimation From Sea Clutter},
  journal = {IEEE Journal of Oceanic Engineering},
  year    = {2026},
  doi     = {10.1109/JOE.2026.3724015}
}
```

## License

This project is released under the [MIT License](LICENSE).
