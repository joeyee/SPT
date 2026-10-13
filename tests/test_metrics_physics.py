import numpy as np

from spt.metrics import regression_metrics
from spt.physics import dispersion_mask, feature_wnr, intrinsic_omega


def test_regression_metrics_and_bins():
    truth = np.array([1.0, 3.0, 5.0])
    pred = np.array([1.1, 2.8, 5.3])
    result = regression_metrics(truth, pred)
    assert result["n"] == 3
    assert set(result["bins"]) == {"low", "moderate", "high"}
    assert np.isclose(result["rmse"], np.sqrt((0.01 + 0.04 + 0.09) / 3))


def test_dispersion_mask_and_wnr():
    kx = np.array([0.04, 0.04])
    modeled = intrinsic_omega(np.abs(kx))
    points = np.column_stack((np.ones(2), kx, np.zeros(2), [modeled[0], modeled[1] + 0.2], np.ones(2)))
    mask, _ = dispersion_mask(points, harmonic_orders=(0,), tolerance=0.01)
    assert mask.tolist() == [True, False]
    assert np.isclose(feature_wnr([0.8, 0.2], mask), 4.0)
