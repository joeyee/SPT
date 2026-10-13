import numpy as np

from src.spectral import RadarGrid, extract_spectral_points


def test_top_points_are_sorted_and_well_formed():
    grid = RadarGrid(nx=8, ny=8, nt=8, dx=2.0, dy=2.0, dt=1.0, low_frequency_cutoff=0.0)
    rng = np.random.default_rng(4)
    cube = rng.normal(size=(8, 8, 8))
    points = extract_spectral_points(cube, zone_id=2, grid=grid, num_points=12)
    assert points.shape == (12, 5)
    assert np.all(points[:, 0] == 2)
    assert np.all(np.diff(points[:, 4]) <= 0)
    assert np.all(points[:, 3] > 0)


def test_selection_is_deterministic():
    grid = RadarGrid(nx=8, ny=8, nt=8, low_frequency_cutoff=0.0)
    cube = np.ones((8, 8, 8))
    np.testing.assert_array_equal(
        extract_spectral_points(cube, 1, grid, 10),
        extract_spectral_points(cube, 1, grid, 10),
    )
