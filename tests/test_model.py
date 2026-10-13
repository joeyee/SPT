import torch

from src.model import SPT, SPTConfig, load_checkpoint


def test_output_shape_and_permutation_invariance():
    torch.manual_seed(1)
    model = SPT(SPTConfig(embed_dim=32, heads=4, blocks=2, num_points=16)).eval()
    points = torch.randn(3, 16, 4)
    permutation = torch.randperm(16)
    with torch.no_grad():
        original = model(points)
        permuted = model(points[:, permutation])
    assert original.shape == (3,)
    torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)


def test_checkpoint_round_trip(tmp_path):
    original = SPT()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": original.state_dict(), "config": original.config.to_dict()}, checkpoint)
    restored, payload = load_checkpoint(checkpoint)
    assert payload["config"]["input_dim"] == 4
    assert restored.config.num_points == 512
    assert sum(p.numel() for p in restored.parameters()) == 701_057
    for expected, actual in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
