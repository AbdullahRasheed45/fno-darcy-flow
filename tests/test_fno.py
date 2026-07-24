import torch
from src.models.fno import FNO2d, SpectralConv2d


def test_spectral_conv_shape():
    layer = SpectralConv2d(4, 8, modes1=6, modes2=6)
    out = layer(torch.randn(2, 4, 32, 32))
    assert out.shape == (2, 8, 32, 32)


def test_fno_forward_and_resolution_invariance():
    model = FNO2d(modes=8, width=16, layers=2)
    for grid in (32, 48):
        out = model(torch.randn(2, grid, grid))
        assert out.shape == (2, grid, grid)


def test_fno_backward():
    model = FNO2d(modes=8, width=16, layers=2)
    loss = model(torch.randn(2, 32, 32)).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_fno_forward_under_autocast():
    """Regression: the spectral layer must run under AMP autocast without hitting
    an unsupported complex-half kernel. Mirrors the --amp training path."""
    model = FNO2d(modes=8, width=16, layers=2)
    with torch.autocast(device_type="cpu", enabled=True):
        out = model(torch.randn(2, 32, 32))
    assert out.shape == (2, 32, 32)
    assert torch.isfinite(out).all()
