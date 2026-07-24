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


def test_no_complex_parameters():
    """Regression: AMP's GradScaler cannot unscale complex gradients on CUDA, so
    every learnable parameter must be real-dtyped (complex weights are stored as
    real [.., 2] tensors and reassembled with view_as_complex in forward)."""
    model = FNO2d(modes=8, width=16, layers=2)
    assert not any(p.is_complex() for p in model.parameters())


def test_amp_optimizer_step():
    """Regression for the GradScaler complex-gradient crash: a full
    autocast -> scale -> unscale/step -> update cycle must complete."""
    model = FNO2d(modes=8, width=16, layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    with torch.autocast(device_type="cpu", enabled=True):
        loss = model(torch.randn(2, 32, 32)).pow(2).mean()
    scaler.scale(loss).backward()
    scaler.step(opt)      # calls unscale_ over all params — would raise on complex grads
    scaler.update()
    assert all(torch.isfinite(p).all() for p in model.parameters())
