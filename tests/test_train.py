import json
import sys

import numpy as np

from src import train


def _make_dataset(path, n=16, grid=16, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, grid, grid)).astype("float32")
    u = rng.standard_normal((n, grid, grid)).astype("float32")
    np.savez(path, a=a, u=u)


def test_relative_l2_zero_on_exact_match():
    import torch
    x = torch.randn(4, 8, 8)
    assert train.relative_l2(x, x).item() < 1e-6


def test_train_single_process_runs(tmp_path, monkeypatch):
    """End-to-end smoke test of the training loop (the loop had no test before)."""
    data = tmp_path / "d.npz"
    _make_dataset(data)
    ckpt = tmp_path / "m.pt"
    metrics = tmp_path / "metrics.json"
    argv = [
        "prog", "--data", str(data), "--epochs", "2", "--batch-size", "4",
        "--modes", "6", "--width", "8", "--layers", "2", "--num-workers", "0",
        "--out", str(ckpt), "--metrics-out", str(metrics),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()

    assert ckpt.exists()
    m = json.loads(metrics.read_text())
    assert m["world_size"] == 1
    assert m["throughput_samples_per_s"] > 0
    assert np.isfinite(m["final_val_relL2"])

    import torch
    saved = torch.load(ckpt, map_location="cpu")
    assert set(saved) == {"model", "config", "stats"}
    assert saved["config"] == {"modes": 6, "width": 8, "layers": 2}
