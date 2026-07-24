"""Distributed-path tests: multi-rank DDP/FSDP training, grad accumulation, reporting.

The FSDP test is skipped on macOS: this torch build's FSDP device autodetect
queries the Apple MPS backend and crashes there. It runs for real on Linux CI.
"""

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.benchmark import format_table, make_plot

REPO = Path(__file__).resolve().parents[1]
IS_MAC = platform.system() == "Darwin"


def _has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def _make_data(path, n=16, grid=16):
    rng = np.random.default_rng(0)
    np.savez(path,
             a=rng.standard_normal((n, grid, grid)).astype("float32"),
             u=rng.standard_normal((n, grid, grid)).astype("float32"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(world, parallel, data, out, metrics, extra=None):
    """Run src.train across `world` processes; return their exit codes."""
    cmd = [sys.executable, "-m", "src.train", "--data", str(data),
           "--epochs", "1", "--batch-size", "4", "--modes", "6", "--width", "8",
           "--layers", "2", "--num-workers", "0", "--parallel", parallel,
           "--out", str(out), "--metrics-out", str(metrics)] + (extra or [])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    if world == 1:
        return [subprocess.run(cmd, cwd=REPO, env=env).returncode]
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()), WORLD_SIZE=str(world))
    if IS_MAC:
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    procs = []
    for r in range(world):
        e = env.copy()
        e.update(RANK=str(r), LOCAL_RANK=str(r))
        procs.append(subprocess.Popen(cmd, cwd=REPO, env=e))
    return [p.wait() for p in procs]


def test_ddp_two_rank_training(tmp_path):
    """Two real DDP ranks train an epoch and agree on a reduced metric."""
    data = tmp_path / "d.npz"
    _make_data(data)
    metrics = tmp_path / "m.json"
    codes = _launch(2, "ddp", data, tmp_path / "m.pt", metrics)
    assert codes == [0, 0], f"DDP ranks exited {codes}"
    m = json.loads(metrics.read_text())
    assert m["world_size"] == 2 and m["parallel"] == "ddp"
    assert m["throughput_samples_per_s"] > 0
    assert np.isfinite(m["final_val_relL2"])


@pytest.mark.skipif(not _has_cuda(), reason="FSDP requires a CUDA accelerator: torch raises "
                                            "'FSDP needs a non-CPU accelerator device' on CPU")
def test_fsdp_two_rank_training(tmp_path):
    """FSDP shards params/grads/optimizer state across 2 ranks and still trains + checkpoints."""
    data = tmp_path / "d.npz"
    _make_data(data)
    metrics = tmp_path / "m.json"
    ckpt = tmp_path / "m.pt"
    codes = _launch(2, "fsdp", data, ckpt, metrics)
    assert codes == [0, 0], f"FSDP ranks exited {codes}"
    m = json.loads(metrics.read_text())
    assert m["world_size"] == 2 and m["parallel"] == "fsdp"
    assert np.isfinite(m["final_val_relL2"])
    # The sharded state dict must be gathered into a full checkpoint on rank 0.
    import torch
    saved = torch.load(ckpt, map_location="cpu")
    assert set(saved) == {"model", "config", "stats"}
    assert any(k.endswith("w1") for k in saved["model"]), "spectral weights missing from gather"


def test_grad_accum_raises_effective_batch(tmp_path):
    """Gradient accumulation multiplies the effective batch without more memory."""
    data = tmp_path / "d.npz"
    _make_data(data)
    metrics = tmp_path / "m.json"
    codes = _launch(1, "ddp", data, tmp_path / "m.pt", metrics, extra=["--grad-accum", "2"])
    assert codes == [0]
    m = json.loads(metrics.read_text())
    assert m["grad_accum"] == 2
    assert m["effective_batch_size"] == m["per_rank_batch_size"] * m["world_size"] * 2


def test_fsdp_on_cpu_raises_actionable_error():
    """FSDP can't initialise on CPU; fail with a clear message, not torch's internal one."""
    import torch
    from src.train import wrap_parallel
    with pytest.raises(RuntimeError, match="requires CUDA"):
        wrap_parallel(torch.nn.Linear(4, 4), "fsdp", world=2, device=torch.device("cpu"))


def test_unknown_parallel_strategy_rejected():
    import torch
    from src.train import wrap_parallel
    with pytest.raises(ValueError):
        wrap_parallel(torch.nn.Linear(4, 4), "nope", world=2, device=torch.device("cpu"))


def _fake(world, parallel, tp, mem=None):
    return {"world_size": world, "parallel": parallel, "mean_epoch_time_s": 1.0,
            "throughput_samples_per_s": tp, "peak_mem_mb": mem, "final_val_relL2": 0.01}


def test_format_table_reports_speedup_and_memory():
    rows = [_fake(1, "ddp", 100, 900), _fake(2, "ddp", 190, 900), _fake(2, "fsdp", 150, 500)]
    table = format_table(rows)
    assert "peak mem/GPU" in table
    assert "1.90×" in table          # DDP scaling vs the 1-GPU baseline
    assert "FSDP ×2" in table


def test_scaling_plot_is_written(tmp_path):
    out = tmp_path / "scaling.png"
    make_plot([_fake(1, "ddp", 100), _fake(2, "ddp", 190), _fake(4, "ddp", 350)], out)
    assert out.exists() and out.stat().st_size > 0
