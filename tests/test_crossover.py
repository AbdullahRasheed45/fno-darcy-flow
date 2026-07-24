"""Tests for the DDP-OOM / FSDP-survives crossover sweep.

The sweep itself needs multiple GPUs, but its two risky pieces are pure logic and
are tested here: classifying an out-of-memory failure (an *expected* outcome)
apart from a genuine bug, and rendering the table/plot.
"""

from src.crossover import classify, format_table, make_plot


def test_classify_success():
    assert classify(0, "") == "ok"


def test_classify_cuda_oom_variants():
    """A real torch OOM must be recorded as 'oom', not as a crash."""
    msgs = [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "RuntimeError: CUDA out of memory.",
        "torch.cuda.OutOfMemoryError: CUDA out of memory",
    ]
    for m in msgs:
        assert classify(1, m) == "oom", m


def test_classify_real_bug_is_not_mistaken_for_oom():
    """A genuine error must NOT be reported as a memory limit — that would
    silently turn a bug into a headline result."""
    assert classify(1, "AttributeError: 'NoneType' object has no attribute 'shape'") == "error"
    assert classify(1, "RuntimeError: No backend type associated with device type cpu") == "error"


def _row(width, parallel, status, mem=None, params=None):
    return {"width": width, "parallel": parallel, "status": status,
            "peak_mem_mb": mem, "params": params}


def test_table_marks_the_crossover():
    rows = [
        _row(64, "ddp", "ok", 800, 9_000_000), _row(64, "fsdp", "ok", 500, 9_000_000),
        _row(256, "ddp", "oom"), _row(256, "fsdp", "ok", 6100, 140_000_000),
    ]
    table = format_table(rows)
    assert "OOM" in table
    assert "9.0M" in table and "140.0M" in table
    assert "6,100 MB" in table


def test_crossover_plot_is_written(tmp_path):
    rows = [
        _row(64, "ddp", "ok", 800), _row(64, "fsdp", "ok", 500),
        _row(256, "ddp", "oom"), _row(256, "fsdp", "ok", 6100),
    ]
    out = tmp_path / "crossover.png"
    make_plot(rows, out)
    assert out.exists() and out.stat().st_size > 0
