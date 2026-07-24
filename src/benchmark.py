"""Scaling benchmark: run the same training across world sizes and strategies.

    # throughput scaling curve across 1, 2, 4 GPUs
    python -m src.benchmark --data data/darcy_train.npz --gpus 1,2,4 --epochs 30 --amp

    # DDP (replicate) vs FSDP (shard) at the same world size -- compare peak memory
    python -m src.benchmark --data data/darcy_train.npz --gpus 2 --parallel ddp,fsdp --amp

For each configuration it launches K training processes (rank 0..K-1) wired up
through environment variables -- the same path torchrun uses, but without the
torchrun rendezvous, which sidesteps a macOS-only IPv6 loopback hang and works
identically with NCCL on a real multi-GPU box.

Emits a Markdown table (epoch time, throughput, peak memory, val relL2,
speed-up) and a scaling plot with an ideal-linear reference line.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

# Model size presets: "large" is compute-bound enough that communication stops
# dominating, which is where data-parallel scaling approaches linear.
PRESETS = {"small": {"modes": 12, "width": 32}, "large": {"modes": 24, "width": 64}}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_config(k: int, parallel: str, args, metrics_path: Path) -> dict:
    """Launch K training processes for one (world_size, strategy) pair; return rank-0 metrics."""
    common = [
        sys.executable, "-m", "src.train",
        "--data", str(args.data),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--parallel", parallel,
        "--modes", str(args.modes),
        "--width", str(args.width),
        "--seed", str(args.seed),
        "--out", str(Path(args.workdir) / f"fno_{parallel}_w{k}.pt"),
        "--metrics-out", str(metrics_path),
    ]
    if args.amp:
        common.append("--amp")

    env = os.environ.copy()
    if k == 1:
        procs = [subprocess.Popen(common, env=env)]
    else:
        port = _free_port()
        env.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "WORLD_SIZE": str(k)})
        if platform.system() == "Darwin":
            env["GLOO_SOCKET_IFNAME"] = "lo0"  # macOS CPU/gloo loopback
        procs = []
        for rank in range(k):
            e = env.copy()
            e.update({"RANK": str(rank), "LOCAL_RANK": str(rank)})
            procs.append(subprocess.Popen(common, env=e))

    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"{parallel} world_size={k}: a process exited non-zero: {codes}")
    return json.loads(metrics_path.read_text())


def format_table(results: list[dict]) -> str:
    # Speed-up is relative to the smallest world size that succeeded (normally 1 GPU).
    base = min(results, key=lambda r: r["world_size"])["throughput_samples_per_s"]
    has_mem = any(r.get("peak_mem_mb") for r in results)
    head = "| Strategy | GPUs | epoch time (s) | throughput (samples/s) |"
    sep = "|----------|------|----------------|------------------------|"
    if has_mem:
        head += " peak mem/GPU (MB) |"
        sep += "-------------------|"
    head += " val relL2 | speed-up |"
    sep += "-----------|----------|"
    lines = [head, sep]
    for r in results:
        k, mode = r["world_size"], r.get("parallel", "ddp")
        label = "single GPU" if k == 1 else f"{mode.upper()} ×{k}"
        row = (f"| {label} | {k} | {r['mean_epoch_time_s']:.2f} | "
               f"{r['throughput_samples_per_s']:,.0f} |")
        if has_mem:
            mem = r.get("peak_mem_mb")
            row += f" {mem:,.0f} |" if mem else " – |"
        row += (f" {r['final_val_relL2']:.4f} | "
                f"{r['throughput_samples_per_s'] / base:.2f}× |")
        lines.append(row)
    return "\n".join(lines)


def make_plot(results: list[dict], out: Path) -> None:
    """Throughput vs GPU count, one line per strategy, against ideal linear scaling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_mode: dict[str, list[dict]] = {}
    for r in results:
        by_mode.setdefault(r.get("parallel", "ddp"), []).append(r)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    base = min(results, key=lambda r: r["world_size"])["throughput_samples_per_s"]
    all_k = sorted({r["world_size"] for r in results})
    ax.plot(all_k, [base * k for k in all_k], "k--", alpha=0.5, label="ideal (linear)")
    for mode, rows in sorted(by_mode.items()):
        rows = sorted(rows, key=lambda r: r["world_size"])
        ks = [r["world_size"] for r in rows]
        tp = [r["throughput_samples_per_s"] for r in rows]
        ax.plot(ks, tp, "o-", label=mode.upper())
        for k, t in zip(ks, tp):
            ax.annotate(f"{t / base:.2f}×", (k, t), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("GPUs")
    ax.set_ylabel("throughput (samples/s)")
    ax.set_title("FNO Darcy-flow: distributed training scaling")
    ax.set_xticks(all_k)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--gpus", type=str, default="1,2", help="comma-separated world sizes, e.g. 1,2,4")
    p.add_argument("--parallel", type=str, default="ddp",
                   help="comma-separated strategies to compare: ddp, fsdp, or ddp,fsdp")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16, help="per-process micro-batch size")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--size", choices=sorted(PRESETS), default="small",
                   help="model preset; 'large' is compute-bound and scales closer to linear")
    p.add_argument("--modes", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", type=Path, default=Path("scaling.md"))
    p.add_argument("--plot", type=Path, default=Path("docs/scaling.png"))
    args = p.parse_args()

    preset = PRESETS[args.size]
    args.modes = args.modes if args.modes is not None else preset["modes"]
    args.width = args.width if args.width is not None else preset["width"]

    worlds = [int(x) for x in args.gpus.split(",")]
    modes = [m.strip() for m in args.parallel.split(",")]
    print(f"model preset '{args.size}' (modes={args.modes}, width={args.width}); "
          f"world sizes {worlds}; strategies {modes}")

    with tempfile.TemporaryDirectory() as tmp:
        args.workdir = tmp
        results, failed = [], []
        for mode in modes:
            for k in worlds:
                # FSDP only differs from plain training when there is more than one rank.
                if mode == "fsdp" and k == 1 and "ddp" in modes:
                    continue
                print(f"\n=== {mode} · world_size = {k} ===", flush=True)
                try:
                    m = run_config(k, mode, args, Path(tmp) / f"metrics_{mode}_w{k}.json")
                except Exception as e:  # keep partial results if one config fails
                    print(f"  {mode} world_size={k} FAILED: {e}")
                    failed.append((mode, k))
                    continue
                results.append(m)
                mem = f"  peak {m['peak_mem_mb']:,.0f} MB" if m.get("peak_mem_mb") else ""
                print(f"  epoch {m['mean_epoch_time_s']:.2f}s  "
                      f"{m['throughput_samples_per_s']:,.0f} samples/s  "
                      f"val relL2 {m['final_val_relL2']:.4f}{mem}")

    if not results:
        raise SystemExit("all benchmark runs failed — see errors above")
    table = format_table(results)
    print("\n" + table)
    if failed:
        print(f"\nNOTE: {failed} failed; table shows successful runs only.")
    args.out.write_text(table + "\n")
    print(f"\nwrote {args.out}")
    try:
        make_plot(results, args.plot)
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
