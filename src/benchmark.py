"""Run the same training on 1 vs 2 (or N) GPUs and emit the scaling table.

    python -m src.benchmark --data data/darcy_train.npz --gpus 1,2 --epochs 30

For each world size K it launches K training processes (rank 0..K-1) wired up
through environment variables -- the same path torchrun uses, but without the
torchrun rendezvous, which sidesteps a macOS-only IPv6 loopback hang and works
identically with NCCL on a real multi-GPU box (e.g. Kaggle's 2x T4).

It parses each run's metrics JSON and prints a Markdown table of epoch time,
throughput, final val relL2, and speed-up -- paste it straight into the README.
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_world(k: int, args, metrics_path: Path) -> dict:
    """Launch K training processes and return rank 0's metrics dict."""
    common = [
        sys.executable, "-m", "src.train",
        "--data", str(args.data),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--modes", str(args.modes),
        "--width", str(args.width),
        "--seed", str(args.seed),
        "--out", str(Path(args.workdir) / f"fno_w{k}.pt"),
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
        raise RuntimeError(f"world_size={k}: a process exited non-zero: {codes}")
    return json.loads(metrics_path.read_text())


def format_table(results: list[dict]) -> str:
    # Speed-up is relative to the smallest world size that succeeded (normally 1 GPU).
    base = min(results, key=lambda r: r["world_size"])["throughput_samples_per_s"]
    lines = [
        "| Setup | GPUs | epoch time (s) | throughput (samples/s) | val relL2 | speed-up |",
        "|-------|------|----------------|------------------------|-----------|----------|",
    ]
    for r in results:
        k = r["world_size"]
        label = "1x GPU" if k == 1 else f"{k}x GPU (DDP)"
        speedup = r["throughput_samples_per_s"] / base
        lines.append(
            f"| {label} | {k} | {r['mean_epoch_time_s']:.2f} | "
            f"{r['throughput_samples_per_s']:,.0f} | {r['final_val_relL2']:.4f} | "
            f"{speedup:.2f}x |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--gpus", type=str, default="1,2", help="comma-separated world sizes, e.g. 1,2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16, help="per-process batch size")
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", type=Path, default=Path("scaling.md"))
    args = p.parse_args()

    worlds = [int(x) for x in args.gpus.split(",")]
    with tempfile.TemporaryDirectory() as tmp:
        args.workdir = tmp
        results, failed = [], []
        for k in worlds:
            print(f"\n=== world_size = {k} ===", flush=True)
            try:
                m = run_world(k, args, Path(tmp) / f"metrics_w{k}.json")
            except Exception as e:  # keep partial results if one config fails
                print(f"  world_size={k} FAILED: {e}")
                failed.append(k)
                continue
            results.append(m)
            print(f"  epoch {m['mean_epoch_time_s']:.2f}s  "
                  f"{m['throughput_samples_per_s']:,.0f} samples/s  "
                  f"val relL2 {m['final_val_relL2']:.4f}")

    if not results:
        raise SystemExit("all benchmark runs failed — see errors above")
    table = format_table(results)
    print("\n" + table)
    if failed:
        print(f"\nNOTE: world sizes {failed} failed; table shows successful runs only.")
    args.out.write_text(table + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
