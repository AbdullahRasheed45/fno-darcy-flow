"""Find the memory crossover: where DDP runs out of memory but FSDP still trains.

    python -m src.crossover --data data/darcy_train.npz --widths 64,128,192,256 --amp

This is the experiment that shows *why* sharded training exists. It grows the
model until data parallelism can no longer fit a replica on each GPU, while FSDP
-- which shards parameters, gradients and optimizer state across ranks -- keeps
going. An out-of-memory error from DDP is an EXPECTED result here, recorded as a
data point rather than treated as a crash.

Outputs a Markdown table and a memory-vs-model-size plot marking the crossover.
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

OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda oom",
               "tried to allocate", "no executable batch size")


def classify(returncode: int, stderr: str) -> str:
    """Map a training process outcome to ok / oom / error.

    OOM is the interesting signal, so it must be distinguished from ordinary
    failures (a bug would otherwise be silently reported as a memory limit).
    """
    if returncode == 0:
        return "ok"
    low = stderr.lower()
    if any(m in low for m in OOM_MARKERS):
        return "oom"
    return "error"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_case(width: int, parallel: str, args, workdir: str) -> dict:
    """Train briefly at a given model width; return {status, peak_mem_mb, params}."""
    metrics = Path(workdir) / f"m_{parallel}_{width}.json"
    cmd = [
        sys.executable, "-m", "src.train",
        "--data", str(args.data),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--modes", str(args.modes),
        "--width", str(width),
        "--layers", str(args.layers),
        "--parallel", parallel,
        "--num-workers", "0",
        "--out", str(Path(workdir) / f"ckpt_{parallel}_{width}.pt"),
        "--metrics-out", str(metrics),
    ]
    if args.amp:
        cmd.append("--amp")

    env = os.environ.copy()
    k = args.gpus
    if k > 1:
        env.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(_free_port()),
                    "WORLD_SIZE": str(k)})
        if platform.system() == "Darwin":
            env["GLOO_SOCKET_IFNAME"] = "lo0"

    procs, errs = [], []
    for rank in range(k):
        e = env.copy()
        if k > 1:
            e.update({"RANK": str(rank), "LOCAL_RANK": str(rank)})
        procs.append(subprocess.Popen(cmd, env=e, stderr=subprocess.PIPE, text=True))
    codes = []
    for p in procs:
        _, err = p.communicate()
        errs.append(err or "")
        codes.append(p.returncode)

    status = "ok"
    for code, err in zip(codes, errs):
        s = classify(code, err)
        if s != "ok":
            status = s
            break

    out = {"width": width, "parallel": parallel, "status": status,
           "peak_mem_mb": None, "params": None}
    if status == "ok" and metrics.exists():
        m = json.loads(metrics.read_text())
        out["peak_mem_mb"] = m.get("peak_mem_mb")
        out["params"] = m.get("model_params")
    elif status == "error":
        tail = (errs[0] or "").strip().splitlines()[-1:] or [""]
        out["error"] = tail[0][:200]
    return out


def format_table(rows: list[dict]) -> str:
    lines = ["| Model width | Params | DDP (replicated) | FSDP (sharded) |",
             "|-------------|--------|------------------|----------------|"]
    by_width: dict[int, dict] = {}
    for r in rows:
        by_width.setdefault(r["width"], {})[r["parallel"]] = r

    def cell(r):
        if r is None:
            return "–"
        if r["status"] == "ok":
            return f"✅ {r['peak_mem_mb']:,.0f} MB" if r["peak_mem_mb"] else "✅ ran"
        if r["status"] == "oom":
            return "❌ **OOM**"
        return f"⚠️ error"

    for w in sorted(by_width):
        pair = by_width[w]
        params = next((p["params"] for p in pair.values() if p.get("params")), None)
        pstr = f"{params / 1e6:.1f}M" if params else "–"
        lines.append(f"| {w} | {pstr} | {cell(pair.get('ddp'))} | {cell(pair.get('fsdp'))} |")
    return "\n".join(lines)


def make_plot(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, colour in (("ddp", "tab:blue"), ("fsdp", "tab:orange")):
        ok = [r for r in rows if r["parallel"] == mode and r["status"] == "ok" and r["peak_mem_mb"]]
        if ok:
            ok.sort(key=lambda r: r["width"])
            ax.plot([r["width"] for r in ok], [r["peak_mem_mb"] for r in ok],
                    "o-", color=colour, label=mode.upper())
        for r in rows:
            if r["parallel"] == mode and r["status"] == "oom":
                ax.scatter([r["width"]], [ax.get_ylim()[1]], marker="X", s=140,
                           color=colour, zorder=5)
                ax.annotate(f"{mode.upper()} OOM", (r["width"], ax.get_ylim()[1]),
                            textcoords="offset points", xytext=(0, -18),
                            ha="center", fontsize=9, color=colour)
    ax.set_xlabel("model width (channels)")
    ax.set_ylabel("peak memory per GPU (MB)")
    ax.set_title("Where data parallelism runs out of memory and sharding doesn't")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[1]:  # no labelled lines when nothing recorded memory
        ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--widths", type=str, default="64,128,192,256",
                   help="model widths to sweep; grow until DDP OOMs")
    p.add_argument("--gpus", type=int, default=2, help="world size to run both strategies at")
    p.add_argument("--modes", type=int, default=24)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2, help="only need enough to allocate memory")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", type=Path, default=Path("crossover.md"))
    p.add_argument("--plot", type=Path, default=Path("docs/crossover.png"))
    args = p.parse_args()

    widths = [int(w) for w in args.widths.split(",")]
    print(f"sweeping widths {widths} at world_size={args.gpus}, modes={args.modes}\n")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for w in widths:
            for mode in ("ddp", "fsdp"):
                print(f"=== width {w} · {mode} ===", flush=True)
                r = run_case(w, mode, args, tmp)
                rows.append(r)
                if r["status"] == "ok":
                    mem = f"{r['peak_mem_mb']:,.0f} MB" if r["peak_mem_mb"] else "ran (no CUDA)"
                    print(f"  ok — peak {mem}")
                elif r["status"] == "oom":
                    print("  OUT OF MEMORY (expected for DDP past the crossover)")
                else:
                    print(f"  error: {r.get('error', '')}")

    table = format_table(rows)
    print("\n" + table)
    args.out.write_text(table + "\n")
    print(f"\nwrote {args.out}")

    crossed = [r["width"] for r in rows
               if r["parallel"] == "ddp" and r["status"] == "oom"
               and any(o["width"] == r["width"] and o["parallel"] == "fsdp"
                       and o["status"] == "ok" for o in rows)]
    if crossed:
        print(f"\nCROSSOVER FOUND at width {min(crossed)}: DDP cannot fit the model, "
              f"FSDP trains it. This is the headline result.")
    else:
        print("\nNo crossover yet — raise --widths (and/or --modes) until DDP OOMs.")
    try:
        make_plot(rows, args.plot)
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
