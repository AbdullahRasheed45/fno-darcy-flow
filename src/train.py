"""Train an FNO surrogate for Darcy flow, single-GPU or distributed.

Single GPU / CPU:
    python -m src.train --data data/darcy_train.npz

Multi-GPU (DistributedDataParallel), e.g. 2 GPUs on one node:
    torchrun --standalone --nproc_per_node=2 -m src.train --data data/darcy_train.npz

The script detects a distributed launch via environment variables (RANK /
WORLD_SIZE), so it runs identically under `torchrun` or under the manual
process-spawning launcher in `src/benchmark.py` -- no code changes between
single- and multi-GPU modes. Metrics reported: relative L2 error (the
standard neural-operator benchmark metric) plus throughput (samples/s).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler

from src.models.fno import FNO2d


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean relative L2 error over the batch: ||pred - u|| / ||u||.

    Relative (not absolute) L2 is the neural-operator standard because the
    solution magnitude varies sample to sample with the permeability field;
    normalising by ||u|| makes the metric scale-invariant and comparable
    across samples and resolutions.
    """
    diff = torch.linalg.vector_norm(pred - target, dim=(-2, -1))
    denom = torch.linalg.vector_norm(target, dim=(-2, -1))
    return (diff / denom).mean()


def setup_distributed() -> tuple[int, int, torch.device]:
    """Return (rank, world_size, device); initialises the process group if launched
    distributed. Works under torchrun and under a bare env-var launch (RANK /
    WORLD_SIZE / MASTER_ADDR / MASTER_PORT set by the caller)."""
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world = dist.get_world_size()
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return rank, world, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def load_data(path: Path) -> tuple[TensorDataset, TensorDataset, dict]:
    raw = np.load(path)
    a = torch.from_numpy(raw["a"]).float()
    u = torch.from_numpy(raw["u"]).float()
    # Normalise: permeability to zero mean / unit std, solution scaled by its std.
    stats = {
        "a_mean": float(a.mean()),
        "a_std": float(a.std()),
        "u_std": float(u.std()),
    }
    a = (a - stats["a_mean"]) / stats["a_std"]
    u = u / stats["u_std"]
    ds = TensorDataset(a, u)
    n_val = max(1, int(0.1 * len(ds)))
    train, val = random_split(ds, [len(ds) - n_val, n_val],
                              generator=torch.Generator().manual_seed(0))
    return train, val, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16, help="per-process batch size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="mixed-precision training")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("checkpoints/fno_darcy.pt"))
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="if set, write a JSON summary (used by the scaling benchmark)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rank, world, device = setup_distributed()
    is_main = rank == 0

    train_ds, val_ds, stats = load_data(args.data)
    train_sampler = DistributedSampler(train_ds) if world > 1 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=(train_sampler is None), sampler=train_sampler,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = FNO2d(modes=args.modes, width=args.width, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if world > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    if is_main:
        print(f"model: FNO2d(modes={args.modes}, width={args.width}, layers={args.layers})  "
              f"params={n_params:,}  world={world}  device={device.type}  "
              f"global_batch={args.batch_size * world}")

    epoch_times: list[float] = []
    last_train, last_val = float("nan"), float("nan")
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # reshuffle differently each epoch across ranks
        model.train()
        t0, train_loss, n_batches, local_samples = time.time(), 0.0, 0, 0
        for a, u in train_dl:
            a, u = a.to(device, non_blocking=True), u.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                loss = relative_l2(model(a), u)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()
            n_batches += 1
            local_samples += a.shape[0]
        sched.step()
        epoch_dt = time.time() - t0
        epoch_times.append(epoch_dt)

        # Global samples/s: sum the samples every rank processed, divide by wall time.
        samples = torch.tensor(float(local_samples))
        if world > 1:
            dist.all_reduce(samples, op=dist.ReduceOp.SUM)
        throughput = samples.item() / epoch_dt

        model.eval()
        with torch.no_grad():
            val_err = torch.stack(
                [relative_l2(model(a.to(device)), u.to(device)) for a, u in val_dl]
            ).mean()
        if world > 1:
            dist.all_reduce(val_err, op=dist.ReduceOp.AVG)

        last_train, last_val = train_loss / max(n_batches, 1), val_err.item()
        if is_main:
            print(f"epoch {epoch + 1:3d}/{args.epochs}  "
                  f"train relL2 {last_train:.4f}  "
                  f"val relL2 {last_val:.4f}  "
                  f"{epoch_dt:.2f}s  {throughput:,.0f} samples/s  world={world}")

    # Mean epoch time excludes epoch 0 (warm-up: allocator, cuDNN autotune, DDP buckets).
    mean_epoch = float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else epoch_times[0]
    total_samples = len(train_ds)
    throughput_mean = total_samples / mean_epoch

    if is_main:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        state = model.module.state_dict() if world > 1 else model.state_dict()
        torch.save({
            "model": state,
            "config": {"modes": args.modes, "width": args.width, "layers": args.layers},
            "stats": stats,
        }, args.out)
        print(f"saved {args.out}")
        print(f"mean epoch time (excl. warm-up): {mean_epoch:.3f}s  "
              f"({throughput_mean:,.0f} samples/s global)")

        if args.metrics_out is not None:
            metrics = {
                "world_size": world,
                "device": device.type,
                "epochs": args.epochs,
                "per_rank_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * world,
                "model_params": n_params,
                "mean_epoch_time_s": mean_epoch,
                "throughput_samples_per_s": throughput_mean,
                "final_train_relL2": last_train,
                "final_val_relL2": last_val,
            }
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(json.dumps(metrics, indent=2))

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
