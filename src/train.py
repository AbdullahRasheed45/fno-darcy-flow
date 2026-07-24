"""Train an FNO surrogate for Darcy flow: single-GPU, DDP, or FSDP.

Single GPU / CPU:
    python -m src.train --data data/darcy_train.npz

Data-parallel (DDP) across 2 GPUs on one node:
    torchrun --standalone --nproc_per_node=2 -m src.train --data data/darcy_train.npz

Sharded (FSDP / ZeRO-3) across 2 GPUs -- shards params, grads, and optimizer
state so a model too large for one GPU still fits:
    torchrun --standalone --nproc_per_node=2 -m src.train \
        --data data/darcy_train.npz --parallel fsdp

The script detects a distributed launch via environment variables (RANK /
WORLD_SIZE), so it runs identically under `torchrun` or under the manual
process-spawning launcher in `src/benchmark.py`. Reports relative-L2 error,
throughput (samples/s), and peak GPU memory per rank.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler

from src.models.fno import FNO2d, FNOBlock


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


def wrap_parallel(model: torch.nn.Module, parallel: str, world: int, device: torch.device):
    """Wrap the model for the chosen parallelism. Returns the (possibly wrapped) model.

    - ddp:  full replica per rank, gradients all-reduced each step (data parallel).
    - fsdp: params/grads/optimizer-state sharded across ranks (ZeRO-3), all-gathered
            layer-by-layer during forward/backward -- trades communication for memory,
            so a model too big to replicate still fits.
    """
    if world == 1:
        return model  # nothing to shard/replicate on a single process
    if parallel == "ddp":
        return DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    if parallel == "fsdp":
        if device.type != "cuda":
            raise RuntimeError(
                "--parallel fsdp requires CUDA GPUs: FSDP shards parameters onto an "
                "accelerator and torch refuses to initialise it on CPU. Use --parallel ddp "
                "for CPU/single-GPU runs."
            )
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        # Wrap each sizeable submodule (e.g. every FNOBlock) as its own FSDP unit so
        # only one block's full parameters are materialised at a time.
        policy = functools.partial(size_based_auto_wrap_policy, min_num_params=20_000)
        kwargs = {"auto_wrap_policy": policy}
        if device.type == "cuda":
            kwargs["device_id"] = device.index
        return FSDP(model, **kwargs)
    raise ValueError(f"unknown --parallel {parallel!r}")


def make_scaler(parallel: str, amp: bool, device: torch.device):
    """AMP gradient scaler. FSDP needs the sharded variant (grads live in shards)."""
    enabled = amp and device.type == "cuda"
    if parallel == "fsdp" and enabled:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        return ShardedGradScaler(enabled=True)
    return torch.amp.GradScaler(enabled=enabled)


def gather_state_dict(model, parallel: str, world: int) -> dict:
    """Collective: assemble a full (unsharded) state dict on rank 0. All ranks must call.

    Under FSDP the parameters live in shards, so saving on rank 0 alone would write
    a fragment; every rank has to participate in the gather.
    """
    if parallel == "fsdp" and world > 1:
        try:
            # Current API (works across FSDP1/FSDP2/DDP).
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions, get_model_state_dict)
            return get_model_state_dict(
                model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        except Exception:
            # Fall back to the legacy context manager on older torch.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                return model.state_dict()
    if world > 1:
        return model.module.state_dict()
    return model.state_dict()


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
    p.add_argument("--batch-size", type=int, default=16, help="per-process micro-batch size")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="micro-batches per optimizer step (raises effective batch without memory)")
    p.add_argument("--parallel", choices=["ddp", "fsdp"], default="ddp",
                   help="multi-GPU strategy: ddp (replicate) or fsdp (shard/ZeRO-3)")
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
    accum = max(1, args.grad_accum)

    train_ds, val_ds, stats = load_data(args.data)
    train_sampler = DistributedSampler(train_ds) if world > 1 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=(train_sampler is None), sampler=train_sampler,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = FNO2d(modes=args.modes, width=args.width, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model = wrap_parallel(model, args.parallel, world, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = make_scaler(args.parallel, args.amp, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    effective_batch = args.batch_size * world * accum
    if is_main:
        print(f"model: FNO2d(modes={args.modes}, width={args.width}, layers={args.layers})  "
              f"params={n_params:,}  world={world}  parallel={args.parallel}  "
              f"device={device.type}  micro_batch={args.batch_size}  "
              f"grad_accum={accum}  effective_batch={effective_batch}")

    epoch_times: list[float] = []
    last_train, last_val = float("nan"), float("nan")
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # reshuffle differently each epoch across ranks
        model.train()
        t0, train_loss, n_batches, local_samples = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        n_steps = len(train_dl)
        for i, (a, u) in enumerate(train_dl):
            a, u = a.to(device, non_blocking=True), u.to(device, non_blocking=True)
            step_now = ((i + 1) % accum == 0) or (i + 1 == n_steps)
            # During accumulation, skip the DDP/FSDP gradient sync on non-step
            # micro-batches (no_sync) so we all-reduce once per optimizer step, not per micro-batch.
            sync_ctx = (model.no_sync() if (world > 1 and not step_now and hasattr(model, "no_sync"))
                        else nullcontext())
            with sync_ctx:
                with torch.autocast(device.type, enabled=scaler.is_enabled()):
                    raw = relative_l2(model(a), u)
                scaler.scale(raw / accum).backward()
            if step_now:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            train_loss += raw.item()
            n_batches += 1
            local_samples += a.shape[0]
        sched.step()
        epoch_dt = time.time() - t0
        epoch_times.append(epoch_dt)

        # Global samples/s: sum the samples every rank processed, divide by wall time.
        # The tensor must live on the compute device: NCCL collectives only operate
        # on CUDA tensors (a CPU tensor raises "No backend type associated with cpu").
        samples = torch.tensor(float(local_samples), device=device)
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
                  f"{epoch_dt:.2f}s  {throughput:,.0f} samples/s  "
                  f"{args.parallel}/world={world}")

    # Mean epoch time excludes epoch 0 (warm-up: allocator, cuDNN autotune, comm buckets).
    mean_epoch = float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else epoch_times[0]
    throughput_mean = len(train_ds) / mean_epoch

    # Peak memory: the headline FSDP metric. Report the worst rank (all-reduce MAX).
    if device.type == "cuda":
        peak = torch.tensor(float(torch.cuda.max_memory_allocated(device)), device=device)
        if world > 1:
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        peak_mem_mb = peak.item() / 1e6
    else:
        peak_mem_mb = None

    state = gather_state_dict(model, args.parallel, world)  # collective: all ranks call

    if is_main:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": state,
            "config": {"modes": args.modes, "width": args.width, "layers": args.layers},
            "stats": stats,
        }, args.out)
        print(f"saved {args.out}")
        mem_str = f"  peak mem/rank {peak_mem_mb:,.0f} MB" if peak_mem_mb else ""
        print(f"mean epoch time (excl. warm-up): {mean_epoch:.3f}s  "
              f"({throughput_mean:,.0f} samples/s global){mem_str}")

        if args.metrics_out is not None:
            metrics = {
                "world_size": world,
                "parallel": args.parallel,
                "device": device.type,
                "epochs": args.epochs,
                "per_rank_batch_size": args.batch_size,
                "grad_accum": accum,
                "effective_batch_size": effective_batch,
                "model_params": n_params,
                "mean_epoch_time_s": mean_epoch,
                "throughput_samples_per_s": throughput_mean,
                "peak_mem_mb": peak_mem_mb,
                "final_train_relL2": last_train,
                "final_val_relL2": last_val,
            }
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(json.dumps(metrics, indent=2))

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
