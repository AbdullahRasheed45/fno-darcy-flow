# Running on GPUs — migration guide

Three tiers, in the order you should do them. The whole point is the **scaling
table**: same training on 1 GPU vs 2 GPUs, real epoch-time and throughput
numbers. Everything below produces that.

---

## 0. Push to GitHub first (5 min)

```bash
cd physics-surrogate-ddp
git init && git add -A && git commit -m "FNO Darcy-flow surrogate with DDP"
gh repo create physics-surrogate-ddp --public --source=. --push
# or create the repo in the GitHub UI and: git remote add origin <url> && git push -u origin main
```

CI runs the test suite on that first push. A `data/` dir of generated `.npz`
and `checkpoints/` are big — add a `.gitignore` for them (a starter one is in
the repo). Commit `scaling.md` and `docs/prediction.png` once you have them —
those are the artifacts a reviewer looks at.

---

## 1. DGX Spark — single-GPU baseline (do this first, it's free)

The Spark is one GB10 (Grace-Blackwell) superchip = **one GPU** with unified
memory. That makes it the ideal box for the **single-GPU baseline number** and
for verifying the whole pipeline end-to-end at full 64×64 resolution. (It has
one GPU, so it gives you the `1×` row, not the scaling comparison — that's
tier 2.)

```bash
# The Spark ships with NVIDIA's PyTorch. Easiest path is the NGC container:
#   docker run --gpus all -it -v $PWD:/work nvcr.io/nvidia/pytorch:24.10-py3 bash
# or use the preinstalled conda/torch. Then, inside:
cd /work   # your repo
pip install scipy matplotlib        # torch is already there; don't reinstall it

python -m src.data.darcy --n-samples 1200 --grid 64 --out data/darcy_train.npz
python -m src.train --data data/darcy_train.npz --epochs 150 --amp
python -m src.infer --checkpoint checkpoints/fno_darcy.pt --data data/darcy_train.npz
```

Record the `mean epoch time` and `samples/s` line — that's your 1× baseline.
The Spark's unified memory means you can push `--grid 128 --width 64` far past
what a 16 GB T4 allows, which is itself a talking point.

> ARM64 note: the Spark is aarch64. Use the NGC container or NVIDIA's aarch64
> CUDA wheels — **not** `pip install torch` from plain PyPI, and never the
> `whl/cpu` index. If `torch.cuda.is_available()` is `False`, that's the cause.

---

## 2. Kaggle — the 2× GPU scaling run (free, this is the money shot)

Kaggle gives **2× T4 GPUs free** (30 GPU-hours/week), on Linux, so `torchrun`
and NCCL just work. This is the only free option that gives you *two* GPUs,
which is what makes "distributed training" a measured fact rather than code.

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. Upload `notebooks/kaggle_2xT4.ipynb` (File → Import Notebook), or paste the
   cells from it.
3. Right panel → **Accelerator: GPU T4 ×2**, **Internet: On**.
4. Edit cell 1: set `REPO_URL` to your GitHub repo.
5. **Run All.** It clones, installs, generates data, and runs
   `python -m src.benchmark --gpus 1,2 --amp`, which prints the scaling table.
6. **Save Version** to persist `scaling.md`, the checkpoint, and the figure to
   the notebook Output tab; download and commit them.

Manual equivalent, if you'd rather run cells yourself:

```bash
python -m src.data.darcy --n-samples 1200 --grid 64 --out data/darcy_train.npz

# throughput scaling (table + plot)
python -m src.benchmark --data data/darcy_train.npz --gpus 1,2 --epochs 100 --amp

# DDP vs FSDP peak-memory comparison at the same world size
python -m src.benchmark --data data/darcy_train.npz --gpus 2 --parallel ddp,fsdp \
    --epochs 30 --size large --amp
```

**Reading the FSDP result:** expect FSDP to be *slower* than DDP here and to use
*less* peak memory per GPU. That is the correct outcome, not a failure — FSDP
trades communication (all-gather + reduce-scatter per layer) for memory. Its
value shows up when the model no longer fits under DDP at all: raise `--size
large --width 128` until DDP OOMs and FSDP still runs. That crossover is the
most convincing thing you can demonstrate about sharded training.

---

## 3. Rented cloud (Lambda / RunPod / Vast) — optional, bigger GPUs

Same commands as Kaggle; you just get faster/more GPUs (e.g. 2× or 4× A10/A100).

```bash
git clone https://github.com/AbdullahRasheed45/fno-darcy-flow.git && cd fno-darcy-flow
pip install -r requirements.txt
python -m src.data.darcy --n-samples 1200 --grid 64 --out data/darcy_train.npz
# a 4-GPU box turns the two-point comparison into a real scaling *curve*
python -m src.benchmark --data data/darcy_train.npz --gpus 1,2,4 --size large --epochs 100 --amp
```

A 1/2/4-GPU curve (with the ideal-linear reference line the benchmark plots) is
substantially more convincing than a single 1-vs-2 number — it shows *where*
scaling starts to fall off, which is the interesting engineering content.

Or the Docker path (the repo's `Dockerfile` is CUDA-ready):

```bash
docker build -t fno .
docker run --gpus all -v $PWD/data:/workspace/data fno \
    python -m src.benchmark --data data/darcy_train.npz --gpus 1,2 --epochs 100 --amp
```

---

## Reading the scaling result

- **Throughput speed-up** near `2.0×` on 2 GPUs = near-linear scaling. That's
  the headline.
- **Sub-linear** (e.g. `1.6×`)? That's the *interesting* result to analyse:
  profile whether the time is in data loading, the NCCL all-reduce, or compute.
  `torch.profiler` or even `nvidia-smi dmon` during a run tells you. This
  investigation is the most interview-valuable part — don't paper over it.
- **Val relL2 should match** across 1 vs 2 GPUs at the *same total epochs*.
  DDP with a doubled global batch may need a slightly higher LR (linear scaling
  rule) or a few more epochs to hit the same accuracy — worth noting if you see
  a gap. A well-trained model at 64×64 lands around 1–3% relative L2.
