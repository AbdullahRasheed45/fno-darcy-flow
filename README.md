# physics-surrogate-ddp

A machine-learning surrogate for **Darcy flow**: a Fourier Neural Operator (FNO)
trained to map random permeability fields `a(x)` to pressure solutions `u(x)`
of the PDE `-div(a grad u) = f`, replacing a numerical solver with fast
neural inference. Training runs single-GPU or **multi-GPU with PyTorch
DistributedDataParallel**, with no code changes between modes.

Why this problem: field-to-field operator learning on PDE data is the core
workload behind AI-accelerated engineering simulation. The dataset here is
generated from scratch with a finite-volume solver (harmonic face averaging,
correct for discontinuous coefficients), so the whole pipeline — data
generation, model, distributed training, evaluation — is self-contained and
reproducible.

![prediction vs. ground truth](docs/prediction.png)

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data (~1200 solved PDE instances at 64x64)
python -m src.data.darcy --n-samples 1200 --grid 64 --out data/darcy_train.npz

# 2a. Train on a single GPU (or CPU)
python -m src.train --data data/darcy_train.npz --epochs 100 --amp

# 2b. Train on 2 GPUs with DDP
torchrun --standalone --nproc_per_node=2 -m src.train \
    --data data/darcy_train.npz --epochs 100 --amp

# 3. Evaluate + render prediction figure
python -m src.infer --checkpoint checkpoints/fno_darcy.pt --data data/darcy_train.npz
```

Metric: **relative L2 error** `||u_pred - u|| / ||u||`, the standard
neural-operator benchmark metric. A well-trained FNO at this resolution
should reach roughly 1–3% relative L2 on validation.

**Running on GPUs (Kaggle free 2× T4, DGX Spark, rented cloud):** see
[GPU_GUIDE.md](GPU_GUIDE.md) and the ready-to-run
[notebooks/kaggle_2xT4.ipynb](notebooks/kaggle_2xT4.ipynb).

## Scaling benchmark

The whole point of the DDP work is a measured scaling result. One command runs
the same training on 1 and 2 GPUs and emits the table:

```bash
python -m src.benchmark --data data/darcy_train.npz --gpus 1,2 --epochs 100 --amp
```

| Setup | GPUs | epoch time (s) | throughput (samples/s) | val relL2 | speed-up |
|-------|------|----------------|------------------------|-----------|----------|
| 1x GPU | 1 | _fill in_ | _fill in_ | _fill in_ | 1.00x |
| 2x GPU (DDP) | 2 | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

Run it on real GPUs and paste the generated table here. Near-linear throughput
scaling (~2×) with matched accuracy is the headline; if it's sub-linear,
profiling *where* the time goes (data loading vs. all-reduce vs. compute) is the
most interview-valuable part of the whole project — see the notes in
[GPU_GUIDE.md](GPU_GUIDE.md).

> The benchmark launches ranks via environment variables rather than `torchrun`.
> That path is identical to what torchrun does (same NCCL init), but it also
> runs on macOS/CPU for local testing, where `torchrun --standalone` can hang on
> IPv6 loopback rendezvous.

## Design decisions (and why)

The questions a reviewer will ask, answered:

- **Why harmonic averaging of permeability at cell faces?** The permeability is
  piecewise-constant (two-phase media), so it's *discontinuous* at phase
  boundaries. Flux continuity across a face between cells of permeability `a_i`
  and `a_j` is preserved by the **harmonic** mean `2 a_i a_j / (a_i + a_j)`, not
  the arithmetic mean — the arithmetic mean lets a high-permeability cell
  dominate a low one and produces the wrong flux (and wrong solution) across
  interfaces. This is the standard finite-volume treatment for discontinuous
  coefficients.

- **Why relative L2, not MSE?** The solution magnitude varies sample-to-sample
  with the permeability field. Normalising the error by `||u||` makes it
  scale-invariant, comparable across samples and across grid resolutions, and it
  is the metric the FNO literature (Li et al., 2021) reports — so numbers are
  directly comparable to published results.

- **Why `sampler.set_epoch(epoch)`?** `DistributedSampler` shards the dataset
  across ranks and shuffles deterministically from a seed. Without
  `set_epoch`, every epoch uses the *same* shuffle, so each rank sees the same
  order forever — you lose the regularisation benefit of shuffling. Calling it
  per epoch reseeds the shuffle identically on all ranks (so shards stay
  disjoint) while varying it across epochs.

- **How does DDP's all-reduce work?** Each rank holds a full model replica and
  computes gradients on its own data shard. During `backward()`, DDP
  **all-reduces** (averages) the gradients across ranks — overlapping the
  communication with backprop by bucketing gradients — so every replica ends the
  step with identical averaged gradients and the optimizer keeps them in sync.
  Effective batch size = per-rank batch × world size. In this repo the *metric*
  is also all-reduced (`ReduceOp.AVG`) so the logged val relL2 is the true global
  average, and the sample count is all-reduced (`SUM`) for correct global
  throughput.

- **Why spectral convolutions / why is the FNO resolution-invariant?** An FNO
  layer multiplies the lowest Fourier modes of the input by learned complex
  weights (a global convolution), then inverse-transforms. Because it
  parameterises the operator in Fourier space over a fixed number of *modes*
  (not a fixed pixel grid), the same trained weights apply at any resolution —
  verified by a test that runs the model at 32×32 and 48×48 with one set of
  weights.

- **Why AMP / cosine LR / AdamW?** Mixed precision (`--amp`, CUDA only) roughly
  halves memory and speeds up the FFT-heavy forward pass on tensor-core GPUs;
  cosine annealing gives a smooth LR decay that reaches lower final error than a
  constant LR; AdamW's decoupled weight decay is the standard regulariser for
  this model family.

## Tests

```bash
python -m pytest -q
```

- **Solver** (`test_darcy.py`): positivity, symmetry, and linearity of the
  discretisation (`u` scales inversely with permeability).
- **Model** (`test_fno.py`): spectral-conv shapes, gradient flow, and
  resolution invariance.
- **Training** (`test_train.py`): an end-to-end single-process run of the loop,
  checking it converges to a finite metric and writes a valid checkpoint.

CI runs the suite on every push.

## Layout

```
src/data/darcy.py    # GRF sampling + finite-volume Darcy solver + dataset generation
src/models/fno.py    # SpectralConv2d / FNO2d
src/train.py         # DDP-aware training loop, AMP, cosine LR, throughput + metrics JSON
src/benchmark.py     # runs 1 vs N GPUs, emits the scaling table
src/infer.py         # eval a checkpoint + render prediction/ground-truth/error figure
tests/               # solver + model + training tests
notebooks/           # ready-to-run Kaggle 2x T4 benchmark notebook
Dockerfile           # CUDA runtime image for cloud GPU boxes
GPU_GUIDE.md         # how to run on Kaggle / DGX Spark / rented cloud
```
