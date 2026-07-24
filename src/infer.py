"""Evaluate a trained FNO checkpoint and render prediction vs. ground truth.

    python -m src.infer --checkpoint checkpoints/fno_darcy.pt --data data/darcy_train.npz

Loads the checkpoint (which carries its own model config + normalisation stats),
reports relative L2 over the dataset, and saves a figure comparing the input
permeability, the true solution, the FNO prediction, and the absolute error for
a few samples -- the kind of qualitative panel a portfolio README wants.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.fno import FNO2d


def load_model(checkpoint: Path, device: torch.device) -> tuple[FNO2d, dict]:
    ckpt = torch.load(checkpoint, map_location=device)
    model = FNO2d(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["stats"]


@torch.no_grad()
def predict(model: FNO2d, a_raw: torch.Tensor, stats: dict, device: torch.device) -> torch.Tensor:
    """Normalise input, run the model, and denormalise back to physical units."""
    a = (a_raw - stats["a_mean"]) / stats["a_std"]
    u = model(a.to(device)) * stats["u_std"]
    return u.cpu()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--n-show", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("docs/prediction.png"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_model(args.checkpoint, device)

    raw = np.load(args.data)
    a = torch.from_numpy(raw["a"]).float()
    u = torch.from_numpy(raw["u"]).float()

    pred = predict(model, a, stats, device)
    rel = (torch.linalg.vector_norm(pred - u, dim=(-2, -1))
           / torch.linalg.vector_norm(u, dim=(-2, -1)))
    print(f"relative L2 over {len(u)} samples: "
          f"mean {rel.mean():.4f}  median {rel.median():.4f}  max {rel.max():.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(args.n_show, len(u))
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    axes = np.atleast_2d(axes)
    titles = ["permeability a(x)", "true u(x)", "FNO prediction", "abs error"]
    for i in range(n):
        err = (pred[i] - u[i]).abs()
        panels = [a[i], u[i], pred[i], err]
        cmaps = ["viridis", "magma", "magma", "inferno"]
        for j, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[i, j]
            im = ax.imshow(panel, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if i == 0:
                ax.set_title(titles[j])
            ax.set_xticks([]); ax.set_yticks([])
        axes[i, 0].set_ylabel(f"sample {i}\nrelL2 {rel[i]:.3f}")
    fig.suptitle("FNO Darcy-flow surrogate: prediction vs. ground truth", y=1.0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
