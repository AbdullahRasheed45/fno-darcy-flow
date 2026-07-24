"""Darcy flow dataset generation.

Solves the 2D Darcy flow equation on the unit square with a
finite-difference discretisation:

    -div( a(x) * grad(u(x)) ) = f(x),   x in (0,1)^2
    u(x) = 0 on the boundary

where a(x) is a random permeability field sampled from a Gaussian
random field pushed through a threshold (piecewise-constant media),
and f(x) = 1. The learning task is the operator a(x) -> u(x),
the standard benchmark from Li et al., "Fourier Neural Operator for
Parametric Partial Differential Equations" (ICLR 2021).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def gaussian_random_field(n: int, alpha: float = 2.0, tau: float = 3.0,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a mean-zero Gaussian random field on an n x n grid.

    Spectral density ~ tau^(alpha-1) * (pi^2 (k1^2 + k2^2) + tau^2)^(-alpha/2),
    i.e. draws from N(0, (-Laplacian + tau^2 I)^(-alpha)).
    """
    rng = rng or np.random.default_rng()
    k = np.arange(n)
    k1, k2 = np.meshgrid(k, k, indexing="ij")
    coef = (np.pi ** 2) * (k1 ** 2 + k2 ** 2) + tau ** 2
    sqrt_eig = (tau ** (alpha - 1.0)) * coef ** (-alpha / 2.0)
    sqrt_eig[0, 0] = 0.0  # remove the mean mode
    xi = rng.standard_normal((n, n))
    # Inverse discrete cosine-like transform via FFT of symmetrised field
    from scipy.fft import idctn
    return idctn(sqrt_eig * xi, norm="ortho")


def permeability_from_grf(grf: np.ndarray, hi: float = 12.0, lo: float = 3.0) -> np.ndarray:
    """Threshold a GRF into a two-phase permeability field (as in the FNO paper)."""
    return np.where(grf >= 0.0, hi, lo)


def solve_darcy(a: np.ndarray, f: float = 1.0) -> np.ndarray:
    """Solve -div(a grad u) = f with homogeneous Dirichlet BCs.

    Uses a 5-point finite-volume stencil with harmonic averaging of the
    permeability at cell faces, which is the standard discretisation for
    discontinuous coefficients.
    """
    n = a.shape[0]
    h = 1.0 / (n + 1)

    def harmonic(x, y):
        return 2.0 * x * y / (x + y)

    # Face permeabilities (interior cells only)
    a_e = np.zeros((n, n)); a_w = np.zeros((n, n))
    a_n = np.zeros((n, n)); a_s = np.zeros((n, n))
    a_e[:, :-1] = harmonic(a[:, :-1], a[:, 1:]); a_e[:, -1] = a[:, -1]
    a_w[:, 1:] = harmonic(a[:, 1:], a[:, :-1]); a_w[:, 0] = a[:, 0]
    a_n[:-1, :] = harmonic(a[:-1, :], a[1:, :]); a_n[-1, :] = a[-1, :]
    a_s[1:, :] = harmonic(a[1:, :], a[:-1, :]); a_s[0, :] = a[0, :]

    diag = (a_e + a_w + a_n + a_s).ravel()
    idx = np.arange(n * n).reshape(n, n)

    rows, cols, vals = [np.array([], dtype=np.int64)] * 0, [], []
    rows = [idx.ravel()]; cols = [idx.ravel()]; vals = [diag]

    def add(r, c, v):
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(-v.ravel())

    add(idx[:, :-1], idx[:, 1:], a_e[:, :-1])   # east neighbour
    add(idx[:, 1:], idx[:, :-1], a_w[:, 1:])    # west neighbour
    add(idx[:-1, :], idx[1:, :], a_n[:-1, :])   # north neighbour
    add(idx[1:, :], idx[:-1, :], a_s[1:, :])    # south neighbour

    A = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * n, n * n),
    )
    b = np.full(n * n, f * h * h)
    u = spsolve(A, b)
    return u.reshape(n, n)


def generate(n_samples: int, grid: int, seed: int, out: Path) -> None:
    rng = np.random.default_rng(seed)
    a_all = np.empty((n_samples, grid, grid), dtype=np.float32)
    u_all = np.empty((n_samples, grid, grid), dtype=np.float32)
    for i in range(n_samples):
        a = permeability_from_grf(gaussian_random_field(grid, rng=rng))
        a_all[i] = a
        u_all[i] = solve_darcy(a)
        if (i + 1) % 50 == 0:
            print(f"  solved {i + 1}/{n_samples}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, a=a_all, u=u_all)
    print(f"wrote {out} ({n_samples} samples at {grid}x{grid})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=1200)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/darcy_train.npz"))
    args = p.parse_args()
    generate(args.n_samples, args.grid, args.seed, args.out)
