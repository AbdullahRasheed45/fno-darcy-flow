import numpy as np
from src.data.darcy import gaussian_random_field, permeability_from_grf, solve_darcy


def test_solver_positive_and_symmetric():
    """Constant permeability with symmetric forcing gives a positive,
    symmetric solution peaked at the domain centre."""
    n = 32
    u = solve_darcy(np.full((n, n), 5.0))
    assert (u > 0).all()
    assert np.allclose(u, u.T, atol=1e-10)
    assert u[n // 2, n // 2] == u.max() or u[n // 2 - 1, n // 2 - 1] == u.max()


def test_solver_scales_inversely_with_permeability():
    n = 32
    u1 = solve_darcy(np.full((n, n), 1.0))
    u2 = solve_darcy(np.full((n, n), 2.0))
    assert np.allclose(u1, 2.0 * u2, rtol=1e-8)


def test_grf_and_threshold():
    grf = gaussian_random_field(64, rng=np.random.default_rng(0))
    a = permeability_from_grf(grf)
    assert set(np.unique(a)) == {3.0, 12.0}
