"""Tests of jernn.  The numpy parts always run; the network parts need torch (skipped without)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jernn import data as jd                                    # noqa: E402
from jernn.diffusion import (density_coefficients, forward_sample, hermite, linearisation,  # noqa: E402
                             noised_pdf, noised_score)
from jernn.features import Cut, Feature                         # noqa: E402

RNG = np.random.default_rng(7)
C = np.array([1.0, 0.3 - 0.2j, -0.15 + 0.1j, 0.05j, 0.02])      # an asymmetric amplitude, J = 4


# ------------------------------------------------------------------ diffusion (closed form)
def test_linearisation_matches_quadrature():
    J = 4
    z, wq = np.polynomial.hermite_e.hermegauss(40)
    wq = wq / np.sqrt(2 * np.pi)
    P = hermite(z, 2 * J)
    ref = np.einsum("q,qm,qn,qk->mnk", wq, P[:, :J + 1], P[:, :J + 1], P)
    assert np.allclose(linearisation(J), ref, atol=1e-10)


def test_density_coefficients_reproduce_born_density():
    z = np.linspace(-6, 6, 301)
    h = hermite(z, len(C) - 1) @ C
    born = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) * np.abs(h) ** 2 / np.sum(np.abs(C) ** 2)
    assert np.allclose(noised_pdf(z, 0.0, 1.0, C, 0.0), born, atol=1e-12)
    assert np.isclose(density_coefficients(C[None])[0, 0], 1.0)


@pytest.mark.parametrize("t", [0.0, 0.1, 0.7, 3.0])
def test_noised_density_is_normalised_and_positive(t):
    y = np.linspace(-1.0, 3.0, 4001)
    p = noised_pdf(y, 1.0, 0.1, C, t)
    assert np.all(p > -1e-12)
    assert np.isclose(np.trapezoid(p, y), 1.0, atol=1e-6)


def test_noised_density_matches_forward_samples():
    mu, sig, t = 1.0, 0.1, 0.4
    # draw y0 from the Born density by inverse CDF, then noise forward exactly
    yg = np.linspace(mu - 10 * sig, mu + 10 * sig, 20001)
    F = np.cumsum(noised_pdf(yg, mu, sig, C, 0.0)); F /= F[-1]
    y0 = np.interp(RNG.random(400000), F, yg)
    yt = forward_sample(y0, mu, sig, t, RNG)
    h, e = np.histogram(yt, 60, range=(mu - 4 * sig, mu + 4 * sig))
    pt = noised_pdf(0.5 * (e[1:] + e[:-1]), mu, sig, C, t) * np.diff(e) * len(yt)
    pull = (h - pt) / np.sqrt(np.maximum(pt, 1))
    assert np.mean(pull ** 2) < 2.0


def test_noised_density_tends_to_reference_gaussian():
    y = np.linspace(0.5, 1.5, 11)
    g = np.exp(-0.5 * ((y - 1) / 0.1) ** 2) / (0.1 * np.sqrt(2 * np.pi))
    assert np.allclose(noised_pdf(y, 1.0, 0.1, C, 30.0), g, rtol=1e-8)


def test_mean_relaxes_exponentially():
    y = np.linspace(-1, 3, 8001)
    m0 = np.trapezoid(y * noised_pdf(y, 1.0, 0.1, C, 0.0), y)
    for t in (0.2, 1.0):
        mt = np.trapezoid(y * noised_pdf(y, 1.0, 0.1, C, t), y)
        assert np.isclose(mt - 1.0, np.exp(-t) * (m0 - 1.0), atol=1e-7)


def test_score_is_derivative_of_log_density():
    y = np.linspace(0.8, 1.2, 41)
    h = 1e-6
    for t in (0.0, 0.3):
        num = (np.log(noised_pdf(y + h, 1, 0.1, C, t)) - np.log(noised_pdf(y - h, 1, 0.1, C, t))) / (2 * h)
        assert np.allclose(noised_score(y, 1, 0.1, C, t), num, rtol=1e-5, atol=1e-3)


def test_rows_shapes():
    c = np.stack([C, C * 1j, C[::-1]])
    p_rows = noised_pdf(np.array([1.0, 1.1, 0.9]), np.ones(3), np.full(3, 0.1), c, 0.2)
    p_grid = noised_pdf(np.linspace(0.5, 1.5, 7)[None, :], np.ones((3, 1)), np.full((3, 1), 0.1), c, 0.2)
    assert p_rows.shape == (3,) and p_grid.shape == (3, 7)
    assert np.isclose(p_grid[0, 3], noised_pdf(1.0, 1.0, 0.1, C, 0.2))


# ------------------------------------------------------------------ features, data
def _write(tmp_path, name, weights=True, n=2000):
    pt = np.exp(RNG.uniform(np.log(30), np.log(3000), n))
    cols = dict(genpt=pt.astype(np.float32), recpt=(pt * RNG.normal(1, 0.1, n)).astype(np.float32),
                receta=RNG.uniform(-4, 4, n).astype(np.float32), dR=RNG.uniform(0, 0.6, n).astype(np.float32),
                rho=RNG.uniform(0, 40, n).astype(np.float32))
    if weights:
        cols["weight"] = RNG.lognormal(0, 1, n)
    p = tmp_path / name
    np.savez(p, **cols)
    return str(p)


def test_features_and_cuts(tmp_path):
    f = _write(tmp_path, "a.npz")
    D = jd.read([f], Feature("response"), [Feature("genpt:log"), Feature("receta:abs"), Feature("rho")],
                [Cut("dr<0.4"), Cut("genpt>=97"), Cut("abs(receta)<2.5")], verbose=False)
    with np.load(f) as z:
        m = (z["dR"] < 0.4) & (z["genpt"] >= 97) & (np.abs(z["receta"]) < 2.5)
        assert len(D["y"]) == m.sum()
        assert np.allclose(D["y"], (z["recpt"] / z["genpt"])[m], rtol=1e-6)
        assert np.allclose(D["X"][:, 0], np.log(z["genpt"][m]), rtol=1e-6)
        assert np.allclose(D["X"][:, 1], np.abs(z["receta"][m]))
        assert np.allclose(D["w"], z["weight"][m])
    assert D["meta"]["weight"] == "weight"
    with pytest.raises(KeyError):
        jd.read([f], Feature("response"), [Feature("nVtx")], verbose=False)


def test_unit_weights_without_weight_column(tmp_path):
    f = _write(tmp_path, "data.npz", weights=False)
    D = jd.read([f], Feature("response"), [Feature("genpt:log")], weight="weight", verbose=False)
    assert np.all(D["w"] == 1) and D["meta"]["weight"] == "unit"
    D2 = jd.read([_write(tmp_path, "b.npz")], Feature("response"), [Feature("genpt:log")], weight="none", verbose=False)
    assert np.all(D2["w"] == 1)


def test_mixing_weighted_and_unweighted_is_refused(tmp_path):
    with pytest.raises(ValueError):
        jd.read([_write(tmp_path, "w.npz"), _write(tmp_path, "u.npz", weights=False)], Feature("response"),
                [Feature("genpt:log")], verbose=False)


def test_inputs_unchanged(tmp_path):
    f = _write(tmp_path, "c.npz")
    before = open(f, "rb").read()
    jd.read([f], Feature("response"), [Feature("genpt:log")], fraction=0.5, verbose=False)
    assert open(f, "rb").read() == before


def test_balance():
    x = RNG.uniform(0, 1, 50000)
    w = np.where(x < 0.5, 10.0, 1.0) * RNG.lognormal(0, 0.3, len(x))
    ref = np.arange(len(x))
    g = jd.balance(x, w, ref, nbins=10, power=0.0)              # equal totals per (quantile) bin
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    tot = np.bincount(b, g * w)
    assert np.allclose(tot / tot.mean(), 1, atol=1e-6)
    assert np.isclose((g * w).sum(), w.sum())
    g1 = jd.balance(x, np.ones_like(x), ref, nbins=10, power=1.0)  # unit weights, alpha 1: no change
    assert np.allclose(g1, 1)


# ------------------------------------------------------------------ network (torch)
def test_network_density_normalised_and_censoring():
    torch = pytest.importorskip("torch")
    from jernn.model import BornNet
    torch.manual_seed(0)
    m = BornNet(2, J=4, width=16, depth=2, y_loc=1.0, y_scale=0.1, y_range=(0.7, 1.3))
    with torch.no_grad():
        m.head.weight.normal_(0, 0.3)                           # a non-trivial shape
    X = torch.tensor([[5.0, 0.3], [7.0, 2.0]])
    yg = torch.linspace(0.0, 2.0, 20001)
    for i in range(2):
        xi = X[i:i + 1].repeat(len(yg), 1)
        mu, sig, cre, cim = m(xi)
        lnorm = torch.log((cre ** 2 + cim ** 2).sum(-1))
        z = (yg - mu) / sig
        lp = m._log_shape(z, cre, cim, lnorm) - torch.log(sig)
        p = torch.exp(lp).double()
        inside = (yg >= 0.7) & (yg < 1.3)
        p_in = torch.trapezoid(p[inside], yg[inside].double())
        lpc, _ = m.log_prob(torch.tensor([0.1, 1.9]), X[i:i + 1].repeat(2, 1))
        total = p_in + torch.exp(lpc.double()).sum()
        assert abs(float(total) - 1.0) < 2e-3
