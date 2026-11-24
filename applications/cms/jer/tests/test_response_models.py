"""Tail monotonicity penalty of response_models.BornModel.  Run: python -m pytest tests/test_response_models.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from response_models import BornModel, Grid                      # noqa: E402


def _setup(seed=0, scale=0.3):
    mod = BornModel(6, 2, 2, "nsc", (np.log(100.0), np.log(2000.0)))
    rng = np.random.default_rng(seed)
    Bc = np.zeros((7, 3), complex); Bc[0, 0] = 1
    Bc += scale * (rng.normal(size=Bc.shape) + 1j * rng.normal(size=Bc.shape)); Bc[0, 0] = 1
    th = mod.pack([1.0, 0.01, 0.0], [0.0, 0.8, 0.06], Bc)
    pt = np.exp(np.linspace(mod.x0, mod.x1, 7))
    j, ra, rb, sg = [], [], [], []
    for jj in range(len(pt)):
        for lo, hi, s in ((1.2, 2.0, 1.0), (0.0, 0.8, -1.0)):
            r = np.linspace(lo, hi, 81)
            j += [jj] * 80; ra += list(r[:-1]); rb += list(r[1:]); sg += [s] * 80
    T = dict(j=np.array(j), ra=np.array(ra), rb=np.array(rb), sgn=np.array(sg), w=np.full(len(j), 1.0))
    return mod, th, Grid(mod, pt), T


def test_mono_penalty_gradient():
    mod, th, g, T = _setup()
    v, gr = mod.mono_penalty(th, g, T)
    assert v > 0
    num = np.array([(mod.mono_penalty(th + e, g, T, False)[0] - mod.mono_penalty(th - e, g, T, False)[0]) / 2e-6
                    for e in np.eye(len(th)) * 1e-6])
    assert np.allclose(gr, num, rtol=1e-4, atol=1e-6 * np.abs(num).max())


def test_mono_penalty_zero_for_gaussian():
    mod, th, g, T = _setup()
    m, sp, Bc = mod.unpack(th); Bc[:] = 0; Bc[0, 0] = 1
    v, gr = mod.mono_penalty(mod.pack(m, sp, Bc), g, T)
    assert v == 0 and not gr.any()


def test_mono_penalty_counts_rises_only():
    """the penalty equals sum w relu(sgn * diff ln p)^2 computed from the density directly."""
    mod, th, g, T = _setup(seed=3, scale=0.5)
    v, _ = mod.mono_penalty(th, g, T, grad=False)
    pa = np.array([mod.pdf_grid(th, g.pt[j], np.array([r]))[0, 0] for j, r in zip(T["j"], T["ra"])])
    pb = np.array([mod.pdf_grid(th, g.pt[j], np.array([r]))[0, 0] for j, r in zip(T["j"], T["rb"])])
    d = T["sgn"] * (np.log(pb) - np.log(pa))
    assert np.isclose(v, np.sum(T["w"] * np.maximum(d, 0) ** 2), rtol=1e-8)
