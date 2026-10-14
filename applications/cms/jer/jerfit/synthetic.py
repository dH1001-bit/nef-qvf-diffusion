"""A toy sample in the export schema, for developing and testing the fit
without the cluster.

Generator pt follows a steeply falling spectrum produced slice by slice, as
the H_T-binned simulation is: each slice draws uniformly in log pt and carries
importance weights, so that -- exactly as for the real cross-section weights
-- the weights vary within a slice, jump between slices, and all slices add
into one spectrum with no further factor.  The response is Gaussian with the
NSC width, plus an optional low-side tail component to stand in for the
out-of-cone shoulder seen in the simulation (off by default, so the Gaussian
closure test is exact).
"""
from __future__ import annotations
import numpy as np
from .rows import Rows
from .width import nsc_sigma

TRUE = dict(mu=1.0, N=2.0, S=0.9, C=0.04)
SLICES = [(30, 100), (100, 200), (200, 300), (300, 500), (500, 700),
          (700, 1000), (1000, 1500), (1500, 2000), (2000, 3500)]


def make(n_per_slice=20000, params=TRUE, spectral_index=5.0, tail_fraction=0.0,
         tail_loc=0.7, tail_width=0.12, absy_max=2.5, seed=0, era="TOY", slices=SLICES):
    """Return a list of Rows, one per slice, each with its own metadata (as
    the export writes one file per input file)."""
    rng = np.random.default_rng(seed)
    out = []
    for i, (a, b) in enumerate(slices):
        u = rng.uniform(np.log(a), np.log(b), n_per_slice)
        pt = np.exp(u)
        # target spectrum f(pt) ~ pt^-index; proposal density in pt is
        # 1/(pt (log b - log a)); the weight is f/proposal/n.
        w = pt ** (-spectral_index) * pt * (np.log(b) - np.log(a)) / n_per_slice
        w *= 1e6                                     # an arbitrary 'pb' scale
        y = rng.uniform(-absy_max, absy_max, n_per_slice)
        s = nsc_sigma(pt, params["N"], params["S"], params["C"])
        r = rng.normal(params["mu"], s)
        if tail_fraction > 0:
            t = rng.uniform(size=n_per_slice) < tail_fraction
            r[t] = rng.normal(tail_loc, tail_width, t.sum())
        dr = rng.rayleigh(0.05, n_per_slice)
        dr2 = dr + rng.exponential(1.0, n_per_slice)
        cols = dict(response=r, genpt=pt, geny=y, recpt=r * pt, recy=y + rng.normal(0, 0.01, n_per_slice),
                    weight=w, dr=dr, dr2=dr2)
        gen = dict(g_pt=pt, g_y=y, g_dr=dr)
        meta = dict(era=era, slice="%dto%d" % (a, b), source="synthetic seed %d" % seed,
                    n_events=n_per_slice, sumw_events=float(w.sum()))
        out.append(Rows(cols, gen=gen, meta=meta))
    return out
