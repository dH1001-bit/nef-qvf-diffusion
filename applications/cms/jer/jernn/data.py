"""Rows for the network: one pass over npz files, only the needed columns, unbinned.

    D = read(files, target=Feature("response"), conds=[Feature("genpt:log"), Feature("receta:abs")],
             cuts=[Cut("dR<0.4")], weight="weight")

returns a dict of float32 arrays ``y`` (n,), ``X`` (n, d), float64 ``w`` (n,) and int32 ``file_id``,
plus ``meta``.  Nothing is binned: a new condition is a new column of X.

Weights.  ``weight`` names the weight column.  If it is ``None``/``"none"``, or a file has no such
column (real data), every row of that file gets weight 1 and ``meta['unit_weight_files']`` lists
it.  Mixing weighted and unweighted files is refused, because their weights would be on different
scales.  Negative weights are kept (they are legitimate in NLO samples); the loss stays the
weighted mean, but n_eff and the balancing below use |w|.

Memory: each kept row costs 4 (1 + d) + 8 + 4 bytes.  The full UL18 export (519 M rows) with three
conditions is about 14 GB; ``fraction`` keeps a random fraction of the rows of every file (a
uniform thinning, unbiased for the weighted distribution), ``max_rows`` stops early (only for
quick looks: the HT slices are not mixed then).
"""
from __future__ import annotations

import glob as _glob
import os
import time

import numpy as np

from .features import Cut, Feature


def expand(files):
    out = []
    for f in files if isinstance(files, (list, tuple)) else [files]:
        hits = sorted(_glob.glob(f))
        out.extend(hits or [f])
    missing = [f for f in out if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError("no such file(s): %s" % ", ".join(missing[:5]))
    return out


def read(files, target: Feature, conds: list[Feature], cuts: list[Cut] = (), weight="weight",
         fraction=1.0, max_rows=None, seed=0, verbose=True):
    files = expand(files)
    rng = np.random.default_rng(seed)
    use_w = weight not in (None, "", "none", "None")
    ys, Xs, ws, fid = [], [], [], []
    unit = []
    n_tot = 0
    t0 = time.time()
    for i, fn in enumerate(files):
        with np.load(fn, allow_pickle=False) as z:
            avail = set(z.files)
            cache = {}

            def get(name):
                if name not in cache:
                    cache[name] = z[name]
                return cache[name]

            y_all = target(get, avail)
            n = len(y_all)
            m = np.ones(n, bool)
            for c in cuts:
                m &= c.mask(get, avail)
            if fraction < 1.0:
                m &= rng.random(n) < fraction
            idx = np.flatnonzero(m)
            sub = lambda name: get(name)[idx]             # noqa: E731
            y = y_all[idx]
            X = np.stack([f(sub, avail) for f in conds], 1) if conds else np.zeros((len(idx), 0))
            if use_w and weight in avail:
                w = np.asarray(get(weight)[idx], np.float64)
            else:
                w = np.ones(len(idx))
                unit.append(os.path.basename(fn))
        ok = np.isfinite(y) & np.all(np.isfinite(X), 1) & np.isfinite(w)
        ys.append(y[ok].astype(np.float32)); Xs.append(X[ok].astype(np.float32)); ws.append(w[ok])
        fid.append(np.full(int(ok.sum()), i, np.int32))
        n_tot += int(ok.sum())
        if verbose:
            print("  %3d/%d %-36s %10d rows  %5.0f s" % (i + 1, len(files), os.path.basename(fn), ok.sum(),
                                                       time.time() - t0), flush=True)
        if max_rows and n_tot >= max_rows:
            break
    if use_w and unit and len(unit) < len(ys):
        raise ValueError("weight column %r missing in some files only (%s): weighted and unweighted rows "
                         "cannot be mixed" % (weight, ", ".join(unit[:5])))
    D = dict(y=np.concatenate(ys), X=np.concatenate(Xs), w=np.concatenate(ws), file_id=np.concatenate(fid))
    if max_rows:
        D = {k: v[:max_rows] for k, v in D.items()}
    D["meta"] = dict(files=[os.path.abspath(f) for f in files[: len(ys)]], target=target.spec,
                     conds=[f.spec for f in conds], cuts=[c.spec for c in cuts],
                     weight=weight if use_w and not unit else "unit", unit_weight_files=unit,
                     fraction=fraction, n_rows=len(D["y"]), n_eff=n_eff(D["w"]))
    return D


def n_eff(w):
    a = np.abs(w)
    return float(a.sum() ** 2 / max((a * a).sum(), 1e-300))


def split(n, val_frac, seed=0):
    """Random train/validation split of row indices (jets of one event may land on both sides;
    harmless for a density estimate, mentioned for completeness)."""
    perm = np.random.default_rng(seed + 12345).permutation(n)
    k = int(round(val_frac * n))
    return np.sort(perm[k:]), np.sort(perm[:k])


def balance(x, w, ref, nbins=100, power=1.0):
    """Factors g(x) along one condition x (e.g. ln pt_gen) such that every quantile bin b of x
    carries a total |weight| proportional to n_eff_b^power; bins, n_eff_b and the normalisation
    (sum g |w| = sum |w| over ``ref``) are taken from the rows ``ref`` (the training rows).

    power 1: as fit_response_global.py's default (--pt-weighting neff); for unit weights it changes
    nothing.  power 0: every bin the same total weight.

    Why this is allowed: the loss sum_i w_i g(x_i) ln p(y_i | x_i) still has its optimum at the
    true conditional density for every x, because g depends on the condition only.  It only moves
    where the network spends its capacity (with cross-section weights the lowest pt dominates
    everything, as in the global fit with --pt-weighting xsec).
    """
    x = np.asarray(x, np.float64)
    aw = np.abs(w)
    edges = np.unique(np.quantile(x[ref], np.linspace(0, 1, nbins + 1)))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    Wb = np.bincount(b[ref], aw[ref], len(edges) - 1)
    W2b = np.bincount(b[ref], aw[ref] ** 2, len(edges) - 1)
    neff_b = Wb ** 2 / np.maximum(W2b, 1e-300)
    gb = np.where(Wb > 0, neff_b ** power / np.maximum(Wb, 1e-300), 0.0)
    g = gb[b]
    return g * aw[ref].sum() / max((g[ref] * aw[ref]).sum(), 1e-300)


def tail_reference(X, y, w, ref, nbins=20, min_rows=200, max_sample=5_000_000, seed=0):
    """Per-row core of the response for the tail monotonicity penalty (jernn.model.mono_penalty):
    the conditions are cut into cells (``nbins`` quantile bins of every column of X, from the rows
    ``ref``), and every cell gets the |w|-weighted median and sigma_68 = (Q84 - Q16)/2 of y over its
    ``ref`` rows (at most ``max_sample`` of them, drawn uniformly).  Cells with fewer than
    ``min_rows`` rows take the values of all sampled rows.

    Returns center (n,), width (n,) and the cell index (n,) of every row of X (see cell_prob)."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    cell = np.zeros(n, np.int64)
    for k in range(d):
        xk = np.asarray(X[:, k], np.float64)                     # one column at a time (memory)
        e = np.unique(np.quantile(xk[ref], np.linspace(0, 1, nbins + 1)))
        b = np.clip(np.searchsorted(e, xk, side="right") - 1, 0, max(len(e) - 2, 0))
        cell = cell * nbins + b
    smp = np.asarray(ref)
    if len(smp) > max_sample:
        smp = np.sort(rng.choice(smp, max_sample, replace=False))
    aw = np.abs(np.asarray(w, np.float64))

    def q3(yy, ww):
        o = np.argsort(yy)
        c = np.cumsum(ww[o]); c = c / c[-1]
        i = np.minimum(np.searchsorted(c, [0.16, 0.5, 0.84]), len(yy) - 1)
        q = yy[o][i]
        return q[1], max(0.5 * (q[2] - q[0]), 1e-4)

    g_c, g_w = q3(np.asarray(y, np.float64)[smp], aw[smp])
    uc, inv = np.unique(cell[smp], return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uc) + 1))
    tab_c, tab_w = {}, {}
    ys, ws = np.asarray(y, np.float64)[smp][order], aw[smp][order]
    for u, lo, hi in zip(uc, bounds[:-1], bounds[1:]):
        if hi - lo >= min_rows and ws[lo:hi].sum() > 0:
            tab_c[u], tab_w[u] = q3(ys[lo:hi], ws[lo:hi])
    keys = np.array(sorted(tab_c), np.int64)
    center, width = np.full(n, g_c), np.full(n, g_w)
    if len(keys):
        pos = np.clip(np.searchsorted(keys, cell), 0, len(keys) - 1)
        hit = keys[pos] == cell
        center[hit] = np.array([tab_c[k] for k in keys])[pos[hit]]
        width[hit] = np.array([tab_w[k] for k in keys])[pos[hit]]
    return center, width, cell


def cell_prob(cell, rows):
    """Draw probabilities (len(cell),): zero outside ``rows``, inside proportional to 1 / (rows of the
    same cell), so that a draw picks every populated cell equally often.  For the monotonicity
    penalty: an average over the condition space (as the pt average of fit_response_global.py),
    not over the events, which would leave the sparse corners, where the wiggles are, unchecked."""
    _, inv = np.unique(cell[rows], return_inverse=True)
    p = np.zeros(len(cell))
    p[rows] = 1.0 / np.bincount(inv)[inv]
    return p / p.sum()
