"""The global unbinned NSC fit across generator pt."""
from __future__ import annotations
import numpy as np
from .mle import mle, FitResult
from .gaussian import GaussianResponse, fit_bin, fit_core
from .width import nsc_sigma


def fit_nsc(rows, model=None, theta0=None, max_iter=20, tol=0.05) -> FitResult:
    """Maximise sum_i w_i log p(r_i | pt_i; theta) over the rows.

    With a windowed model (``model.k`` set) the window depends on theta, so
    the fit is iterated: fit, re-select the rows inside the window at the new
    theta, refit, until the parameters stop moving."""
    model = model or GaussianResponse()
    r, pt, w = rows.response, rows.genpt, rows.weight
    theta = np.asarray(theta0 if theta0 is not None else model.init(rows), float)
    res = None
    for it in range(max_iter):
        win = model.window(theta, pt)
        if win is None:
            res = mle(model.logpdf, model.names, theta, (r, pt), w,
                      bounds=getattr(model, "bounds", None))
            theta = res.values
            break
        a, b = win
        m = (r > a) & (r < b)
        res = mle(model.logpdf, model.names, theta, (r[m], pt[m], a[m], b[m]), w[m],
                  bounds=getattr(model, "bounds", None))
        # converged when the window re-selection moves the parameters by a
        # small fraction of their statistical errors
        moved = np.max(np.abs(res.values - theta) / np.maximum(res.errors, 1e-12))
        theta = res.values
        if moved < tol:
            break
    res.extra.update(iterations=it + 1, model=type(model).__name__, k=getattr(model, "k", None),
                     n_total=len(r))
    return res


def fit_per_bin(rows, pt_edges, core_k=None):
    """The conventional route: a Gaussian per generator-pt bin.  Returns a
    list of dicts with the bin, the weighted-mean pt, and the FitResult."""
    out = []
    for lo, hi in zip(pt_edges[:-1], pt_edges[1:]):
        m = (rows.genpt >= lo) & (rows.genpt < hi)
        if m.sum() < 20:
            continue
        r, w = rows.response[m], rows.weight[m]
        res = fit_core(r, w, k=core_k) if core_k else fit_bin(r, w)
        out.append(dict(lo=lo, hi=hi, pt=float((rows.genpt[m] * w).sum() / w.sum()),
                        n=int(m.sum()), fit=res))
    return out


def fit_nsc_to_points(points):
    """Chi^2 fit of the NSC form to per-bin (pt, sigma +- err) points, the
    binned reference for the unbinned result."""
    from scipy.optimize import least_squares
    pt = np.array([p["pt"] for p in points])
    s = np.array([p["fit"]["sigma"] for p in points])
    e = np.array([p["fit"].error("sigma") for p in points])

    def resid(theta):
        return (nsc_sigma(pt, *theta) - s) / e

    r = least_squares(resid, [1.0, 0.8, 0.03])
    J = r.jac
    cov = np.linalg.pinv(J.T @ J)
    return FitResult(("N", "S", "C"), r.x, cov, 0.5 * float(r.cost * 2), len(pt), len(pt),
                     float(len(pt)), r.success, extra=dict(chi2=float(2 * r.cost), ndf=len(pt) - 3))
