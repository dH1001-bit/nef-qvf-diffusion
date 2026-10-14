"""Weighted unbinned maximum likelihood with sandwich errors.

The rows carry cross-section weights w_i, so the objective is the weighted
log likelihood  sum_i w_i log p(r_i | x_i; theta).  Its maximiser is the
usual estimator; its covariance is NOT the inverse Hessian (that would treat
sum w as a number of events) but the sandwich

    cov = H^-1 J H^-1,   H = -sum_i w_i d^2 log p_i,   J = sum_i w_i^2 s_i s_i^T,

with s_i the per-row score.  For unit weights both coincide with the usual
inverse Fisher information.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize


@dataclass
class FitResult:
    names: tuple
    values: np.ndarray
    cov: np.ndarray
    nll: float                 # -sum w log p at the optimum
    n: int                     # rows used
    sumw: float
    n_eff: float
    converged: bool
    message: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def errors(self):
        return np.sqrt(np.maximum(np.diag(self.cov), 0.0))

    def __getitem__(self, name):
        return self.values[self.names.index(name)]

    def error(self, name):
        return self.errors[self.names.index(name)]

    def __str__(self):
        lines = ["%-4s = %12.6g +- %-10.3g" % (k, v, e)
                 for k, v, e in zip(self.names, self.values, self.errors)]
        lines.append("rows %d   sum w %.6g   n_eff %.1f   -logL %.6g   %s"
                     % (self.n, self.sumw, self.n_eff, self.nll,
                        "converged" if self.converged else "NOT CONVERGED: " + self.message))
        if self.extra.get("at_bound"):
            lines.append("AT BOUND: %s (errors there are meaningless)" % ", ".join(self.extra["at_bound"]))
        return "\n".join(lines)


def _score(logpdf, theta, args, eps):
    """Per-row score by central differences, shape (p, n)."""
    p = len(theta)
    out = []
    for k in range(p):
        d = np.zeros(p); d[k] = eps[k]
        out.append((logpdf(theta + d, *args) - logpdf(theta - d, *args)) / (2 * eps[k]))
    return np.array(out)


def _hessian(logpdf, theta, args, w, eps):
    """Hessian of sum w log p by central differences of the gradient."""
    p = len(theta)

    def grad(t):
        return _score(logpdf, t, args, eps) @ w

    H = np.zeros((p, p))
    for k in range(p):
        d = np.zeros(p); d[k] = eps[k]
        H[:, k] = (grad(theta + d) - grad(theta - d)) / (2 * eps[k])
    return 0.5 * (H + H.T)


def sandwich(logpdf, theta, args, w, eps=None):
    """Sandwich covariance at theta for the weighted log likelihood."""
    theta = np.asarray(theta, float)
    if eps is None:
        eps = 1e-4 * np.maximum(np.abs(theta), 1e-3)
    S = _score(logpdf, theta, args, eps)                # (p, n)
    J = (S * w * w) @ S.T
    H = -_hessian(logpdf, theta, args, w, eps)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Hi = np.linalg.pinv(H)
    return Hi @ J @ Hi, H


def mle(logpdf, names, theta0, args, w, bounds=None, eps=None):
    """Minimise -sum w log p over theta.  ``logpdf(theta, *args)`` returns the
    per-row log density (n,).  Returns a FitResult."""
    w = np.asarray(w, float)
    theta0 = np.asarray(theta0, float)
    # Cross-section weights can sum to anything; the minimiser's tolerances
    # are absolute, so minimise with weights rescaled to sum to n.  The
    # sandwich covariance is invariant under a common rescaling of w.
    wn = w * (len(w) / w.sum())

    def nll(t):
        v = logpdf(t, *args)
        if not np.all(np.isfinite(v)):
            return 1e300
        return -float(v @ wn)

    res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=2000, ftol=1e-14, gtol=1e-8))
    cov, H = sandwich(logpdf, res.x, args, w, eps)
    at_bound = []
    if bounds is not None:
        for nm, x, (lo, hi) in zip(names, res.x, bounds):
            if (lo is not None and x <= lo + 1e-9) or (hi is not None and x >= hi - 1e-9):
                at_bound.append(nm)
    return FitResult(tuple(names), res.x, cov, float(res.fun * w.sum() / len(w)), len(w), float(w.sum()),
                     float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0,
                     bool(res.success), str(res.message), extra=dict(hessian=H, at_bound=at_bound))
