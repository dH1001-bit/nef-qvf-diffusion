"""The Gaussian response model and the two single-bin fits.

Single bin:  r ~ N(mu, sigma), weights w.
  * ``fit_bin``  -- full range, closed form (weighted mean and variance) with
                    sandwich errors.
  * ``fit_core`` -- the campaign convention: a Gaussian truncated to
                    mu +- k sigma, window iterated to convergence.

Across pt:  r ~ N(mu, nsc_sigma(pt; N, S, C)) -- see ``GaussianResponse`` and
``jerfit.nsc``.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr, log_ndtr
from .mle import mle, sandwich, FitResult
from .width import nsc_sigma, NSC_NAMES

_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def log_window_mass(alpha, beta):
    """log( Phi(beta) - Phi(alpha) ) for alpha < beta, stable far in either
    tail: computed from log Phi on the side where the mass is small."""
    alpha = np.asarray(alpha, float); beta = np.asarray(beta, float)
    flip = alpha + beta > 0                      # window right of the centre
    lo = np.where(flip, -beta, alpha)
    hi = np.where(flip, -alpha, beta)
    lhi, llo = log_ndtr(hi), log_ndtr(lo)
    return lhi + np.log1p(-np.exp(np.minimum(llo - lhi, -1e-300)))


def _gauss_logpdf(theta, r):
    mu, sigma = theta
    z = (r - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - _LOG_SQRT_2PI


def fit_bin(r, w) -> FitResult:
    """Weighted Gaussian MLE over the full range, closed form."""
    r = np.asarray(r, float); w = np.asarray(w, float)
    sw = w.sum()
    mu = (w * r).sum() / sw
    sigma = np.sqrt((w * (r - mu) ** 2).sum() / sw)
    theta = np.array([mu, sigma])
    cov, H = sandwich(_gauss_logpdf, theta, (r,), w)
    nll = -float(_gauss_logpdf(theta, r) @ w)
    return FitResult(("mu", "sigma"), theta, cov, nll, len(r), float(sw),
                     float(sw ** 2 / (w ** 2).sum()), True, extra=dict(hessian=H))


def _trunc_logpdf(theta, r, a, b):
    """Gaussian truncated to the FIXED window [a, b]: normalised by
    Phi((b-mu)/sigma) - Phi((a-mu)/sigma), so that sigma is the width of the
    untruncated Gaussian consistent with the rows inside the window.  (A
    constant normalisation would make the fit on the truncated sample an
    ordinary Gaussian fit, whose sigma shrinks, and an iterated window then
    collapses.)  Parametrised by (mu, log sigma)."""
    mu, logsig = theta
    sigma = np.exp(logsig)
    z = (r - mu) / sigma
    norm = log_window_mass((a - mu) / sigma, (b - mu) / sigma)
    return -0.5 * z * z - logsig - _LOG_SQRT_2PI - norm


def fit_core(r, w, k=1.5, max_iter=20, tol=0.05) -> FitResult:
    """Iterated core fit: truncated Gaussian on mu +- k sigma, window re-centred
    on the fit until the parameters move by less than tol times their errors."""
    r = np.asarray(r, float); w = np.asarray(w, float)
    full = fit_bin(r, w)
    mu, sigma = full.values
    for it in range(max_iter):
        a, b = mu - k * sigma, mu + k * sigma
        m = (r > a) & (r < b)
        res = mle(_trunc_logpdf, ("mu", "logsigma"), [mu, np.log(sigma)], (r[m], a, b), w[m])
        mu_new, sigma_new = res.values[0], np.exp(res.values[1])
        e_mu, e_sig = res.errors[0], sigma_new * res.errors[1]
        moved = max(abs(mu_new - mu) / max(e_mu, 1e-12), abs(sigma_new - sigma) / max(e_sig, 1e-12))
        mu, sigma = mu_new, sigma_new
        if moved < tol:
            break
    # report (mu, sigma): transform the covariance of (mu, log sigma)
    Jac = np.diag([1.0, sigma])
    cov = Jac @ res.cov @ Jac
    return FitResult(("mu", "sigma"), np.array([mu, sigma]), cov, res.nll, res.n,
                     res.sumw, res.n_eff, res.converged, res.message,
                     extra=dict(k=k, iterations=it + 1, window=(mu - k * sigma, mu + k * sigma)))


class GaussianResponse:
    """r | pt ~ N(mu, nsc_sigma(pt; N, S, C)).  Parameters (mu, N, S, C).

    Optional core window ``k``: the density is truncated to the per-row
    window mu +- k sigma(pt) of the *previous* iterate (``window(theta, pt)``),
    normalised on that fixed window, and the fitter iterates until the window
    is self-consistent.  The interface -- ``names``, ``bounds``,
    ``init(rows)``, ``window(theta, pt) -> (a, b) or None``,
    ``logpdf(theta, r, pt, a, b)`` -- is what a non-Gaussian response model
    has to provide to plug into ``jerfit.nsc.fit_nsc``.
    """
    names = ("mu",) + NSC_NAMES

    def __init__(self, k=None):
        self.k = k

    def init(self, rows):
        pt = rows.genpt
        mu = fit_bin(rows.response, rows.weight)["mu"]
        # a crude start: constant term from the high-pt third, S from the rest
        hi = pt > np.quantile(pt, 0.67)
        C = fit_bin(rows.response[hi], rows.weight[hi])["sigma"] if hi.sum() > 50 else 0.03
        return np.array([mu, 1.0, 0.8, C])

    bounds = [(None, None), (-50.0, 50.0), (0.0, 10.0), (0.0, 1.0)]

    def sigma(self, theta, pt):
        return nsc_sigma(pt, *theta[1:4])

    def logpdf(self, theta, r, pt, a=None, b=None):
        s = self.sigma(theta, pt)
        z = (r - theta[0]) / s
        lp = -0.5 * z * z - np.log(s) - _LOG_SQRT_2PI
        if a is not None:
            lp = lp - log_window_mass((a - theta[0]) / s, (b - theta[0]) / s)
        return lp

    def window(self, theta, pt):
        """Per-row window (a, b) at theta, or None for the full range."""
        if self.k is None:
            return None
        s = self.k * self.sigma(theta, pt)
        return theta[0] - s, theta[0] + s
