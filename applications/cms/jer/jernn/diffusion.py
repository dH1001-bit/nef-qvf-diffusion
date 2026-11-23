"""Closed-form noising of a Born density, in numpy, for using the network as the ground truth of a
diffusion model.

The network gives, for every condition x, a Born density in the reference frame z = (y - mu)/sigma

    p(y | x) = N(z) |h(z)|^2 / (sigma ||c||^2),     h(z) = sum_{n<=J} c_n phi_n(z),

phi_n = He_n / sqrt(n!) orthonormal under N(0, 1).  |h|^2 is a polynomial of degree 2J, so

    p(y | x) = N(z) / sigma * sum_{k<=2J} a_k phi_k(z),   a_k = sum_{mn} conj(c_m) c_n Lambda[m, n, k] / ||c||^2,

with the Hermite linearisation Lambda (a_0 = 1).  Under the Ornstein-Uhlenbeck process with the
condition's own reference as stationary law,

    dy = -(y - mu(x)) dt + sqrt(2) sigma(x) dW,
    y_t | y_0  ~  N(mu + e^{-t} (y_0 - mu), sigma^2 (1 - e^{-2t})),

phi_k is an eigenfunction with eigenvalue -k (Mehler), so the noised density at time t is

    p_t(y | x) = N(z) / sigma * sum_k a_k e^{-k t} phi_k(z),

exact, positive, and with an exact score d/dy ln p_t.  This is the repository's reference flow
(z -> e^{-t} z) for the normal family; the network supplies (mu, sigma, c) per condition.
"""
from __future__ import annotations

from functools import lru_cache
from math import lgamma

import numpy as np

LOG2PI = np.log(2 * np.pi)


def hermite(z, K):
    """phi_0 ... phi_K at z (last axis), by the stable recurrence."""
    z = np.asarray(z, float)
    out = np.empty(z.shape + (K + 1,))
    out[..., 0] = 1.0
    if K >= 1:
        out[..., 1] = z
    for n in range(1, K):
        out[..., n + 1] = (z * out[..., n] - np.sqrt(n) * out[..., n - 1]) / np.sqrt(n + 1)
    return out


@lru_cache(maxsize=None)
def linearisation(J):
    """Lambda[m, n, k] = E[phi_m phi_n phi_k] under N(0, 1), m, n <= J, k <= 2J:
    He_m He_n = sum_j C(m,j) C(n,j) j! He_{m+n-2j}."""
    L = np.zeros((J + 1, J + 1, 2 * J + 1))
    for m in range(J + 1):
        for n in range(J + 1):
            for j in range(min(m, n) + 1):
                k = m + n - 2 * j
                # C(m,j) C(n,j) j! sqrt(k!) / sqrt(m! n!), in logs
                L[m, n, k] = np.exp(lgamma(m + 1) + lgamma(n + 1) - lgamma(j + 1) - lgamma(m - j + 1)
                                    - lgamma(n - j + 1) + 0.5 * (lgamma(k + 1) - lgamma(m + 1) - lgamma(n + 1)))
    L.flags.writeable = False                        # cached: shared, read only
    return L


def density_coefficients(c):
    """c (..., J+1) complex -> a (..., 2J+1) real, a_0 = 1."""
    c = np.asarray(c, complex)
    J = c.shape[-1] - 1
    L = linearisation(J)
    a = np.einsum("...m,...n,mnk->...k", np.conj(c), c, L).real
    return a / np.sum(np.abs(c) ** 2, -1, keepdims=True)


def _coeffs_for(z, c, t):
    """a_k e^{-kt} shaped to multiply hermite(z): c (rows, J+1) with z (rows,) or (rows, m), or
    c (J+1,) / (1, J+1) with any z."""
    a = density_coefficients(np.atleast_2d(c))
    a = a * np.exp(-t * np.arange(a.shape[-1]))
    if z.ndim >= 2:
        a = a.reshape((a.shape[0],) + (1,) * (z.ndim - 1) + (a.shape[-1],))
    elif len(a) == 1:
        a = a[0]
    return a


def noised_pdf(y, mu, sigma, c, t):
    """p_t(y | x).  Rows: y (rows,) or (rows, m), mu and sigma broadcast to y, c (rows, J+1);
    or one condition: c (J+1,), any y.  t >= 0 scalar; t = 0 is the network's density."""
    z = (np.asarray(y, float) - mu) / sigma
    a = _coeffs_for(z, c, t)
    s = np.sum(hermite(z, a.shape[-1] - 1) * a, -1)
    return np.exp(-0.5 * z * z - 0.5 * LOG2PI) / sigma * s


def noised_logpdf(y, mu, sigma, c, t):
    """ln p_t(y | x), in log space (no underflow far in the tails); shapes as noised_pdf."""
    z = (np.asarray(y, float) - mu) / sigma
    a = _coeffs_for(z, c, t)
    s = np.sum(hermite(z, a.shape[-1] - 1) * a, -1)
    return -0.5 * z * z - 0.5 * LOG2PI - np.log(sigma) + np.log(np.maximum(s, 1e-300))


_GLX, _GLW = np.polynomial.legendre.leggauss(64)


def log_tail(zb, c, side, length=25.0):
    """ln P(z < zb) (side -1) or ln P(z >= zb) (side +1) of the Born density N(z)|h(z)|^2/||c||^2
    per row (zb (rows,), c (rows, J+1)); Gauss-Legendre over ``length`` units beyond the boundary,
    in log space.  The same rule as the network's censoring (jernn.model)."""
    zb = np.asarray(zb, float)
    lo = zb - length if side < 0 else zb
    z = lo[:, None] + (_GLX[None, :] + 1) * (length / 2)
    lq = noised_logpdf(z, 0.0, 1.0, c, 0.0) + np.log(_GLW * length / 2)[None, :]
    m = lq.max(1, keepdims=True)
    return (m + np.log(np.exp(lq - m).sum(1, keepdims=True)))[:, 0]


try:
    from scipy.special import ndtr as _ndtr
except ImportError:                                  # pragma: no cover
    from math import erfc as _erfc
    _ndtr = np.vectorize(lambda x: 0.5 * _erfc(-x / np.sqrt(2.0)))


def noised_cdf(y, mu, sigma, c, t):
    """P(Y_t <= y | x) in closed form; shapes as noised_pdf.

    With p_t = N(z)/sigma sum_k a_k e^{-kt} phi_k(z) and int_{-inf}^z N(u) phi_k(u) du =
    -N(z) phi_{k-1}(z)/sqrt(k) (k >= 1; from (N He_{k-1})' = -N He_k):

        F_t(y) = Phi(z) - N(z) sum_{k>=1} a_k e^{-kt} phi_{k-1}(z) / sqrt(k),   a_0 = 1."""
    z = (np.asarray(y, float) - mu) / sigma
    return cdf_from_a(z, _coeffs_for(z, c, t))


def cdf_from_a(z, a):
    """The same in the reference frame, from density coefficients a (already shaped to z, as
    _coeffs_for returns them; e.g. a[:, None, :] for z (rows, m)): for repeated evaluations of
    the same rows (bisection) without recomputing a."""
    K = a.shape[-1] - 1
    P = hermite(z, K - 1)
    s = np.sum(P * (a[..., 1:] / np.sqrt(np.arange(1, K + 1))), -1)
    return _ndtr(z) - np.exp(-0.5 * z * z - 0.5 * LOG2PI) * s


def pdf_from_a(z, a):
    """N(z) sum_k a_k phi_k(z): the density in z (divide by sigma for y)."""
    return np.exp(-0.5 * z * z - 0.5 * LOG2PI) * np.sum(hermite(z, a.shape[-1] - 1) * a, -1)


def noised_score(y, mu, sigma, c, t):
    """d/dy ln p_t(y | x), exact (phi_k' = sqrt(k) phi_{k-1}); shapes as noised_pdf."""
    z = (np.asarray(y, float) - mu) / sigma
    a = _coeffs_for(z, c, t)
    K = a.shape[-1] - 1
    P = hermite(z, K)
    s = np.sum(P * a, -1)
    ds = np.sum(P[..., :-1] * a[..., 1:] * np.sqrt(np.arange(1, K + 1)), -1)
    return (ds / s - z) / sigma


def forward_sample(y0, mu, sigma, t, rng=None):
    """One exact draw of y_t | y_0 (the noising step of the diffusion)."""
    rng = np.random.default_rng(rng)
    e = np.exp(-t)
    return mu + e * (np.asarray(y0) - mu) + sigma * np.sqrt(1 - e * e) * rng.standard_normal(np.shape(y0))
