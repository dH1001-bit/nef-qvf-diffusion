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
