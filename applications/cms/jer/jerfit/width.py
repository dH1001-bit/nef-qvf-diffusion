"""The NSC width function and its parametrisation."""
import numpy as np

NSC_NAMES = ("N", "S", "C")


def nsc_sigma(pt, N, S, C):
    """sigma(pt)/pt = sqrt( sgn(N) N^2/pt^2 + S^2/pt + C^2 ).

    The JME convention lets the noise term carry a sign so that a negative
    N^2 (a width that *falls* faster than 1/sqrt(pt) at low pt) is
    representable; the radicand is clipped at a small positive value so the
    likelihood stays finite during minimisation."""
    pt = np.asarray(pt, float)
    v = np.sign(N) * N * N / (pt * pt) + S * S / pt + C * C
    return np.sqrt(np.maximum(v, 1e-12))


def nsc_dsigma(pt, N, S, C):
    """d sigma / d(N, S, C), shape (3, n)."""
    pt = np.asarray(pt, float)
    s = nsc_sigma(pt, N, S, C)
    return np.array([np.sign(N) * N / (pt * pt), S / pt, C * np.ones_like(pt)]) / s
