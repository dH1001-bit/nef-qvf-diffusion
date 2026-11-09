"""Shared pieces of the jet-response fits: the Hermite basis, the Gaussian log density and the
pt-smooth Born (amplitude) model of fit_response_global.py.  numpy/scipy only.

The pt-smooth Born model, for one |eta| bin:

    p(r | pt) = N(r; mu(pt), sigma(pt)) |h(z; pt)|^2 / ||c~(pt)||^2,   z = (r - mu) / sigma
    h(z; pt)  = sum_{n=0}^{J} c~_n(pt) phi_n(z),                      phi_n = He_n / sqrt(n!)

    u          = 2 (ln pt - x0) / (x1 - x0) - 1          (the fitted ln pt range mapped to [-1, 1])
    mu(pt)     = sum_{k <= mu_deg} m_k P_k(u)             (Legendre P_k)
    sigma(pt)  = sqrt(A t^2 + B^2 t + C^2),  t = pt_scale / pt      (sigma_form "nsc", A may be < 0)
               = |C|                                              (sigma_form "const")
    c~_n(pt)   = sum_{k <= K} b_nk P_k(u),  b_nk complex,  b_00 = 1 fixed (overall scale and phase)

Because the phi_n are orthonormal under N(0, 1), int N |h|^2 dz = ||c~||^2: every conditional
density is normalised exactly, at every pt, without an integral.  Parameter vector theta:
(m_0 .. m_mu_deg, sigma parameters, Re b_nk without b_00, Im b_nk without b_00), b flattened
row-major in (n, k).  Gradients are analytic (see BornModel.lnl).
"""
from math import factorial
import numpy as np
from scipy.special import logsumexp

LOG2PI = np.log(2 * np.pi)
_GL_T, _GL_W = np.polynomial.legendre.leggauss(96)      # tail integrals
_SIG2_MIN = 1e-8


def hermite_basis(z, J):
    """phi_n(z) = He_n(z) / sqrt(n!), n = 0 ... J: orthonormal under N(0, 1); last axis is the degree."""
    return np.polynomial.hermite_e.hermevander(z, J) / np.sqrt([float(factorial(k)) for k in range(J + 1)])


def log_normal(r, mu, sigma):
    z = (r - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * LOG2PI


class Grid:
    """per-pt quantities that do not depend on theta."""
    def __init__(self, model, pt):
        self.pt = np.asarray(pt, float)
        self.u = model.u_of(self.pt)
        self.t = model.pt_scale / self.pt
        self.Lmu = np.polynomial.legendre.legvander(self.u, model.mu_deg)
        self.Lc = np.polynomial.legendre.legvander(self.u, model.K)
        self.M = len(self.pt)


class BornModel:
    def __init__(self, J, K, mu_deg=2, sigma_form="nsc", x_range=(np.log(100.0), np.log(1000.0)), pt_scale=100.0):
        self.J, self.K, self.mu_deg, self.sigma_form = int(J), int(K), int(mu_deg), sigma_form
        self.x0, self.x1 = float(x_range[0]), float(x_range[1])
        self.pt_scale = float(pt_scale)
        self.nmu = self.mu_deg + 1
        self.nsig = 3 if sigma_form == "nsc" else 1
        self.nc = (self.J + 1) * (self.K + 1)
        self.oc = self.nmu + self.nsig                    # offset of the c block
        self.P = self.oc + 2 * (self.nc - 1)
        self.sqrtn = np.sqrt(np.arange(1, self.J + 1))

    # --- bookkeeping
    def u_of(self, pt):
        x = np.log(pt)
        return np.zeros_like(x) if self.x1 <= self.x0 else 2 * (x - self.x0) / (self.x1 - self.x0) - 1

    def names(self):
        nm = ["m%d" % k for k in range(self.nmu)]
        nm += ["A", "B", "C"] if self.sigma_form == "nsc" else ["C"]
        bk = ["%d,%d" % (n, k) for n in range(self.J + 1) for k in range(self.K + 1)][1:]
        return nm + ["Re b" + s for s in bk] + ["Im b" + s for s in bk]

    def to_dict(self, theta):
        return dict(J=self.J, K=self.K, mu_deg=self.mu_deg, sigma_form=self.sigma_form, x_range=[self.x0, self.x1],
                    pt_scale=self.pt_scale, theta=[float(v) for v in theta], names=self.names())

    @classmethod
    def from_dict(cls, d):
        return cls(d["J"], d["K"], d["mu_deg"], d["sigma_form"], d["x_range"], d["pt_scale"]), np.asarray(d["theta"])

    def unpack(self, th):
        m = th[:self.nmu]
        sp = th[self.nmu:self.oc]
        rest = th[self.oc:]
        re = np.r_[1.0, rest[:self.nc - 1]]
        im = np.r_[0.0, rest[self.nc - 1:]]
        return m, sp, (re + 1j * im).reshape(self.J + 1, self.K + 1)

    def pack(self, m, sp, B):
        B = np.asarray(B, complex).reshape(-1)
        B = B / B[0]                                      # b_00 = 1: scale and phase are not observable
        return np.r_[m, sp, B.real[1:], B.imag[1:]]

    def extend(self, th, K_new, mu_deg_new=None):
        """theta of a model with a higher K (and mu_deg): the new coefficients start at 0."""
        m, sp, B = self.unpack(th)
        B2 = np.zeros((self.J + 1, K_new + 1), complex)
        B2[:, :self.K + 1] = B
        mdn = self.mu_deg if mu_deg_new is None else mu_deg_new
        m2 = np.zeros(mdn + 1); m2[:self.nmu] = m
        new = BornModel(self.J, K_new, mdn, self.sigma_form, (self.x0, self.x1), self.pt_scale)
        return new, new.pack(m2, sp, B2)

    # --- theta-dependent per-pt quantities
    def ref(self, th, g):
        m, sp, B = self.unpack(th)
        mu = g.Lmu @ m
        if self.sigma_form == "nsc":
            A, Bs, C = sp
            s2 = A * g.t ** 2 + Bs * Bs * g.t + C * C
            d2 = np.stack([g.t ** 2, 2 * Bs * g.t, np.full_like(g.t, 2 * C)], 1)    # d sigma^2 / d (A, B, C)
        else:
            s2 = np.full(g.M, sp[0] ** 2)
            d2 = np.full((g.M, 1), 2 * sp[0])
        ok = s2 > _SIG2_MIN
        sig = np.sqrt(np.where(ok, s2, _SIG2_MIN))
        dsig = d2 / (2 * sig[:, None]) * ok[:, None]
        Ct = g.Lc @ B.T                                   # (M, J+1): c~_n at every pt
        norm = np.sum(np.abs(Ct) ** 2, 1)
        Gn = np.zeros((g.M, self.P))
        fac = 2 / norm[:, None, None] * g.Lc[:, None, :]
        Gn[:, self.oc:self.oc + self.nc - 1] = (Ct.real[:, :, None] * fac).reshape(g.M, -1)[:, 1:]
        Gn[:, self.oc + self.nc - 1:] = (Ct.imag[:, :, None] * fac).reshape(g.M, -1)[:, 1:]
        return dict(mu=mu, sig=sig, dsig=dsig, Ct=Ct, lnorm=np.log(norm), Gnorm=Gn)

    def local(self, R, j, r):
        """unnormalised log density lu = ln N(r; mu, sigma) + ln |h|^2 at points (pt index j, r) and
        the pieces of its gradient."""
        s = R["sig"][j]
        z = (r - R["mu"][j]) / s
        Phi = hermite_basis(z, self.J)
        C = R["Ct"][j]
        h = np.sum(Phi * C, 1)
        hp = np.sum(Phi[:, :-1] * C[:, 1:] * self.sqrtn, 1) if self.J > 0 else np.zeros_like(h)   # phi_n' = sqrt(n) phi_{n-1}
        h2 = np.maximum(np.abs(h) ** 2, 1e-300)
        lu = -0.5 * z * z - np.log(s) - 0.5 * LOG2PI + np.log(h2)
        D = 2 * (np.conj(h) * hp).real / h2
        return lu, (z - D) / s, (z * z - 1 - D * z) / s, (2 * h.real / h2)[:, None] * Phi, (2 * h.imag / h2)[:, None] * Phi

    def per_j(self, g, R, j, w, loc):
        """sum over the points of every pt index of w * grad lu: (M, P)."""
        _, dmu, dsg, Rp, Ip = loc
        M = g.M
        out = np.zeros((M, self.P))
        out[:, :self.nmu] = np.bincount(j, w * dmu, M)[:, None] * g.Lmu
        out[:, self.nmu:self.oc] = np.bincount(j, w * dsg, M)[:, None] * R["dsig"]
        for blk, X in ((0, Rp), (1, Ip)):
            Aj = np.stack([np.bincount(j, w * X[:, n], M) for n in range(self.J + 1)], 1)
            o = self.oc + blk * (self.nc - 1)
            out[:, o:o + self.nc - 1] = (Aj[:, :, None] * g.Lc[:, None, :]).reshape(M, -1)[:, 1:]
        return out

    def per_point(self, g, R, j, loc):
        """grad lu at every point: (n, P)."""
        _, dmu, dsg, Rp, Ip = loc
        n = len(j)
        out = np.zeros((n, self.P))
        out[:, :self.nmu] = dmu[:, None] * g.Lmu[j]
        out[:, self.nmu:self.oc] = dsg[:, None] * R["dsig"][j]
        Lc = g.Lc[j]
        out[:, self.oc:self.oc + self.nc - 1] = (Rp[:, :, None] * Lc[:, None, :]).reshape(n, -1)[:, 1:]
        out[:, self.oc + self.nc - 1:] = (Ip[:, :, None] * Lc[:, None, :]).reshape(n, -1)[:, 1:]
        return out

    def tail_nodes(self, R, js, edge, side):
        """Gauss-Legendre nodes for int_{-inf}^{edge} (side 'lo') or int_{edge}^{inf} ('hi') at pt
        indices js, in z out to 25 (beyond: N(z) |h|^2 < 1e-100)."""
        mu, sig = R["mu"][js], R["sig"][js]
        z0 = (edge - mu) / sig
        a, b = (np.minimum(z0 - 10, -25), z0) if side == "lo" else (z0, np.maximum(z0 + 10, 25))
        half, mid = 0.5 * (b - a), 0.5 * (b + a)
        zq = mid[:, None] + half[:, None] * _GL_T
        jq = np.repeat(js, len(_GL_T))
        rq = (mu[:, None] + sig[:, None] * zq).ravel()
        lnw = np.log(sig[:, None] * half[:, None] * _GL_W).ravel()
        return jq, rq, lnw

    def lnl(self, th, D, grad=True):
        """weighted log-likelihood of the binned data D (see fit_response_global.py) and its gradient:
            sum_E S_E ln p(r_E | pt_jE) + sum_j [Wlo_j ln P(r < lo | pt_j) + Whi_j ln P(r >= hi | pt_j)]"""
        g, R = D["grid"], self.ref(th, D["grid"])
        loc = self.local(R, D["j"], D["r"])
        val = D["S"] @ loc[0] - D["Wtot"] @ R["lnorm"]
        gr = self.per_j(g, R, D["j"], D["S"], loc).sum(0) - D["Wtot"] @ R["Gnorm"] if grad else None
        for side, edge in (("lo", D["lo"]), ("hi", D["hi"])):
            Wt = D["W" + side]
            js = np.flatnonzero(Wt > 0)
            if len(js) == 0:
                continue
            jq, rq, lnw = self.tail_nodes(R, js, edge, side)
            lq = self.local(R, jq, rq)
            a = (lnw + lq[0]).reshape(len(js), -1)
            lse = logsumexp(a, 1)
            val += Wt[js] @ lse
            if grad:
                pi = np.exp(a - lse[:, None]).ravel()
                gr += self.per_j(g, R, jq, pi * Wt[jq], lq).sum(0)
        return val, gr

    def scores(self, th, D):
        """per-observation-group scores and their sum w^2, for the sandwich V = sum S2 g g^T:
        every fine 2D bin, and the two tail masses of every pt bin."""
        g, R = D["grid"], self.ref(th, D["grid"])
        loc = self.local(R, D["j"], D["r"])
        G = [self.per_point(g, R, D["j"], loc) - R["Gnorm"][D["j"]]]
        w2 = [D["S2"]]
        for side, edge in (("lo", D["lo"]), ("hi", D["hi"])):
            js = np.flatnonzero(D["W" + side] > 0)
            if len(js) == 0:
                continue
            jq, rq, lnw = self.tail_nodes(R, js, edge, side)
            lq = self.local(R, jq, rq)
            a = (lnw + lq[0]).reshape(len(js), -1)
            pi = np.exp(a - logsumexp(a, 1)[:, None]).ravel()
            G.append(self.per_j(g, R, jq, pi, lq)[js] - R["Gnorm"][js])
            w2.append(D["W2" + side][js])
        return np.vstack(G), np.concatenate(w2)

    # --- evaluation
    def pdf_grid(self, th, pt, r):
        """p(r | pt) for every pt (rows) on the common r grid (columns)."""
        g = Grid(self, np.atleast_1d(pt))
        R = self.ref(th, g)
        j = np.repeat(np.arange(g.M), len(r))
        lu = self.local(R, j, np.tile(r, g.M))[0].reshape(g.M, len(r))
        return np.exp(lu - R["lnorm"][:, None])

    def coefficients(self, th, pt):
        """normalised c(pt) (rows), and mu_ref, sigma_ref."""
        g = Grid(self, np.atleast_1d(pt))
        R = self.ref(th, g)
        return R["Ct"] / np.exp(0.5 * R["lnorm"])[:, None], R["mu"], R["sig"]


def standardised_moments(c, kmax):
    """mean zm, sd zs (in the reference z) and a_k = E[phi_k((z - zm)/zs)], k = 0 ... kmax, of
    N(z) |sum c_n phi_n(z)|^2 (||c|| = 1); exact Gauss-Hermite quadrature."""
    J = len(c) - 1
    t, w = np.polynomial.hermite_e.hermegauss(J + (kmax + 1) // 2 + 2)
    w = w / np.sqrt(2 * np.pi)
    q = np.abs(hermite_basis(t, J) @ c) ** 2
    nrm = w @ q
    zm = w @ (q * t) / nrm
    zs = np.sqrt(w @ (q * t * t) / nrm - zm ** 2)
    return zm, zs, (w * q) @ hermite_basis((t - zm) / zs, kmax) / nrm
