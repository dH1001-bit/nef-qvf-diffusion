"""Tensor-train (TT) Born model of the jet response, all |eta| bins in one fit.  numpy/scipy only.

    p(r | pt, e) = N(r; mu_e(pt), sigma_e(pt)) |sum_n c~_n(u, e) phi_n(z)|^2 / ||c~(u, e)||^2
    z            = (r - mu_e) / sigma_e,   e = |eta| bin (a discrete index: detector boundaries)
    u            = ln pt mapped to [-1, 1] over the common ln pt range of all bins
    mu_e(pt)     = sum_{k <= mu_deg} m_ek P_k(u)                  (one polynomial per |eta| bin)
    sigma_e(pt)  = sqrt(A_e t^2 + B_e^2 t + C_e^2), t = pt_scale / pt (NSC, one per |eta| bin)
    c~_n(u, e)   = sum_k B_e[n, k] P_k(u)
    B_e[n, k]    = sum_{a, b} G1[n, a] G2[a, k, b] G3[b, e]          (the tensor train, bonds chi1, chi2)

G1 (J+1 x chi1): amplitude shapes shared by every pt and |eta|; G2 (chi1 x K+1 x chi2): their pt
dependence; G3 (chi2 x neta): the weights of each |eta| bin.  All complex.  The density is
normalised exactly for any cores, so the fit is plain maximum likelihood.

Unobservable directions (left free; the sandwich uses a pseudo-inverse and counts them):
  bond gauges   G1 -> G1 M1, G2 -> M1^-1 G2 M2, G3 -> M2^-1 G3      2 (chi1^2 + chi2^2) real
  per-bin scale and phase of c~(u, e) (columns of G3)              2 neta real, one complex
                                                                   overlapping with the bond gauge
plus the (mu, sigma) <-> (c_1, c_2) directions of every bin.  canonical() fixes the gauge for
output: G1 orthonormal columns, G2 left-orthonormal, every column of G3 of unit norm with B_e[0,0]
real > 0.

Parameter vector theta: per bin (m_e0 .. m_e,mu_deg, A_e, B_e, C_e), then Re of the cores
(G1, G2, G3 flattened, row-major), then Im of the cores.
"""
import numpy as np
from numpy.polynomial import legendre as npleg
from scipy.special import logsumexp

from response_models import BornModel, _SIG2_MIN


def u_map(x, x_range):
    x0, x1 = x_range
    return 2 * (np.asarray(x, float) - x0) / (x1 - x0) - 1


def reexpand(x_lo, x_hi, xr_from, deg_from, xr_to, deg_to):
    """R with sum_k b_k P_k(u_from(x)) = sum_k (R b)_k P_k(u_to(x)) on [x_lo, x_hi] (exact when
    deg_to >= deg_from: an affine change of variable keeps the polynomial degree; least squares
    on that interval otherwise)."""
    x = 0.5 * (x_lo + x_hi) + 0.5 * (x_hi - x_lo) * np.cos(np.pi * (np.arange(40) + 0.5) / 40)
    Vf = npleg.legvander(u_map(x, xr_from), deg_from)
    Vt = npleg.legvander(u_map(x, xr_to), deg_to)
    return np.linalg.lstsq(Vt, Vf, rcond=None)[0]


def make_data(H, ie, jsel, pt_weighting="neff", pt_power=1.0):
    """binned data of |eta| bin ie, pt bins jsel, from histograms2d.npz (as fit_response_global.py):
    neff weighting rescales pt bin j so that its weight is proportional to n_eff_j^pt_power, with
    total sum_j n_eff_j."""
    R = H["r_edges"]
    rc = 0.5 * (R[1:] + R[:-1])
    Sr, S2r = H["sw"][ie][jsel], H["sw2"][ie][jsel]
    Wtot_raw = Sr.sum(1) + H["Wlo"][ie][jsel] + H["Whi"][ie][jsel]
    W2tot_raw = S2r.sum(1) + H["W2lo"][ie][jsel] + H["W2hi"][ie][jsel]
    if pt_weighting == "neff":
        ne_j = np.where(W2tot_raw > 0, Wtot_raw ** 2 / np.maximum(W2tot_raw, 1e-300), 0.0)
        tgt = ne_j ** pt_power * (ne_j > 0)
        tgt *= ne_j.sum() / max(tgt.sum(), 1e-300)
        s = np.where(Wtot_raw > 0, tgt / np.maximum(Wtot_raw, 1e-300), 0.0)
    else:
        s = np.ones(len(jsel))
    S, S2 = Sr * s[:, None], S2r * (s * s)[:, None]
    nz = S > 0
    jj, kk = np.nonzero(nz)
    Wlo, Whi = H["Wlo"][ie][jsel] * s, H["Whi"][ie][jsel] * s
    W2lo, W2hi = H["W2lo"][ie][jsel] * s * s, H["W2hi"][ie][jsel] * s * s
    LPT = H["lpt_edges"]
    pt = np.where(H["W"][ie][jsel] > 0, H["SP"][ie][jsel] / np.maximum(H["W"][ie][jsel], 1e-300),
                  np.exp(0.5 * (LPT[jsel] + LPT[jsel + 1])))
    Win = S.sum(1)
    return dict(ie=ie, pt=pt, jsel=jsel, j=jj, r=rc[kk], S=S[nz], S2=S2[nz], lo=R[0], hi=R[-1],
                Wlo=Wlo, Whi=Whi, W2lo=W2lo, W2hi=W2hi, Win=Win, Wtot=Win + Wlo + Whi,
                W2tot=S2.sum(1) + W2lo + W2hi)


class TTBornModel:
    def __init__(self, J, K, chi, neta, mu_deg=2, x_range=(np.log(100.0), np.log(1000.0)), pt_scale=100.0):
        self.J, self.K, self.neta, self.mu_deg = int(J), int(K), int(neta), int(mu_deg)
        self.chi1, self.chi2 = int(chi[0]), int(chi[1])
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.pt_scale = float(pt_scale)
        self.nmu = self.mu_deg + 1
        self.nref = self.nmu + 3
        self.n1 = (self.J + 1) * self.chi1
        self.n2 = self.chi1 * (self.K + 1) * self.chi2
        self.n3 = self.chi2 * self.neta
        self.ncore = self.n1 + self.n2 + self.n3
        self.oc = self.neta * self.nref                     # offset of the core block
        self.P = self.oc + 2 * self.ncore
        self._bm = BornModel(self.J, 0)                     # for local() and tail_nodes()

    # --- bookkeeping
    def gauge_dim(self):
        """real dimension of the TT gauge (bond gauges + per-bin scale and phase, one overlap)."""
        return 2 * (self.chi1 ** 2 + self.chi2 ** 2 + self.neta - 1)

    def names(self):
        nm = []
        for e in range(self.neta):
            nm += ["e%d m%d" % (e, k) for k in range(self.nmu)] + ["e%d %s" % (e, s) for s in "ABC"]
        cn = ["G1[%d,%d]" % (n, a) for n in range(self.J + 1) for a in range(self.chi1)]
        cn += ["G2[%d,%d,%d]" % (a, k, b) for a in range(self.chi1) for k in range(self.K + 1) for b in range(self.chi2)]
        cn += ["G3[%d,%d]" % (b, e) for b in range(self.chi2) for e in range(self.neta)]
        return nm + ["Re " + c for c in cn] + ["Im " + c for c in cn]

    def to_dict(self, th):
        return dict(J=self.J, K=self.K, chi=[self.chi1, self.chi2], neta=self.neta, mu_deg=self.mu_deg,
                    x_range=list(self.x_range), pt_scale=self.pt_scale, theta=[float(v) for v in th], names=self.names())

    @classmethod
    def from_dict(cls, d):
        return cls(d["J"], d["K"], d["chi"], d["neta"], d["mu_deg"], d["x_range"], d["pt_scale"]), np.asarray(d["theta"])

    def unpack(self, th):
        ref = th[:self.oc].reshape(self.neta, self.nref)
        z = th[self.oc:self.oc + self.ncore] + 1j * th[self.oc + self.ncore:]
        G1 = z[:self.n1].reshape(self.J + 1, self.chi1)
        G2 = z[self.n1:self.n1 + self.n2].reshape(self.chi1, self.K + 1, self.chi2)
        G3 = z[self.n1 + self.n2:].reshape(self.chi2, self.neta)
        return ref[:, :self.nmu], ref[:, self.nmu:], G1, G2, G3

    def pack(self, m, sp, G1, G2, G3):
        z = np.concatenate([np.asarray(G, complex).ravel() for G in (G1, G2, G3)])
        return np.r_[np.hstack([np.asarray(m, float), np.asarray(sp, float)]).ravel(), z.real, z.imag]

    @staticmethod
    def contract(G1, G2, G3):
        """B (neta, J+1, K+1)."""
        return np.einsum("na,akb,be->enk", G1, G2, G3)

    def canonical(self, th):
        """the same model in the canonical gauge (see the module docstring)."""
        m, sp, G1, G2, G3 = self.unpack(th)
        Q1, R1 = np.linalg.qr(G1)
        G2 = np.einsum("ab,bkc->akc", R1, G2)
        Q2, R2 = np.linalg.qr(G2.reshape(-1, self.chi2))
        G2, G3 = Q2.reshape(self.chi1, self.K + 1, self.chi2), R2 @ G3
        B = self.contract(Q1, G2, G3)
        for e in range(self.neta):
            nrm = np.linalg.norm(G3[:, e])
            ph = np.exp(-1j * np.angle(B[e, 0, 0])) if abs(B[e, 0, 0]) > 0 else 1.0
            G3[:, e] *= ph / max(nrm, 1e-300)
        return self.pack(m, sp, Q1, G2, G3)

    # --- per-bin quantities
    def grid(self, pt):
        u = u_map(np.log(pt), self.x_range)
        t = self.pt_scale / np.asarray(pt, float)
        return dict(pt=np.asarray(pt, float), Lmu=npleg.legvander(u, self.mu_deg), Lc=npleg.legvander(u, self.K), t=t, M=len(pt))

    def ref(self, m_e, sp_e, B_e, g):
        A, Bs, C = sp_e
        s2 = A * g["t"] ** 2 + Bs * Bs * g["t"] + C * C
        d2 = np.stack([g["t"] ** 2, 2 * Bs * g["t"], np.full_like(g["t"], 2 * C)], 1)
        ok = s2 > _SIG2_MIN
        sig = np.sqrt(np.where(ok, s2, _SIG2_MIN))
        Ct = g["Lc"] @ B_e.T                                   # (M, J+1)
        return dict(mu=g["Lmu"] @ m_e, sig=sig, dsig=d2 / (2 * sig[:, None]) * ok[:, None], Ct=Ct,
                    norm=np.sum(np.abs(Ct) ** 2, 1))

    def pieces(self, R, D, grad=True):
        """value and d/d(mu_j, sigma_j, Re Ct_j, Im Ct_j) of the bin's log-likelihood (data + tails)."""
        M, J1 = len(R["mu"]), self.J + 1
        loc = self._bm.local(R, D["j"], D["r"])
        val = D["S"] @ loc[0] - D["Wtot"] @ np.log(R["norm"])
        if grad:
            gmu, gsig = np.zeros(M), np.zeros(M)
            gRe, gIm = np.zeros((M, J1)), np.zeros((M, J1))

            def add(j, w, lc):
                _, dmu, dsg, Rp, Ip = lc
                gmu[:] += np.bincount(j, w * dmu, M)
                gsig[:] += np.bincount(j, w * dsg, M)
                for n in range(J1):
                    gRe[:, n] += np.bincount(j, w * Rp[:, n], M)
                    gIm[:, n] += np.bincount(j, w * Ip[:, n], M)
            add(D["j"], D["S"], loc)
            gRe -= D["Wtot"][:, None] * 2 * R["Ct"].real / R["norm"][:, None]
            gIm -= D["Wtot"][:, None] * 2 * R["Ct"].imag / R["norm"][:, None]
        for side, edge in (("lo", D["lo"]), ("hi", D["hi"])):
            Wt = D["W" + side]
            js = np.flatnonzero(Wt > 0)
            if len(js) == 0:
                continue
            jq, rq, lnw = self._bm.tail_nodes(R, js, edge, side)
            lq = self._bm.local(R, jq, rq)
            a = (lnw + lq[0]).reshape(len(js), -1)
            lse = logsumexp(a, 1)
            val += Wt[js] @ lse
            if grad:
                pi = np.exp(a - lse[:, None]).ravel()
                add(jq, pi * Wt[jq], lq)
        return (val, gmu, gsig, gRe, gIm) if grad else (val,)

    def lnl(self, th, Ds, grad=True):
        """total log-likelihood over the |eta| bins Ds (list of make_data dicts with a "g" grid),
        its gradient, and the per-bin values."""
        m, sp, G1, G2, G3 = self.unpack(th)
        B = self.contract(G1, G2, G3)
        val, per, gr = 0.0, [], np.zeros(self.P) if grad else None
        Gam = np.zeros_like(B)
        for e, D in enumerate(Ds):
            R = self.ref(m[e], sp[e], B[e], D["g"])
            out = self.pieces(R, D, grad)
            val += out[0]; per.append(out[0])
            if grad:
                _, gmu, gsig, gRe, gIm = out
                o = e * self.nref
                gr[o:o + self.nmu] = D["g"]["Lmu"].T @ gmu
                gr[o + self.nmu:o + self.nref] = R["dsig"].T @ gsig
                Gam[e] = (gRe + 1j * gIm).T @ D["g"]["Lc"]            # dl/dRe B + i dl/dIm B
        if grad:
            W = np.einsum("akb,be->eak", G2, G3)
            gG1 = np.einsum("enk,eak->na", Gam, W.conj())
            GW = np.einsum("na,enk->eak", G1.conj(), Gam)
            gG2 = np.einsum("eak,be->akb", GW, G3.conj())
            gG3 = np.einsum("akb,eak->be", G2.conj(), GW)
            z = np.concatenate([gG1.ravel(), gG2.ravel(), gG3.ravel()])
            gr[self.oc:self.oc + self.ncore] = z.real
            gr[self.oc + self.ncore:] = z.imag
        return val, gr, per

    # --- scores for the sandwich and the Fisher-scoring polish
    def _core_jac(self, G1, G2, G3, e, Lc):
        """d Ct_n(j) / d z for every complex core parameter z: (M, J+1, ncore), holomorphic."""
        M, J1 = Lc.shape[0], self.J + 1
        Jc = np.zeros((M, J1, self.ncore), complex)
        W = np.einsum("akb,b->ak", G2, G3[:, e])
        V = Lc @ W.T                                                   # (M, chi1)
        for n in range(J1):
            Jc[:, n, n * self.chi1:(n + 1) * self.chi1] = V
        Jc[:, :, self.n1:self.n1 + self.n2] = np.einsum("na,jk,b->jnakb", G1, Lc, G3[:, e]).reshape(M, J1, -1)
        U = np.einsum("na,jk,akb->jnb", G1, Lc, G2)                    # (M, J+1, chi2)
        cols = self.n1 + self.n2 + np.arange(self.chi2) * self.neta + e
        Jc[:, :, cols] = U
        return Jc

    def score_outer(self, th, Ds):
        """V1 = sum w g g^T and V2 = sum w^2 g g^T over all observation groups (fine 2D bins and the
        two tail masses of every pt bin), g the gradient of ln p of the group; and sum w g (= the
        gradient, a check).  V1 is the Fisher-scoring (BHHH) curvature, V2 the sandwich meat."""
        m, sp, G1, G2, G3 = self.unpack(th)
        B = self.contract(G1, G2, G3)
        V1, V2, gsum = np.zeros((self.P, self.P)), np.zeros((self.P, self.P)), np.zeros(self.P)
        core = np.r_[self.oc:self.P]
        for e, D in enumerate(Ds):
            g = D["g"]
            R = self.ref(m[e], sp[e], B[e], g)
            Jc = self._core_jac(G1, G2, G3, e, g["Lc"])
            qn = 2 * R["Ct"] / R["norm"][:, None]                      # d ln norm / d (Re + i Im) Ct
            idx = np.r_[e * self.nref:(e + 1) * self.nref, core]
            nl = len(idx)
            A1, A2, gs = np.zeros((nl, nl)), np.zeros((nl, nl)), np.zeros(nl)

            def rows(js, dmu, dsg, q):
                """score rows of groups at pt indices js with summed pieces dmu, dsg, q (complex)."""
                s = np.empty((len(js), nl))
                s[:, :self.nmu] = dmu[:, None] * g["Lmu"][js]
                s[:, self.nmu:self.nref] = dsg[:, None] * R["dsig"][js]
                sc = np.einsum("pn,pnc->pc", (q - qn[js]).conj(), Jc[js])
                s[:, self.nref:self.nref + self.ncore] = sc.real
                s[:, self.nref + self.ncore:] = -sc.imag
                return s

            loc = self._bm.local(R, D["j"], D["r"])
            _, dmu, dsg, Rp, Ip = loc
            order = np.argsort(D["j"], kind="stable")
            bounds = np.searchsorted(D["j"][order], np.arange(g["M"] + 1))
            for jj in range(g["M"]):
                sel = order[bounds[jj]:bounds[jj + 1]]
                if len(sel) == 0:
                    continue
                s = rows(np.full(len(sel), jj), dmu[sel], dsg[sel], Rp[sel] + 1j * Ip[sel])
                A1 += (s * D["S"][sel, None]).T @ s
                A2 += (s * D["S2"][sel, None]).T @ s
                gs += D["S"][sel] @ s
            for side, edge in (("lo", D["lo"]), ("hi", D["hi"])):
                js = np.flatnonzero(D["W" + side] > 0)
                if len(js) == 0:
                    continue
                jq, rq, lnw = self._bm.tail_nodes(R, js, edge, side)
                _, tmu, tsg, tRp, tIp = lq = self._bm.local(R, jq, rq)
                a = (lnw + lq[0]).reshape(len(js), -1)
                pi = np.exp(a - logsumexp(a, 1)[:, None])
                nq = a.shape[1]
                sm = lambda v: np.einsum("jq,jq...->j...", pi, v.reshape(len(js), nq, *v.shape[1:]))
                s = rows(js, sm(tmu), sm(tsg), sm(tRp + 1j * tIp))
                w1, w2 = D["W" + side][js], D["W2" + side][js]
                A1 += (s * w1[:, None]).T @ s
                A2 += (s * w2[:, None]).T @ s
                gs += w1 @ s
            V1[np.ix_(idx, idx)] += A1
            V2[np.ix_(idx, idx)] += A2
            gsum[idx] += gs
        return V1, V2, gsum

    # --- evaluation
    def born_theta(self, th, e, x_range_e):
        """the model of |eta| bin e as a response_models.BornModel theta on its own ln pt range
        (b_00 = 1: the per-bin scale and phase are unobservable)."""
        m, sp, G1, G2, G3 = self.unpack(th)
        B = np.einsum("na,akb,b->nk", G1, G2, G3[:, e])
        Rmu = reexpand(*x_range_e, self.x_range, self.mu_deg, x_range_e, self.mu_deg)
        Rc = reexpand(*x_range_e, self.x_range, self.K, x_range_e, self.K)
        bm = BornModel(self.J, self.K, self.mu_deg, "nsc", x_range_e, self.pt_scale)
        return bm, bm.pack(Rmu @ m[e], sp[e], B @ Rc.T)
