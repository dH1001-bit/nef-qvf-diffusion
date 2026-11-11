#!/usr/bin/env python3
"""Tensor-train (TT) amplitude fit of the jet response: all |eta| bins in ONE likelihood, the
amplitude coefficients shared through a tensor train (model: response_tt.TTBornModel)

    c~_n(u, e) = sum_k B_e[n, k] P_k(u),   B_e[n, k] = sum_{a,b} G1[n, a] G2[a, k, b] G3[b, e]

with mu_e(pt) (Legendre in u) and the NSC sigma_e(pt) separate for every |eta| bin e.

    python3 scripts/fit_response_tt.py --load fig/global_receta/histograms2d.npz \
        --init fig/global_K3/global_fits.json --K 3 --chi 4 4

Workflow
1. Data: histograms2d.npz of fit_response_global.py (read only), the same binned likelihood with
   tail terms and the same pt weighting (--pt-weighting, --pt-power) as there.
2. Start (--init, a global_fits.json of fit_response_global.py, read only): every |eta| bin's
   mu polynomial and coefficient matrix re-expressed exactly on the common ln pt range (an affine
   change of variable keeps the polynomial degree), per-bin scale and phase normalised, then a
   TT-SVD truncated to --chi gives the starting cores.  NSC parameters copied.
3. Fit: L-BFGS-B with the analytic gradient, then a Fisher-scoring (BHHH) polish: step
   V1^+ grad with V1 = sum w g g^T over the fine bins, backtracking line search, stop when the
   step's predicted gain is below --polish-tol (in units of n_eff * lnL).  The gauge directions
   are flat; the pseudo-inverse leaves them alone.
4. The result is brought to the canonical gauge (response_tt.TTBornModel.canonical).
5. Errors: sandwich H^+ V2 H^+ with V2 = sum w^2 g g^T and H the numerical Hessian (--hessian
   numeric, 2 P gradient evaluations) or -V1 (--hessian bhhh, cheap).
6. Outputs in --outdir:
     summary.txt        per |eta| bin n_eff*lnL of the TT fit and of the --init fits on the same
                        data, totals, parameter counts, AIC_eff, bond singular values
     tt_fit.json        the TT model (TTBornModel.to_dict), covariance, lnL per bin, settings
     global_fits.json   every |eta| bin exported as a response_models.BornModel on its own ln pt
                        range (b_00 = 1), with its covariance (Jacobian of the export): the format
                        of fit_response_global.py, so compare_global_cells.py, tt_svd_diagnostic.py
                        and anything else that reads global_fits.json work unchanged
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from response_models import BornModel, Grid
from response_tt import TTBornModel, make_data, reexpand

ap = argparse.ArgumentParser()
ap.add_argument("--load", required=True, help="histograms2d.npz of fit_response_global.py")
ap.add_argument("--init", required=True, help="global_fits.json of fit_response_global.py (all |eta| bins)")
ap.add_argument("--K", type=int, default=3, help="Legendre degree in ln pt of the TT")
ap.add_argument("--chi", type=int, nargs=2, default=(4, 4), metavar=("CHI1", "CHI2"),
                help="bond dimensions: shapes (Hermite | rest) and |eta| modes (rest | |eta|)")
ap.add_argument("--mu-degree", type=int, default=None, help="default: that of the --init fits")
ap.add_argument("--pt-weighting", default="neff", choices=("neff", "xsec"))
ap.add_argument("--pt-power", type=float, default=1.0, help="as in fit_response_global.py")
ap.add_argument("--maxiter", type=int, default=5000)
ap.add_argument("--ftol", type=float, default=1e-12, help="L-BFGS-B relative tolerance on lnL/sum w")
ap.add_argument("--gtol", type=float, default=1e-8, help="L-BFGS-B projected-gradient tolerance")
ap.add_argument("--polish-iter", type=int, default=30, help="Fisher-scoring steps after L-BFGS-B (0: off)")
ap.add_argument("--polish-tol", type=float, default=0.01, help="stop when the predicted gain < this (n_eff*lnL)")
ap.add_argument("--hessian", default="numeric", choices=("numeric", "bhhh"), help="H of the sandwich")
ap.add_argument("--seed", type=int, default=1, help="for the padding when --chi exceeds the starting rank")
ap.add_argument("--outdir", default=None, help="default: fig/tt_<var>_K<K>_chi<a>x<b>")
a = ap.parse_args()
t_start = time.time()

# ------------------------------------------------------------------ 1. data
with np.load(a.load) as z:
    H = {k: z[k] for k in z.files}
var = str(H["var"])
ETA = H["eta_edges"]
neta = len(ETA) - 1
outdir = a.outdir or os.path.join("fig", "tt_%s_K%d_chi%dx%d" % (var, a.K, *a.chi))
os.makedirs(outdir, exist_ok=True)

Ds, xr_e, ne_e, W_e = [], [], [], []
for e in range(neta):
    jsel = np.flatnonzero(H["W"][e] > 0)
    D = make_data(H, e, jsel, a.pt_weighting, a.pt_power)
    Ds.append(D)
    xr_e.append((float(np.log(D["pt"].min())), float(np.log(D["pt"].max()))))
    W_e.append(D["Wtot"].sum())
    ne_e.append(W_e[-1] ** 2 / D["W2tot"].sum())
x_range = (min(x[0] for x in xr_e), max(x[1] for x in xr_e))
Wsum = sum(W_e)
print("data: %d |%s| bins, common ln pt range %.3f - %.3f (%.0f - %.0f GeV), %d filled 2D bins"
      % (neta, var, *x_range, *np.exp(x_range), sum(len(D["S"]) for D in Ds)), flush=True)

# ------------------------------------------------------------------ 2. start from the --init fits
with open(a.init) as fh:
    init = {f["eta_bin"]: f for f in json.load(fh)["fits"]}
missing = [e for e in range(neta) if e not in init]
if missing:
    raise SystemExit("--init has no fit for |eta| bins %s" % missing)
init_models = [BornModel.from_dict(init[e]["model"]) for e in range(neta)]
J = init_models[0][0].J
mu_deg = a.mu_degree if a.mu_degree is not None else max(bm.mu_deg for bm, _ in init_models)
if not (1 <= a.chi[0] <= J + 1 and 1 <= a.chi[1] <= a.chi[0] * (a.K + 1)):
    raise SystemExit("--chi: need 1 <= chi1 <= J+1 = %d and 1 <= chi2 <= chi1 (K+1)" % (J + 1))
mod = TTBornModel(J, a.K, a.chi, neta, mu_deg, x_range)
for e, D in enumerate(Ds):
    D["g"] = mod.grid(D["pt"])

m0, sp0, T = np.zeros((neta, mu_deg + 1)), np.zeros((neta, 3)), np.zeros((neta, J + 1, a.K + 1), complex)
lnl_init = []
for e, (bm, th) in enumerate(init_models):
    m, sp, B = bm.unpack(th)
    xb = (bm.x0, bm.x1)
    m0[e] = reexpand(*xr_e[e], xb, bm.mu_deg, x_range, mu_deg) @ m
    sp0[e] = sp if bm.sigma_form == "nsc" else (0.0, 0.0, sp[0])
    Bc = B @ reexpand(*xr_e[e], xb, bm.K, x_range, a.K).T
    T[e] = Bc * np.exp(-1j * np.angle(Bc[0, 0])) / np.linalg.norm(Bc)
    # the --init model on exactly this data, for the comparison in summary.txt
    R = bm.ref(th, Grid(bm, Ds[e]["pt"]))
    R["norm"] = np.exp(R["lnorm"])
    lnl_init.append(mod.pieces(R, Ds[e], grad=False)[0])
P_init = sum(bm.P for bm, _ in init_models)

rng = np.random.default_rng(a.seed)


def truncate(Mx, chi):
    """left factor (orthonormal columns) and right factor of a rank-chi SVD; padded with small
    random directions when chi exceeds the available rank (a zero column would be a saddle)."""
    U, s, Vh = np.linalg.svd(Mx, full_matrices=False)
    err = np.sqrt(max(1 - np.sum(s[:chi] ** 2) / np.sum(s ** 2), 0))
    if chi > len(s):
        r, n = Mx.shape
        Up = rng.normal(size=(r, chi - len(s))) + 1j * rng.normal(size=(r, chi - len(s)))
        U = np.linalg.qr(np.hstack([U, Up]))[0][:, :chi] if chi <= r else np.hstack([U, Up / np.linalg.norm(Up, axis=0)])
        Vp = 1e-3 * s[0] * (rng.normal(size=(chi - len(s), n)) + 1j * rng.normal(size=(chi - len(s), n)))
        return U, np.vstack([s[:, None] * Vh, Vp]), err
    return U[:, :chi], s[:chi, None] * Vh[:chi], err


G1, rest, e1 = truncate(T.transpose(1, 2, 0).reshape(J + 1, -1), a.chi[0])
c1 = G1.shape[1]
G2f, G3, e2 = truncate(rest.reshape(c1 * (a.K + 1), neta), a.chi[1])
G2 = G2f.reshape(c1, a.K + 1, -1)
th = mod.pack(m0, sp0, G1, G2, G3)
v0 = mod.lnl(th, Ds, grad=False)[0]
print("start: TT-SVD truncation error (relative, of the normalised coefficient tensor): bond 1 %.4f, bond 2 %.4f"
      % (e1, e2))
print("       lnL/sum w: --init fits %.7f, truncated TT %.7f;  P: --init %d, TT %d (of which %d gauge)"
      % (sum(lnl_init) / Wsum, v0 / Wsum, P_init, mod.P, mod.gauge_dim()), flush=True)

# ------------------------------------------------------------------ 3. fit
nit = [0]


def f(t):
    v, g, _ = mod.lnl(t, Ds)
    nit[0] += 1
    return -v / Wsum, -g / Wsum


t0 = time.time()
res = minimize(f, th, jac=True, method="L-BFGS-B",
               options=dict(maxiter=a.maxiter, maxfun=2 * a.maxiter, ftol=a.ftol, gtol=a.gtol))
th, conv = res.x, bool(res.success)
v = mod.lnl(th, Ds, grad=False)[0]
print("L-BFGS-B: lnL/sum w %.7f, %s (%s), %d iterations, %.0f s"
      % (v / Wsum, "converged" if conv else "NOT converged", res.message, res.nit, time.time() - t0), flush=True)

t0 = time.time()
for it in range(1, a.polish_iter + 1):
    v, g, _ = mod.lnl(th, Ds)
    V1, _, gs = mod.score_outer(th, Ds)
    if it == 1:
        print("  polish: score check |sum w g - grad| / |grad| = %.2e" % (np.linalg.norm(gs - g) / max(np.linalg.norm(g), 1e-300)))
    step = np.linalg.pinv(V1, rcond=1e-10, hermitian=True) @ g
    gain = float(g @ step)                                  # predicted gain in lnL (sum-w units)
    if gain < a.polish_tol:                                 # with neff weighting lnL is n_eff * lnL
        conv = True
        print("  polish: converged after %d step(s), predicted gain %.3g" % (it - 1, gain))
        break
    tl, ok = 1.0, False
    for _ in range(30):
        vn = mod.lnl(th + tl * step, Ds, grad=False)[0]
        if vn > v:
            ok = True
            break
        tl *= 0.5
    print("  polish %2d: predicted gain %.3g, step %.3g, gain %.3g, %.0f s" % (it, gain, tl, vn - v if ok else 0.0,
                                                                              time.time() - t0), flush=True)
    if not ok:
        break
    th = th + tl * step
else:
    if a.polish_iter > 0:
        conv = False

# ------------------------------------------------------------------ 4. canonical gauge
v_before = mod.lnl(th, Ds, grad=False)[0]
th = mod.canonical(th)
lnl_tot, _, lnl_e = mod.lnl(th, Ds)
print("canonical gauge: lnL change %.3g (should be ~0)" % (lnl_tot - v_before))

# ------------------------------------------------------------------ 5. covariance
t0 = time.time()
V1, V2, _ = mod.score_outer(th, Ds)
if a.hessian == "numeric":
    Hm = np.zeros((mod.P, mod.P))
    for i in range(mod.P):
        h = 1e-5 * max(1.0, abs(th[i]))
        ev = np.zeros(mod.P); ev[i] = h
        Hm[i] = -(mod.lnl(th + ev, Ds)[1] - mod.lnl(th - ev, Ds)[1]) / (2 * h)
    Hm = 0.5 * (Hm + Hm.T)
else:
    Hm = V1
ev_ = np.linalg.eigvalsh(Hm)
nflat = int(np.sum(np.abs(ev_) < 1e-9 * np.abs(ev_).max()))
Hi = np.linalg.pinv(Hm, rcond=1e-9, hermitian=True)
cov = Hi @ V2 @ Hi
print("covariance (%s Hessian): %d flat direction(s), of which %d expected from the TT gauge; %.0f s"
      % (a.hessian, nflat, mod.gauge_dim(), time.time() - t0), flush=True)

# ------------------------------------------------------------------ 6. outputs
_, _, G1, G2, G3 = mod.unpack(th)
B = mod.contract(G1, G2, G3)
Bn = B / np.linalg.norm(B, axis=(1, 2), keepdims=True)
sv1 = np.linalg.svd(Bn.transpose(1, 2, 0).reshape(J + 1, -1), compute_uv=False)
sv2 = np.linalg.svd(Bn.transpose(1, 2, 0).reshape(-1, neta), compute_uv=False)

fits = []
for e in range(neta):
    bm, the = mod.born_theta(th, e, xr_e[e])
    Jx = np.zeros((bm.P, mod.P))
    for i in range(mod.P):
        h = 1e-6 * max(1.0, abs(th[i]))
        ev = np.zeros(mod.P); ev[i] = h
        Jx[:, i] = (mod.born_theta(th + ev, e, xr_e[e])[1] - mod.born_theta(th - ev, e, xr_e[e])[1]) / (2 * h)
    fits.append(dict(eta_bin=e, abs_eta=[float(ETA[e]), float(ETA[e + 1])], n_eff=float(ne_e[e]), sum_w=float(W_e[e]),
                     pt_range=[float(Ds[e]["pt"].min()), float(Ds[e]["pt"].max())], model=bm.to_dict(the),
                     cov=(Jx @ cov @ Jx.T).tolist(), flat_directions=nflat, lnL=float(lnl_e[e]), converged=conv,
                     stages=[dict(K=a.K, P=mod.P, lnL=float(lnl_e[e]), converged=conv, model=bm.to_dict(the))],
                     cells=[], tt=dict(chi=list(a.chi), note="exported from the joint TT fit (tt_fit.json)")))
with open(os.path.join(outdir, "global_fits.json"), "w") as fh:
    json.dump(dict(var=var, eta_edges=[float(x) for x in ETA], J=J, fits=fits), fh)
with open(os.path.join(outdir, "tt_fit.json"), "w") as fh:
    json.dump(dict(var=var, eta_edges=[float(x) for x in ETA], model=mod.to_dict(th), cov=cov.tolist(),
                   lnL=float(lnl_tot), lnL_per_bin=[float(x) for x in lnl_e], converged=conv, flat_directions=nflat,
                   gauge_dim=mod.gauge_dim(), bond_singular_values=[sv1.tolist(), sv2.tolist()],
                   settings=vars(a), x_range_per_bin=xr_e, init_lnL_per_bin=[float(x) for x in lnl_init],
                   init_P=P_init), fh)

sc = lambda e, l: ne_e[e] * l / W_e[e]                 # n_eff * lnL / sum w per bin
L = ["# TT amplitude fit: J = %d, K = %d, chi = %d x %d, mu degree %d, pt weighting %s (power %g)"
     % (J, a.K, *a.chi, mu_deg, a.pt_weighting, a.pt_power),
     "# data %s; start and comparison: %s" % (os.path.abspath(a.load), os.path.abspath(a.init)),
     "# n_eff*lnL per |eta| bin as in fit_response_global.py; init = the --init fits evaluated here",
     "%-18s %14s %14s %10s" % ("|%s|" % var, "init n_eff*lnL", "TT n_eff*lnL", "TT - init")]
for e in range(neta):
    L.append("%4.2f<|%s|<%-4.2f %16.1f %14.1f %10.1f" % (ETA[e], var, ETA[e + 1], sc(e, lnl_init[e]), sc(e, lnl_e[e]),
                                                       sc(e, lnl_e[e]) - sc(e, lnl_init[e])))
tot_i = sum(sc(e, lnl_init[e]) for e in range(neta))
tot_t = sum(sc(e, lnl_e[e]) for e in range(neta))
P_eff = mod.P - mod.gauge_dim()
L += ["%-18s %16.1f %14.1f %10.1f" % ("total", tot_i, tot_t, tot_t - tot_i),
      "# parameters: init %d (7 independent fits), TT %d (%d without the gauge directions)" % (P_init, mod.P, P_eff),
      "# AIC_eff = 2 P - 2 n_eff*lnL:  init %.1f,  TT %.1f (P without gauge)" % (2 * P_init - 2 * tot_i, 2 * P_eff - 2 * tot_t),
      "# fit: %s; %d flat directions (TT gauge %d); covariance from the %s Hessian"
      % ("converged" if conv else "NOT converged", nflat, mod.gauge_dim(), a.hessian),
      "# bond 1 (Hermite | pt, eta) singular values / s1: " + " ".join("%.2e" % (s / sv1[0]) for s in sv1),
      "# bond 2 (Hermite, pt | eta) singular values / s1: " + " ".join("%.2e" % (s / sv2[0]) for s in sv2),
      "# total time %.0f s" % (time.time() - t_start)]
print("\n" + "\n".join(L))
with open(os.path.join(outdir, "summary.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("\nsummary.txt, tt_fit.json, global_fits.json in", outdir)
print("next: python3 scripts/compare_global_cells.py --global %s --hist %s --amp-fits ... --dscb-fits ..."
      % (os.path.join(outdir, "global_fits.json"), a.load))
