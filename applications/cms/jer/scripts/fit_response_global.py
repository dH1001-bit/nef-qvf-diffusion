#!/usr/bin/env python3
"""Global (pt-smooth) amplitude fit of the jet response, one fit per |eta| bin over the whole
pt_gen range.  The model (response_models.BornModel):

    p(r | pt) = N(r; mu(pt), sigma(pt)) |sum_n c_n(pt) phi_n(z)|^2,      z = (r - mu)/sigma
    mu(pt)    = quadratic in ln pt (Legendre in u, u = ln pt mapped to [-1, 1] over the fit range)
    sigma(pt) = sqrt(A t^2 + B^2 t + C^2), t = 100 GeV / pt     (NSC form of the reference width)
    c_n(pt)   = sum_{k<=K} b_nk P_k(u) / norm,   complex b_nk,  b_00 = 1

    python3 scripts/fit_response_global.py                          # read the export, fit all |eta| bins
    python3 scripts/fit_response_global.py --load fig/global_receta/histograms2d.npz --K 2 2 2 2 1 1 1
    python3 scripts/fit_response_global.py --load ... --eta-only 0  # one |eta| bin

Workflow
1. Accumulation (one pass over the export, or --load): per fine |eta| bin a 2D histogram in
   (ln pt_gen: --pt-bins bins over --pt-range, r: --rbins bins over --range) of sum w and sum w^2,
   and per (|eta|, pt) bin sum w, sum w^2, sum w pt_gen and the weight (and w^2) below and above
   the r range.  Saved to histograms2d.npz.  Bins of 2 % in pt and 0.002 in r are much finer than
   any scale of the model: the binned likelihood is the unbinned one for all practical purposes.
2. Likelihood per |eta| bin, every jet at its fine bin centre (r_k, <pt>_j):
       l = sum_jk S_jk ln p(r_k | pt_j) + sum_j [Wlo_j ln P(r < lo | pt_j) + Whi_j ln P(r >= hi | pt_j)]
   The tail terms keep the jets outside the r range in the fit: no truncation bias.
3. Fit: L-BFGS with the analytic gradient, warm-started K = 0 -> 1 -> ... -> --K.  Starting
   values: mu from a quadratic fit to the per-pt-bin means, NSC from a fit to the per-pt-bin
   sigma_68^2, c = e_0 (+ 0.01 i in c_1 ... c_J: at real c the gradient in Im c vanishes).
4. Errors: sandwich covariance H^+ V H^+ (H numerical Hessian of l, V = sum S2 g g^T over the
   fine bins, S2 = sum w^2: correct for weighted events), pseudo-inverse because of the flat
   directions (mu, sigma) <-> (c_1, c_2).  Single parameters along those directions are not
   meaningful; the gauge-invariant curves (quantiles, widths) get bands by the delta method.
5. Benchmark: independent fits of the same likelihood in every cell of --cell-pt-edges (constant
   mu, sigma and c within a cell, K = 0).  The global fit's deficit in n_eff * lnL shows how
   restrictive the smoothness in pt is; AIC_eff = 2 P - 2 n_eff * lnL/sum w compares the models.
6. Outputs in --outdir:
     summary.txt               lnL, Delta lnL, AIC per |eta| bin and model; reference parameters
     global_fits.json          every model (BornModel.to_dict), covariance, lnL; reload with
                               response_models.BornModel.from_dict to evaluate p(r | pt)
     curves_eta<i>.png         median, sigma_68/median, in-range mean and standardised quantiles
                               (Q_p - Q_50)/sigma_68 vs ln pt: model curve with 1 sigma band, model
                               mixture over each data group (open squares), data (points), cell fits
     coefficients_eta<i>.png   c_n(pt) (display phase: c_0 real) and moments a_k(pt)
     cells/cell_eta<i>_pt<j>.png  data vs model in every cell of --cell-pt-edges, linear and log,
                               relative residuals below.  The model is the mixture
                               sum_j W_j p(r | pt_j) over the cell's pt bins, not p at <pt>.
"""
import argparse, glob, json, os, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from response_models import BornModel, Grid, standardised_moments

ap = argparse.ArgumentParser()
ap.add_argument("--files", default="/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz")
ap.add_argument("--max-files", type=int, default=None, help="only the first N files (a quick look; biased spectrum)")
ap.add_argument("--load", default=None, help="histograms2d.npz of an earlier run: refit without reading the export")
ap.add_argument("--var", default="receta", choices=("receta", "recy", "geneta", "geny"))
ap.add_argument("--max-dr", type=float, default=0.4, help="matching radius, at most the export's 0.4")
ap.add_argument("--eta-edges", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0])
ap.add_argument("--pt-range", nargs=2, type=float, default=(97.0, 6500.0), metavar=("LO", "HI"),
                help="pt_gen range of the 2D histograms [GeV]")
ap.add_argument("--pt-bins", type=int, default=200, help="bins in ln pt_gen")
ap.add_argument("--range", nargs=2, type=float, default=(0.0, 2.0), metavar=("LO", "HI"), help="response range")
ap.add_argument("--rbins", type=int, default=1000, help="fine response bins")
ap.add_argument("--fit-pt-range", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                help="fit only pt bins inside (default: the whole --pt-range)")
ap.add_argument("--pt-weighting", default="neff", choices=("neff", "xsec"),
                help="share of each pt bin in the likelihood: its n_eff (default) or its cross section "
                     "(xsec: the fit is dominated by the lowest pt bins)")
ap.add_argument("--amp-degree", type=int, default=8, help="highest Hermite degree J")
ap.add_argument("--K", nargs="+", type=int, default=[2],
                help="highest Legendre order of c_n(ln pt): one value, or one per |eta| bin")
ap.add_argument("--mu-degree", type=int, default=2, help="order of mu_ref(ln pt)")
ap.add_argument("--ref-penalty", type=float, default=0.0,
                help="lambda * sum_k |b_1k|^2 + |b_2k|^2 per unit weight: removes the flat directions "
                     "(mu, sigma) <-> (c_1, c_2); 0 = off")
ap.add_argument("--maxiter", type=int, default=3000)
ap.add_argument("--eta-only", type=int, nargs="+", default=None, help="fit only these |eta| bin indices")
ap.add_argument("--cell-pt-edges", nargs="+", type=float,
                default=[97, 140, 200, 280, 400, 550, 750, 1000, 1400, 2000, 3500, 6500],
                help="cells for the benchmark fits and the cell plots")
ap.add_argument("--no-cells", action="store_true", help="skip the per-cell benchmark fits")
ap.add_argument("--cell-plots", action=argparse.BooleanOptionalAction, default=True)
ap.add_argument("--group-neff", type=float, default=5000.0,
                help="data points of the curve plots: consecutive pt bins merged to this n_eff")
ap.add_argument("--bins", type=int, default=100, help="response bins in the cell plots")
ap.add_argument("--kmax", type=int, default=6, help="highest moment a_k drawn")
ap.add_argument("--res-max", type=float, default=0.5, help="y range +-res-max of the residual panels")
ap.add_argument("--outdir", default=None, help="default: fig/global_<var>")
a = ap.parse_args()

outdir = a.outdir or os.path.join("fig", "global_%s" % a.var)
os.makedirs(outdir, exist_ok=True)

# ------------------------------------------------------------------ 1. accumulation
if a.load:
    with np.load(a.load) as z:
        H = {k: z[k] for k in z.files}
    var = str(H["var"])
    print("loaded", a.load)
else:
    var = a.var
    ETA = np.asarray(a.eta_edges)
    LPT = np.linspace(np.log(a.pt_range[0]), np.log(a.pt_range[1]), a.pt_bins + 1)
    R = np.linspace(*a.range, a.rbins + 1)
    neta, M, nr = len(ETA) - 1, a.pt_bins, a.rbins
    files = sorted(glob.glob(a.files))[: a.max_files] if a.max_files else sorted(glob.glob(a.files))
    if not files:
        raise SystemExit("no files match " + a.files)
    nb = neta * M
    sw, sw2 = np.zeros(nb * nr), np.zeros(nb * nr)
    acc = {k: np.zeros(nb) for k in ("n", "W", "W2", "SP", "Wlo", "Whi", "W2lo", "W2hi")}
    t0 = time.time()
    for i, f in enumerate(files):
        with np.load(f) as z:
            ip = np.digitize(np.log(z["genpt"]), LPT) - 1
            ie = np.digitize(np.abs(z[var]), ETA) - 1
            sel = (ip >= 0) & (ip < M) & (ie >= 0) & (ie < neta)
            if a.max_dr < 0.4:
                sel &= z["dR"] < a.max_dr
            idx = np.flatnonzero(sel)
            b = ie[idx] * M + ip[idx]
            pt = z["genpt"][idx].astype(np.float64)
            r = z["recpt"][idx].astype(np.float64) / pt
            w = z["weight"][idx].astype(np.float64)
        ir = np.digitize(r, R) - 1
        lo, hi = ir < 0, ir >= nr
        ins = ~(lo | hi)
        bc = lambda k, wt: np.bincount(k, wt, nb)
        acc["n"] += np.bincount(b, minlength=nb); acc["W"] += bc(b, w); acc["W2"] += bc(b, w * w)
        acc["SP"] += bc(b, w * pt)
        acc["Wlo"] += bc(b[lo], w[lo]); acc["W2lo"] += bc(b[lo], w[lo] ** 2)
        acc["Whi"] += bc(b[hi], w[hi]); acc["W2hi"] += bc(b[hi], w[hi] ** 2)
        k = b[ins] * nr + ir[ins]
        sw += np.bincount(k, w[ins], nb * nr); sw2 += np.bincount(k, w[ins] ** 2, nb * nr)
        print("  %2d/%d %-32s %9d in range  %4.0f s" % (i + 1, len(files), os.path.basename(f), len(idx),
                                                      time.time() - t0), flush=True)
    H = dict(var=var, eta_edges=ETA, lpt_edges=LPT, r_edges=R, sw=sw.reshape(neta, M, nr),
             sw2=sw2.reshape(neta, M, nr), **{k: v.reshape(neta, M) for k, v in acc.items()})
    np.savez(os.path.join(outdir, "histograms2d.npz"), **H)
    print("histograms:", os.path.join(outdir, "histograms2d.npz"))

ETA, LPT, R = H["eta_edges"], H["lpt_edges"], H["r_edges"]
neta, M, nr = len(ETA) - 1, len(LPT) - 1, len(R) - 1
rc = 0.5 * (R[1:] + R[:-1])
dr = R[1] - R[0]
J = a.amp_degree
Ks = a.K * neta if len(a.K) == 1 else a.K
if len(Ks) != neta:
    raise SystemExit("--K: one value or %d values" % neta)
eta_list = a.eta_only if a.eta_only is not None else list(range(neta))
CELL = np.asarray(a.cell_pt_edges)
PQ = np.array([0.001, 0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99, 0.999])
ZG = np.array([-3.0902, -2.3263, -1.6449, -0.9945, 0.0, 0.9945, 1.6449, 2.3263, 3.0902]) / 0.9945


def neff(w, w2):
    return w * w / w2 if w2 > 0 else 0.0


def make_data(ie, jsel):
    """the binned data of |eta| bin ie, pt bins jsel (indices into the full pt axis).
    --pt-weighting neff: every pt bin j is rescaled by s_j = sum w / sum w^2, so that its weights
    sum to its n_eff.  The relative weights inside a pt bin (which combine the H_T slices) are
    unchanged; only the share of each pt bin in the likelihood changes, which the conditional
    model p(r | pt) does not depend on."""
    Sr, S2r = H["sw"][ie][jsel], H["sw2"][ie][jsel]
    Wtot_raw = Sr.sum(1) + H["Wlo"][ie][jsel] + H["Whi"][ie][jsel]
    W2tot_raw = S2r.sum(1) + H["W2lo"][ie][jsel] + H["W2hi"][ie][jsel]
    s = np.where(W2tot_raw > 0, Wtot_raw / np.maximum(W2tot_raw, 1e-300), 0.0) if a.pt_weighting == "neff" \
        else np.ones(len(jsel))
    S, S2 = Sr * s[:, None], S2r * (s * s)[:, None]
    nz = S > 0
    jj, kk = np.nonzero(nz)
    Win = S.sum(1)
    Wlo, Whi = H["Wlo"][ie][jsel] * s, H["Whi"][ie][jsel] * s
    W2lo, W2hi = H["W2lo"][ie][jsel] * s * s, H["W2hi"][ie][jsel] * s * s
    pt = np.where(H["W"][ie][jsel] > 0, H["SP"][ie][jsel] / np.maximum(H["W"][ie][jsel], 1e-300),
                  np.exp(0.5 * (LPT[jsel] + LPT[jsel + 1])))
    return dict(pt=pt, jsel=jsel, s=s, j=jj, r=rc[kk], S=S[nz], S2=S2[nz], lo=R[0], hi=R[-1],
                Wlo=Wlo, Whi=Whi, W2lo=W2lo, W2hi=W2hi, Win=Win, Wtot=Win + Wlo + Whi,
                W2tot=S2.sum(1) + W2lo + W2hi, Wtot_raw=Wtot_raw, W2tot_raw=W2tot_raw)


def penalty_vec(mod):
    """indices of Re/Im b_1k, b_2k in theta (the reference-trade-off directions)."""
    idx = []
    for blk in (0, 1):
        o = mod.oc + blk * (mod.nc - 1)
        for n in (1, 2):
            for k in range(mod.K + 1):
                if n <= mod.J:
                    idx.append(o + n * (mod.K + 1) + k - 1)
    return np.array(idx, int)


def fit(mod, th0, D, lam=0.0):
    Wsum = D["Wtot"].sum()
    pen = penalty_vec(mod) if lam > 0 else np.array([], int)

    def f(th):
        v, g = mod.lnl(th, D)
        f_, g_ = -v / Wsum, -g / Wsum
        if lam > 0:
            f_ += lam * np.sum(th[pen] ** 2); g_ = g_.copy(); g_[pen] += 2 * lam * th[pen]
        return f_, g_

    res = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=a.maxiter, maxfun=4 * a.maxiter))
    lnl = mod.lnl(res.x, D, grad=False)[0]
    return res.x, lnl, bool(res.success), res.nit


def sandwich(mod, th, D):
    """Cov = H^+ V H^+ with H the numerical Hessian of -l (from the analytic gradient)."""
    P = len(th)
    Hm = np.zeros((P, P))
    for i in range(P):
        h = 1e-5 * max(1.0, abs(th[i]))
        e = np.zeros(P); e[i] = h
        Hm[i] = -(mod.lnl(th + e, D)[1] - mod.lnl(th - e, D)[1]) / (2 * h)
    Hm = 0.5 * (Hm + Hm.T)
    G, w2 = mod.scores(th, D)
    V = (G * w2[:, None]).T @ G
    Hi = np.linalg.pinv(Hm, rcond=1e-9, hermitian=True)
    return Hi @ V @ Hi, int(np.sum(np.linalg.eigvalsh(Hm) < 1e-9 * np.abs(np.linalg.eigvalsh(Hm)).max()))


def start_values(D):
    """per-pt-bin mean and sigma_68 -> quadratic mu(u), NSC sigma^2(t) by weighted least squares."""
    S = H["sw"][D["ie"]][D["jsel"]] * D["s"][:, None]
    Win = D["Win"]
    ok = Win > 0
    mean = (S @ rc)[ok] / Win[ok]
    cum = np.cumsum(S[ok], 1) + D["Wlo"][ok, None]
    tot = D["Wtot"][ok]
    q16 = np.array([np.interp(0.16 * t, c, R[1:]) for c, t in zip(cum, tot)])
    q84 = np.array([np.interp(0.84 * t, c, R[1:]) for c, t in zip(cum, tot)])
    s68 = np.maximum(0.5 * (q84 - q16), 2 * dr)
    return mean, s68, ok


# ------------------------------------------------------------------ 2.-5. fits, per |eta| bin
results, all_json = [], dict(var=var, eta_edges=[float(e) for e in ETA], J=J, fits=[])
t_all = time.time()
for ie in eta_list:
    jsel = np.flatnonzero(H["W"][ie] > 0)
    if a.fit_pt_range:
        ptc = np.exp(0.5 * (LPT[:-1] + LPT[1:]))
        jsel = jsel[(ptc[jsel] >= a.fit_pt_range[0]) & (ptc[jsel] <= a.fit_pt_range[1])]
    if len(jsel) < 3:
        print("|%s| bin %d: fewer than 3 pt bins, skipped" % (var, ie)); continue
    D = make_data(ie, jsel); D["ie"] = ie
    W_all = D["Wtot"].sum()
    ne = neff(W_all, D["W2tot"].sum())
    x_range = (np.log(D["pt"].min()), np.log(D["pt"].max()))
    lab = "%.2f < |%s| < %.2f" % (ETA[ie], var, ETA[ie + 1])
    print("\n=== %s: %d pt bins, %.0f - %.0f GeV, %d filled 2D bins, n_eff %.4g, out of range %.3f %%"
          % (lab, len(jsel), D["pt"].min(), D["pt"].max(), len(D["S"]), ne, 100 * (D["Wlo"].sum() + D["Whi"].sum()) / W_all))

    # --- starting values
    mean, s68, ok = start_values(D)
    mod0 = BornModel(J, 0, a.mu_degree, "nsc", x_range)
    g0 = Grid(mod0, D["pt"][ok])
    wls = np.sqrt(D["Win"][ok])
    m0 = np.linalg.lstsq(g0.Lmu * wls[:, None], mean * wls, rcond=None)[0]
    Xs = np.stack([g0.t ** 2, g0.t, np.ones_like(g0.t)], 1)
    A0, B20, C20 = np.linalg.lstsq(Xs * wls[:, None], s68 ** 2 * wls, rcond=None)[0]
    sp0 = [A0, np.sqrt(max(B20, 1e-6)), np.sqrt(max(C20, 1e-6))]
    Bc = np.zeros((J + 1, 1), complex); Bc[0, 0] = 1; Bc[1:, 0] = 0.01j
    th = mod0.pack(m0, sp0, Bc)
    print("start: mu Legendre %s, NSC A %.3g B %.3g C %.3g" % (np.round(m0, 4), *sp0))

    # --- warm-started fits K = 0 ... Kmax
    D["grid"] = Grid(mod0, D["pt"])
    mod, stages = mod0, []
    for K in range(Ks[ie] + 1):
        if K > 0:
            mod, th = mod.extend(th, K)
        D["grid"] = Grid(mod, D["pt"])
        t0 = time.time()
        th, lnl, conv, nit = fit(mod, th, D, a.ref_penalty)
        stages.append(dict(K=K, model=mod, theta=th.copy(), lnl=lnl, conv=conv, P=mod.P))
        print("  K = %d: P = %2d, lnL/sum w = %.6f, n_eff*lnL/sum w = %.1f, %s after %d it, %.0f s"
              % (K, mod.P, lnl / W_all, ne * lnl / W_all, "converged" if conv else "NOT converged", nit, time.time() - t0))
    cov, nflat = sandwich(mod, th, D)
    print("  sandwich covariance: %d flat direction(s) of the Hessian" % nflat)

    # --- benchmark: independent cells
    cells = []
    if not a.no_cells:
        for ic in range(len(CELL) - 1):
            jc = np.flatnonzero((D["pt"] >= CELL[ic]) & (D["pt"] < CELL[ic + 1]))
            if len(jc) == 0 or D["Win"][jc].sum() <= 0:
                continue
            sub = make_data(ie, jsel[jc]); sub["ie"] = ie
            mc, sc, okc = start_values(sub)
            wc = sub["Win"][okc]
            mc0, sc0 = np.average(mc, weights=wc), np.average(sc, weights=wc)
            cm = BornModel(J, 0, 0, "const", x_range)
            sub["grid"] = Grid(cm, sub["pt"])
            thc, lnlc, convc, _ = fit(cm, cm.pack([mc0], [sc0], Bc), sub)
            cells.append(dict(ic=ic, pt=[float(CELL[ic]), float(CELL[ic + 1])], jc=jc, model=cm, theta=thc,
                              lnl=lnlc, conv=convc, W=sub["Wtot"].sum()))
        lnl_cells = sum(c["lnl"] for c in cells)
        P_cells = sum(c["model"].P for c in cells)
        # the global model on exactly the jets of these cells, for a like-for-like comparison
        jcov = np.concatenate([c["jc"] for c in cells])
        Dsub = make_data(ie, jsel[np.sort(jcov)]); Dsub["grid"] = Grid(mod, Dsub["pt"])
        lnl_glob_cells = mod.lnl(th, Dsub, grad=False)[0]
        W_cells = Dsub["Wtot"].sum()
        print("  cells: %d independent fits, P = %d, n_eff*(lnL_global - lnL_cells)/sum w = %.1f"
              % (len(cells), P_cells, ne * (lnl_glob_cells - lnl_cells) / W_cells))

    results.append(dict(ie=ie, lab=lab, D=D, W=W_all, neff=ne, stages=stages, cov=cov, nflat=nflat, cells=cells,
                        lnl_cells=lnl_cells if cells else None, P_cells=P_cells if cells else None,
                        lnl_glob_cells=lnl_glob_cells if cells else None, W_cells=W_cells if cells else None))
    fin = stages[-1]
    all_json["fits"].append(dict(
        eta_bin=int(ie), abs_eta=[float(ETA[ie]), float(ETA[ie + 1])], n_eff=ne, sum_w=W_all,
        pt_range=[float(D["pt"].min()), float(D["pt"].max())], model=fin["model"].to_dict(fin["theta"]),
        cov=cov.tolist(), flat_directions=nflat, lnL=fin["lnl"], converged=fin["conv"],
        stages=[dict(K=s["K"], P=s["P"], lnL=s["lnl"], converged=s["conv"], model=s["model"].to_dict(s["theta"]))
                for s in stages],
        cells=[dict(pt_gen=c["pt"], lnL=c["lnl"], converged=c["conv"], model=c["model"].to_dict(c["theta"]))
               for c in cells]))

print("\nall fits: %.0f s" % (time.time() - t_all))
with open(os.path.join(outdir, "global_fits.json"), "w") as fh:
    json.dump(all_json, fh, indent=1)

# ------------------------------------------------------------------ summary.txt
L = ["# n_eff*lnL: log-likelihood in units of effective events, n_eff * lnL / sum w",
     "# dlnL: n_eff*lnL minus that of the previous line (K-1); cells: global minus independent cells,",
     "#       both evaluated on the jets of the benchmark cells",
     "# AIC_eff = 2 P - 2 n_eff*lnL (lower is better)",
     "%-18s %-10s %4s %14s %10s %12s  %s" % ("|%s|" % var, "model", "P", "n_eff*lnL", "dlnL", "AIC_eff", "status")]
for res in results:
    ne, W = res["neff"], res["W"]
    prev = None
    for s in res["stages"]:
        v = ne * s["lnl"] / W
        L.append("%-18s %-10s %4d %14.1f %10s %12.1f  %s" % (res["lab"].replace(" ", ""), "K=%d" % s["K"], s["P"], v,
                 "" if prev is None else "%.1f" % (v - prev), 2 * s["P"] - 2 * v, "ok" if s["conv"] else "NOT CONVERGED"))
        prev = v
    if res["cells"]:
        vg = ne * res["lnl_glob_cells"] / res["W_cells"]
        vc = ne * res["lnl_cells"] / res["W_cells"]
        P = res["stages"][-1]["P"]
        L.append("%-18s %-10s %4d %14.1f %10.1f %12.1f  on the cells' jets: global n_eff*lnL %.1f, AIC_eff %.1f"
                 % ("", "cells(%d)" % len(res["cells"]), res["P_cells"], vc, vg - vc, 2 * res["P_cells"] - 2 * vc,
                    vg, 2 * P - 2 * vg))
    mod, th = res["stages"][-1]["model"], res["stages"][-1]["theta"]
    err = np.sqrt(np.maximum(np.diag(res["cov"]), 0))
    nm = mod.names()
    L.append("%-18s reference: " % "" + ", ".join("%s %.4g(%.2g)" % (nm[i], th[i], err[i]) for i in range(mod.oc))
             + "   [ln pt range %.3f - %.3f; %d flat directions]" % (mod.x0, mod.x1, res["nflat"]))
print("\n" + "\n".join(L))
with open(os.path.join(outdir, "summary.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")

# ------------------------------------------------------------------ 6. plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RG = np.linspace(-1.0, 3.0, 8001)                       # r grid for model CDFs / quantiles
RGc = 0.5 * (RG[1:] + RG[:-1])
inr = (RGc >= R[0]) & (RGc < R[-1])


def summaries(pdf):
    """rows of pdf on RG -> quantiles PQ, in-range mean, density at the quantiles."""
    cell = 0.5 * (pdf[:, 1:] + pdf[:, :-1]) * (RG[1] - RG[0])
    cdf = np.concatenate([np.zeros((len(pdf), 1)), np.cumsum(cell, 1)], 1)
    cdf /= cdf[:, -1:]
    Q = np.array([np.interp(PQ, c, RG) for c in cdf])
    mean_in = (cell[:, inr] @ RGc[inr]) / cell[:, inr].sum(1)
    dens = np.array([np.interp(q, RG, p) for q, p in zip(Q, pdf)])
    return Q, mean_in, dens


def derived(Q, mean_in):
    med, s68 = Q[:, 4], 0.5 * (Q[:, 5] - Q[:, 3])
    tails = [(Q[:, i] - med) / s68 for i in (0, 1, 2, 6, 7, 8)]
    return np.stack([med, s68 / med, mean_in] + tails)          # (9, n)


PANELS = ["median $Q_{50}$", "$\\sigma_{68}$ / median", "mean (r in range)"] + \
         ["$(Q_{%g} - Q_{50})/\\sigma_{68}$" % (100 * PQ[i]) for i in (0, 1, 2, 6, 7, 8)]
GAUSS = [None, None, None] + [ZG[i] for i in (0, 1, 2, 6, 7, 8)]


def model_curves(mod, th, pt):
    Q, m, _ = summaries(mod.pdf_grid(th, pt, RG))
    return derived(Q, m)


celldir = os.path.join(outdir, "cells")
if a.cell_plots:
    os.makedirs(celldir, exist_ok=True)

for res in results:
    ie, D, lab = res["ie"], res["D"], res["lab"]
    mod, th, cov = res["stages"][-1]["model"], res["stages"][-1]["theta"], res["cov"]
    x = np.log(D["pt"])

    # --- curves: model with delta-method band
    ptg = np.exp(np.linspace(mod.x0, mod.x1, 60))
    cur = model_curves(mod, th, ptg)
    Jm = np.zeros((cur.size, len(th)))
    for i in range(len(th)):
        h = 1e-4 * max(1.0, abs(th[i]))
        e = np.zeros(len(th)); e[i] = h
        Jm[:, i] = ((model_curves(mod, th + e, ptg) - model_curves(mod, th - e, ptg)) / (2 * h)).ravel()
    band = np.sqrt(np.maximum(np.einsum("ip,pq,iq->i", Jm, cov, Jm), 0)).reshape(cur.shape)

    # --- data groups (consecutive pt bins up to --group-neff) and the model mixture over each
    groups, cur_g, w_, w2_ = [], [], 0.0, 0.0
    for j in range(len(D["pt"])):
        cur_g.append(j)
        w_ += D["Wtot"][j]; w2_ += D["W2tot"][j]
        if neff(w_, w2_) >= a.group_neff or j == len(D["pt"]) - 1:
            groups.append((np.array(cur_g), neff(w_, w2_))); cur_g, w_, w2_ = [], 0.0, 0.0
    pdf_all = mod.pdf_grid(th, D["pt"], RG)
    gx, gdat, gerr, gmix = [], [], [], []
    for js, ng in groups:
        Wj = D["Wtot"][js]
        mix = (Wj @ pdf_all[js]) / Wj.sum()
        Qm, mm, dens = summaries(mix[None])
        gmix.append(derived(Qm, mm)[:, 0])
        S = (H["sw"][ie][D["jsel"][js]] * D["s"][js, None]).sum(0)          # same weighting as the fit
        S2 = (H["sw2"][ie][D["jsel"][js]] * (D["s"][js] ** 2)[:, None]).sum(0)
        tot = Wj.sum()
        cum = np.r_[D["Wlo"][js].sum(), D["Wlo"][js].sum() + np.cumsum(S)]
        Qd = np.array([np.interp(p * tot, cum, R) if cum[0] < p * tot <= cum[-1] else np.nan for p in PQ])
        md = S @ rc / S.sum()
        eQ = np.sqrt(PQ * (1 - PQ) / max(ng, 1)) / np.maximum(dens[0], 1e-12)
        med, s68 = Qd[4], 0.5 * (Qd[5] - Qd[3])
        dd = derived(Qd[None], np.array([md]))[:, 0]
        ee = np.r_[eQ[4], 0.5 * np.hypot(eQ[5], eQ[3]) / med, np.sqrt(S2 @ (rc - md) ** 2) / S.sum(),
                   [eQ[i] / s68 for i in (0, 1, 2, 6, 7, 8)]]
        gx.append(np.log(np.average(D["pt"][js], weights=Wj))); gdat.append(dd); gerr.append(ee)
    gx, gdat, gerr, gmix = map(np.array, (gx, gdat, gerr, gmix))
    cx, cval = [], []
    for c in res["cells"]:
        Qc, mc, _ = summaries(c["model"].pdf_grid(c["theta"], [D["pt"][c["jc"]][0]], RG))
        cx.append(np.log(np.average(D["pt"][c["jc"]], weights=D["Wtot"][c["jc"]]))); cval.append(derived(Qc, mc)[:, 0])

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for p, ax in enumerate(axes.flat):
        xg = np.log(ptg)
        ax.fill_between(xg, cur[p] - band[p], cur[p] + band[p], color="C3", alpha=0.25, lw=0)
        ax.plot(xg, cur[p], color="C3", lw=1.3, label="global fit (K=%d)" % mod.K)
        ax.plot(gx, gmix[:, p], "s", mfc="none", color="C3", ms=5, label="global, mixture over group")
        ax.errorbar(gx, gdat[:, p], gerr[:, p], fmt="ko", ms=3, capsize=1.5, label="data")
        if cval:
            ax.plot(cx, np.array(cval)[:, p], "^", color="C0", ms=4.5, label="independent cell fits")
        if GAUSS[p] is not None:
            ax.axhline(GAUSS[p], color="gray", ls="--", lw=0.8)
        ax.set_title(PANELS[p], fontsize=10); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("ln($p_T^{gen}$ / GeV)")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("global amplitude fit, %s  (dashed: Gaussian)" % lab)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(outdir, "curves_eta%d.png" % ie), dpi=120); plt.close(fig)

    # --- coefficients and moments along pt
    cn, _, _ = mod.coefficients(th, ptg)
    cn = cn * np.exp(-1j * np.angle(cn[:, :1]))          # display phase: c_0 real >= 0
    mom = np.array([standardised_moments(c, a.kmax)[2] for c in cn])
    npan = (J + 1) + max(a.kmax - 2, 0)
    ncol = 5
    fig, axes = plt.subplots(int(np.ceil(npan / ncol)), ncol, figsize=(3.2 * ncol, 2.5 * np.ceil(npan / ncol)),
                             sharex=True, squeeze=False)
    xg = np.log(ptg)
    for n in range(J + 1):
        ax = axes.flat[n]
        ax.plot(xg, cn[:, n].real, color="C0", label="Re"); ax.plot(xg, cn[:, n].imag, color="C3", label="Im")
        ax.plot(xg, np.abs(cn[:, n]), "--", color="gray", lw=1, label="|c|")
        ax.set_title("$c_{%d}$" % n, fontsize=10)
    axes.flat[0].legend(fontsize=7)
    for k in range(3, a.kmax + 1):
        ax = axes.flat[J + 1 + k - 3]
        ax.plot(xg, mom[:, k], color="C2"); ax.set_title("$a_{%d}$" % k, fontsize=10)
    for ax in axes.flat[npan:]:
        ax.set_visible(False)
    for ax in axes.flat[:npan]:
        ax.axhline(0, color="gray", lw=0.6); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("ln($p_T^{gen}$ / GeV)", fontsize=9)
    fig.suptitle("global fit: $c_n(p_T)$ (phase: $c_0$ real) and standardised moments $a_k(p_T)$, %s" % lab, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(outdir, "coefficients_eta%d.png" % ie), dpi=110); plt.close(fig)

    # --- cell plots: data vs the mixture of the global model (and the independent cell fit)
    if not a.cell_plots:
        continue
    fine = nr // a.bins
    edges = R[::fine]
    cen = 0.5 * (edges[1:] + edges[:-1])
    for ic in range(len(CELL) - 1):
        jc = np.flatnonzero((D["pt"] >= CELL[ic]) & (D["pt"] < CELL[ic + 1]))
        if len(jc) == 0 or D["Win"][jc].sum() <= 0:
            continue
        S = H["sw"][ie][D["jsel"][jc]].sum(0).reshape(a.bins, fine).sum(1)
        S2 = H["sw2"][ie][D["jsel"][jc]].sum(0).reshape(a.bins, fine).sum(1)
        Win = S.sum(); width = edges[1] - edges[0]
        pf = mod.pdf_grid(th, D["pt"][jc], rc)              # model on the fine bin centres
        Wr = D["Wtot_raw"][jc]                              # the cell as it is: cross-section weights
        expct = (Wr @ pf).reshape(a.bins, fine).mean(1) * width
        curves = [("global K=%d" % mod.K, "C3", expct)]
        cfit = [c for c in res["cells"] if c["ic"] == ic]
        if cfit:
            pc = cfit[0]["model"].pdf_grid(cfit[0]["theta"], [D["pt"][jc][0]], rc)[0]
            curves.append(("cell fit", "C0", Wr.sum() * pc.reshape(a.bins, fine).mean(1) * width))
        ng = neff(Wr.sum(), D["W2tot_raw"][jc].sum())
        fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
        for col, logy in ((0, False), (1, True)):
            ax, axr = axs[0, col], axs[1, col]
            ok = S > 0
            ax.errorbar(cen[ok], S[ok] / (Win * width), np.sqrt(S2[ok]) / (Win * width), fmt="k.", ms=3, label="data")
            for name, colr, e in curves:
                ax.plot(cen, e / (Win * width), color=colr, lw=1.2, label=name)
                with np.errstate(divide="ignore", invalid="ignore"):
                    axr.errorbar(cen[ok] + (0.002 if colr == "C0" else 0), S[ok] / e[ok] - 1, np.sqrt(S2[ok]) / e[ok],
                                 fmt=".", color=colr, ms=2.5, lw=0.7)
            axr.axhline(0, color="gray", lw=0.8); axr.set_ylim(-a.res_max, a.res_max)
            axr.set_xlabel("$r = p_T^{rec}/p_T^{gen}$"); axr.grid(alpha=0.3)
            if logy:
                ax.set_yscale("log"); ax.set_ylim(bottom=max(0.3 * (S[ok] / (Win * width)).min(), 1e-6))
            ax.grid(alpha=0.3)
        axs[0, 0].set_ylabel("density"); axs[1, 0].set_ylabel("data/model - 1"); axs[0, 0].legend(fontsize=8)
        fig.suptitle("%s,  %g < $p_T^{gen}$ < %g GeV,  n_eff %.0f" % (lab, CELL[ic], CELL[ic + 1], ng))
        fig.savefig(os.path.join(celldir, "cell_eta%d_pt%02d.png" % (ie, ic)), dpi=100, bbox_inches="tight")
        plt.close(fig)

print("\nsummary.txt, global_fits.json, curves_eta*.png, coefficients_eta*.png%s in %s"
      % (", cells/" if a.cell_plots else "", outdir))
