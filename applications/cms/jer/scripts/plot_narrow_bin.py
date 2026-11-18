#!/usr/bin/env python3
"""The global (pt-smooth) model in arbitrary, narrow pt_gen bins: a prediction where a cell fit
is impossible for lack of data.

For a bin [pa, pb) the global model predicts the mixture (response_fits.tex, Eq. binmix)

    p_bin(r) = sum_i f_i p(r | pt_i) / sum_i f_i,      pt_i: --npt points evenly in ln pt inside the bin

with f the pt_gen spectrum inside the bin (--spectrum): the weighted pt_gen of the data events
(default when there are events), flat in ln pt, or a power law pt^-n.  For narrow bins the choice
does not matter; the figure also draws p(r | <pt>) at the bin's mean ln pt (dashed) and reports
the largest relative difference to the mixture.  The band is the delta-method 1 sigma band from
the covariance in global_fits.json.  Densities are conditional on r in [0, 2), like the fits.

Data (all inputs are only read):
  * --files (on the cluster): the event export, as in fit_response_global.py (keys genpt, recpt,
    weight, dR and --var).  --save-events writes the selected events of all requested bins to a
    small npz, which --events reads back (e.g. on a laptop, without the export).
  * --hist (for local tests): histograms2d.npz.  Its pt bins are 2 % wide, so a bin narrower than
    that gets the data of the enclosing fine pt bin(s), labelled as such in the plot.
  * --subsample N: only N randomly chosen events of each bin (event data only): what the data
    look like when a bin holds almost nothing; the model does not change.

    python3 scripts/plot_narrow_bin.py --global fig/global_K3/global_fits.json --eta-bin 0 \
        --pt 300 301 --pt 1500 1510 --save-events fig/narrow_events.npz
    python3 scripts/plot_narrow_bin.py --global ... --events fig/narrow_events.npz --pt 300 301 --subsample 30
    python3 scripts/plot_narrow_bin.py --global figs/global_polynomial/global_fits.json --hist \
        figs/global_polynomial/histograms2d.npz --eta-bin 0 --pt 300 301

Unbinned check (works with a handful of events): the weighted mean of ln p_bin(r) over the
events against its expectation under the model, E_p[ln p] +- sd_p[ln p] / sqrt(n_eff); z is the
deviation in units of that error (|z| < 2 consistent).

Independent local fit (--local-fit, event data only): the amplitude model of one cell, fitted to
the events of this bin alone, to show what the data of the bin can determine by themselves:

    p(r) = N(r; mu*, sigma*) |sum_n c_n phi_n(z)|^2 / ||c||^2,   n = 0 ... J (J of the global fit)

with the reference (mu*, sigma*) fixed to the global model's mu(pt), sigma(pt) at the bin's mean
ln pt, so that the c_n of both fits refer to the same Hermite basis and are directly comparable;
only the 2J real amplitude parameters are free (b_0 = 1).  Weighted unbinned likelihood with the
events at r >= 2 as a tail term, as in fit_response_global.py; best of two starts (the global c
and a neutral one).  Errors: sandwich covariance, propagated to the normalised c_n.  The c_n are
shown with c_0 real >= 0; |h|^2 does not change under c -> conj(c), so the local solution is shown
in the conjugation that lies closer to the global one.  2 Delta lnL (local - global, in n_eff units)
measures what the 2J extra parameters gain: about 2J if the global model is right.

Outputs in --outdir (default <dir of --global>/narrow_bins):
  narrow_eta<i>_pt<lo>-<hi>[_sub<N>].png   data and model, linear and log; below the relative residual
                                            data/model - 1 in every r bin with data (as in
                                            compare_global_cells.py; y range +-res-max)
  narrow_bins.txt / .json                   per bin: events, n_eff, model median and
                                            sigma_68/median with errors, data median, lnL check
  with --local-fit also
  narrow_eta<i>_pt<lo>-<hi>[_sub<N>]_coef.png   c_n: global curves vs ln pt (+-1 sigma) and the local
                                            fit of this bin (points)
  narrow_local_vs_neff_eta<i>.png           (2 or more bins) local c_n of every bin against its
                                            n_eff, global value at the same pt as a bar
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from response_models import BornModel, Grid

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", default="fig/global_receta/global_fits.json")
ap.add_argument("--eta-bin", type=int, nargs="+", default=[0], help="|eta| bin indices of the global fit")
ap.add_argument("--pt", nargs=2, type=float, action="append", metavar=("LO", "HI"),
                help="pt_gen bin in GeV (repeatable; default 300 301)")
src = ap.add_mutually_exclusive_group()
src.add_argument("--files", default=None, help="event export (glob), as in fit_response_global.py")
src.add_argument("--events", default=None, help="npz written by --save-events")
src.add_argument("--hist", default=None, help="histograms2d.npz (local tests; fine pt bins of 2 %%)")
ap.add_argument("--save-events", default=None, help="with --files: write the selected events here")
ap.add_argument("--var", default=None, help="|eta| variable of the export (default: from the global fit)")
ap.add_argument("--max-dr", type=float, default=0.4, help="matching radius, as in fit_response_global.py")
ap.add_argument("--subsample", type=int, default=None, help="keep only N random events per bin")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--spectrum", default="auto", choices=("auto", "data", "flat", "power"),
                help="pt_gen spectrum f inside the bin (auto: data if there are events, else power)")
ap.add_argument("--power", type=float, default=5.0, help="n of f ~ pt^-n with --spectrum power")
ap.add_argument("--npt", type=int, default=21, help="pt points of the mixture inside the bin")
ap.add_argument("--bins", type=int, default=40, help="r bins of the plot")
ap.add_argument("--r-range", nargs=2, type=float, default=None, help="plot range (default: from the model)")
ap.add_argument("--no-band", action="store_true")
ap.add_argument("--res-max", type=float, default=0.5, help="y range +-res-max of the residual panels (data/model - 1)")
ap.add_argument("--local-fit", action="store_true",
                help="also fit the amplitudes c_n to the events of each bin alone and compare with the global c_n(pt)")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
bins_pt = a.pt or [[300.0, 301.0]]


def ptf(x):
    """pt edge for labels and file names: up to 10 digits, so 300.5-300.50001 is not shown as 300.5-300.5"""
    return "%.10g" % x


# colours: global model dark blue / dark red (Re / Im), local fit bluish green / orange (Re / Im),
# so that the fit is told by the hue and Re / Im by the marker; all fully saturated
COL = dict(glob_re="#0033A0", glob_im="#C00000", loc_re="#009E73", loc_im="#E07000",
           glob_dens="#D00000", loc_dens="#009E73")


gdir = os.path.dirname(os.path.abspath(a.glob))
outdir = a.outdir or os.path.join(gdir, "narrow_bins")
os.makedirs(outdir, exist_ok=True)
inputs = [os.path.abspath(p) for p in (a.glob, a.events, a.hist) if p]
if a.save_events and os.path.abspath(a.save_events) in inputs:
    raise SystemExit("--save-events must not overwrite an input")

# ------------------------------------------------------------------ global fit (read only)
with open(a.glob) as fh:
    G = json.load(fh)
FITS = {f["eta_bin"]: f for f in G["fits"]}
var = a.var or G.get("var", "receta")
R = np.linspace(0.0, 2.0, 1001)                        # fine r grid of the densities (as the fits)
rc, dr = 0.5 * (R[1:] + R[:-1]), R[1] - R[0]


def eta_range(f):
    return tuple(float(v) for v in f["abs_eta"])


# ------------------------------------------------------------------ data
def read_export():
    files = sorted(glob.glob(a.files))
    if not files:
        raise SystemExit("no files match " + a.files)
    ev = {}
    lo_all, hi_all = min(b[0] for b in bins_pt), max(b[1] for b in bins_pt)
    for i, fn in enumerate(files):
        with np.load(fn) as z:
            pt = z["genpt"]
            sel = (pt >= lo_all) & (pt < hi_all)
            if a.max_dr < 0.4:
                sel &= z["dR"] < a.max_dr
            idx = np.flatnonzero(sel)
            pt = pt[idx].astype(np.float64)
            r = z["recpt"][idx].astype(np.float64) / pt
            w = z["weight"][idx].astype(np.float64)
            eta = np.abs(z[var][idx]).astype(np.float64)
        for ie in a.eta_bin:
            e0, e1 = eta_range(FITS[ie])
            for b in bins_pt:
                m = (eta >= e0) & (eta < e1) & (pt >= b[0]) & (pt < b[1])
                key = (ie, b[0], b[1])
                old = ev.get(key, (np.zeros(0), np.zeros(0), np.zeros(0)))
                ev[key] = tuple(np.r_[o, n[m]] for o, n in zip(old, (pt, r, w)))
        print("  %d/%d %s" % (i + 1, len(files), os.path.basename(fn)), flush=True)
    return ev


def save_events(ev, path):
    out = {"var": var}
    for k, (ie, lo, hi) in enumerate(ev):
        pt, r, w = ev[(ie, lo, hi)]
        out["key_%d" % k] = np.array([ie, lo, hi])
        out["pt_%d" % k], out["r_%d" % k], out["w_%d" % k] = pt, r, w
    np.savez_compressed(path, **out)
    print("events written to", path)


def load_events(path):
    ev = {}
    with np.load(path) as z:
        for k in range(sum(1 for n in z.files if n.startswith("key_"))):
            ie, lo, hi = z["key_%d" % k]
            ev[(int(ie), float(lo), float(hi))] = (z["pt_%d" % k], z["r_%d" % k], z["w_%d" % k])
    return ev


EV, H = None, None
if a.files:
    EV = read_export()
    if a.save_events:
        save_events(EV, a.save_events)
elif a.events:
    EV = load_events(a.events)
elif a.hist:
    with np.load(a.hist) as z:
        H = {k: z[k] for k in ("eta_edges", "lpt_edges", "r_edges", "sw", "sw2", "W", "SP")}
    if not np.allclose(H["r_edges"], R):
        raise SystemExit("histogram r grid differs from [0, 2) in 1000 bins")
rng = np.random.default_rng(a.seed)


def data_for(ie, lo, hi):
    """-> dict(kind, label, pt/r/w or S/S2/ptj/Wj) or None."""
    if EV is not None:
        # a stored bin that contains [lo, hi) (the exact one, or a wider one: the events are cut
        # down to [lo, hi), so a saved 300-301 bin serves 300-300.5, 300.2-300.3, ...)
        cont = [k for k in EV if k[0] == ie and k[1] <= lo + 1e-9 and hi <= k[2] + 1e-9]
        if not cont:
            print("  no stored events cover |eta| bin %d, %s-%s GeV (stored: %s; run with --files)"
                  % (ie, ptf(lo), ptf(hi), ", ".join("%s-%s" % (ptf(k[1]), ptf(k[2])) for k in EV if k[0] == ie)
                     or "none"))
            return None
        key = min(cont, key=lambda k: k[2] - k[1])
        pt, r, w = EV[key]
        sel = (pt >= lo) & (pt < hi)
        pt, r, w = pt[sel], r[sel], w[sel]
        n_all = len(pt)
        if a.subsample and a.subsample < len(pt):
            k = rng.choice(len(pt), a.subsample, replace=False)
            pt, r, w = pt[k], r[k], w[k]
        ins = (r >= R[0]) & (r < R[-1])
        lo_, hi_ = r < R[0], r >= R[-1]
        lab = "data, %d events" % len(pt) + (" (random subset of %d)" % n_all if len(pt) < n_all else "")
        return dict(kind="events", label=lab, pt=pt[ins], r=r[ins], w=w[ins], n_out=int((~ins).sum()),
                    w_lo=float(w[lo_].sum()), w_hi=float(w[hi_].sum()),            # tail terms of --local-fit
                    w2_lo=float((w[lo_] ** 2).sum()), w2_hi=float((w[hi_] ** 2).sum()))
    if H is not None:
        e0, e1 = eta_range(FITS[ie])
        ke = int(np.flatnonzero(np.isclose(H["eta_edges"][:-1], e0))[0])
        W = H["W"][ke]
        ok = W > 0
        ptj = np.where(ok, H["SP"][ke] / np.where(ok, W, 1.0), np.nan)
        js = np.flatnonzero(ok & (ptj >= lo) & (ptj < hi))
        LPT = H["lpt_edges"]
        if len(js) == 0:                           # bin narrower than a fine bin: the enclosing one(s)
            js = np.flatnonzero((np.exp(LPT[1:]) > lo) & (np.exp(LPT[:-1]) < hi) & ok)
            if len(js) == 0:
                return None
        plo, phi = np.exp(LPT[js[0]]), np.exp(LPT[js[-1] + 1])
        S, S2 = H["sw"][ke][js].sum(0), H["sw2"][ke][js].sum(0)
        return dict(kind="hist", label="data, histogram bin %.0f-%.0f GeV" % (plo, phi), S=S, S2=S2,
                    ptj=ptj[js], Wj=H["sw"][ke][js].sum(1), enclosing=(plo, phi))
    return None


# ------------------------------------------------------------------ model in a bin
def spectrum(lo, hi, D):
    """pt points (bin centres in ln pt) and weights f inside [lo, hi)."""
    xe = np.linspace(np.log(lo), np.log(hi), a.npt + 1)
    x = 0.5 * (xe[1:] + xe[:-1])
    kind = a.spectrum
    if kind == "auto":
        kind = "data" if D is not None and D["kind"] == "events" and D["w"].sum() > 0 else "power"
    if kind == "data":
        if D is None or D["kind"] != "events" or D["w"].sum() <= 0:
            raise SystemExit("--spectrum data needs events in the bin")
        f = np.histogram(np.log(D["pt"]), xe, weights=D["w"])[0]
        lab = "data spectrum"
    elif kind == "flat":
        f, lab = np.ones_like(x), "flat in ln $p_T$"
    else:
        f, lab = np.exp((1 - a.power) * x), "$p_T^{-%g}$ spectrum" % a.power     # dN/dln pt ~ pt^(1-n)
    return np.exp(x), f / f.sum(), lab


def cond(p):
    return p / (p.sum(-1, keepdims=True) * dr)


def quant(dens, q):
    cdf = np.r_[0.0, np.cumsum(dens) * dr]
    return np.interp(np.asarray(q) * cdf[-1], cdf, R)


def summary(mod, t, pts, f):
    """mixture density on the fine grid and (median, sigma_68 / median)."""
    mix = f @ cond(mod.pdf_grid(t, pts, rc))
    q16, q50, q84 = quant(mix, [0.16, 0.5, 0.84])
    return mix, np.array([q50, 0.5 * (q84 - q16) / q50])


def global_coefs(mod, th, cov, pts):
    """global c_n(pt) (rows; c_0 real >= 0), mu, sigma and the delta-method errors of Re, Im c_n."""
    def vec(t):
        c = mod.coefficients(t, pts)[0]
        c = c * np.exp(-1j * np.angle(c[:, :1]))
        return np.r_[c.real.ravel(), c.imag.ravel()]
    c, mu, sig = mod.coefficients(th, pts)
    c = c * np.exp(-1j * np.angle(c[:, :1]))
    err = np.zeros(2 * c.size)
    if not a.no_band:
        Jm = np.zeros((len(err), len(th)))
        for i in range(len(th)):
            h = 1e-5 * max(1.0, abs(th[i]))
            e = np.zeros(len(th)); e[i] = h
            Jm[:, i] = (vec(th + e) - vec(th - e)) / (2 * h)
        err = np.sqrt(np.maximum(np.einsum("ip,pq,iq->i", Jm, cov, Jm), 0))
    return c, mu, sig, err[:c.size].reshape(c.shape), err[c.size:].reshape(c.shape)


def local_fit(mod, th, cov, ptm, D):
    """independent amplitude fit to the events of one bin (module docstring, --local-fit)."""
    cg, mug, sigg, cg_re, cg_im = global_coefs(mod, th, cov, [ptm])
    cg, mu0, s0 = cg[0], float(mug[0]), float(sigg[0])
    lm = BornModel(mod.J, 0, 0, "const", (np.log(ptm), np.log(ptm)), mod.pt_scale)
    w, r = D["w"], D["r"]
    LD = dict(grid=Grid(lm, [ptm]), j=np.zeros(len(r), int), r=r, S=w, S2=w * w, lo=R[0], hi=R[-1],
              Wlo=np.array([D["w_lo"]]), Whi=np.array([D["w_hi"]]),
              W2lo=np.array([D["w2_lo"]]), W2hi=np.array([D["w2_hi"]]))
    LD["Wtot"] = np.array([w.sum() + D["w_lo"] + D["w_hi"]])
    W2tot = (w * w).sum() + D["w2_lo"] + D["w2_hi"]
    Wsum, oc = LD["Wtot"][0], lm.oc
    fixed = np.array([mu0, s0])                       # reference fixed to the global mu, sigma

    def full(x):
        return np.r_[fixed, x]

    def f(x):
        v, g = lm.lnl(full(x), LD)
        return -v / Wsum, -g[oc:] / Wsum

    def coef(x):
        B = lm.unpack(full(x))[2][:, 0]
        return B / np.linalg.norm(B)                   # b_0 = 1 real: c_0 real > 0 already

    x_glob = lm.pack([mu0], [s0], cg[:, None])[oc:]
    x_neut = lm.pack([mu0], [s0], np.r_[1.0, np.full(mod.J, 0.01j)])[oc:]
    best = None
    for x0 in (x_glob, x_neut):
        res = minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, maxfun=20000, gtol=1e-10, ftol=1e-14))
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    x = best.x
    lnl_loc = -best.fun * Wsum
    lnl_glob = lm.lnl(full(x_glob), LD, grad=False)[0]
    neff_ = Wsum ** 2 / W2tot

    # sandwich covariance of the free parameters, propagated to Re, Im of the normalised c_n
    P = len(x)
    Hm = np.zeros((P, P))
    for i in range(P):
        h = 1e-5 * max(1.0, abs(x[i]))
        e = np.zeros(P); e[i] = h
        Hm[i] = (f(x + e)[1] - f(x - e)[1]) * Wsum / (2 * h)
    Hm = 0.5 * (Hm + Hm.T)
    G, w2 = lm.scores(full(x), LD)
    G = G[:, oc:]
    V = (G * w2[:, None]).T @ G
    Hi = np.linalg.pinv(Hm, rcond=1e-9, hermitian=True)
    covx = Hi @ V @ Hi
    Jc = np.zeros((2 * (mod.J + 1), P))
    for i in range(P):
        h = 1e-6 * max(1.0, abs(x[i]))
        e = np.zeros(P); e[i] = h
        cp, cm_ = coef(x + e), coef(x - e)
        Jc[:, i] = np.r_[(cp - cm_).real, (cp - cm_).imag] / (2 * h)
    ec = np.sqrt(np.maximum(np.einsum("ip,pq,iq->i", Jc, covx, Jc), 0))
    c = coef(x)
    if np.linalg.norm(np.conj(c) - cg) < np.linalg.norm(c - cg):     # same density, closer to global
        c = np.conj(c)
    dens = cond(lm.pdf_grid(full(x), [ptm], rc))[0]
    return dict(c=c, c_re_err=ec[:mod.J + 1], c_im_err=ec[mod.J + 1:], c_glob=cg, c_glob_re_err=cg_re[0],
                c_glob_im_err=cg_im[0], mu_ref=mu0, sigma_ref=s0, n_par=P, converged=bool(best.success),
                lnl_local=float(lnl_loc), lnl_global=float(lnl_glob),
                two_dlnl_neff=float(2 * (lnl_loc - lnl_glob) * neff_ / Wsum), n_eff=float(neff_), dens=dens)


rows, local_rows = [], {}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for ie in a.eta_bin:
    if ie not in FITS:
        raise SystemExit("no |eta| bin %d in %s" % (ie, a.glob))
    fit = FITS[ie]
    mod, th = BornModel.from_dict(fit["model"])
    cov = np.asarray(fit["cov"])
    e0, e1 = eta_range(fit)
    for lo, hi in bins_pt:
        D = data_for(ie, lo, hi)
        pts, f, spec_lab = spectrum(lo, hi, D)
        xbar = float(f @ np.log(pts))
        extrap = not (mod.x0 - 1e-9 <= np.log(lo) and np.log(hi) <= mod.x1 + 1e-9)

        # plot range and r bins, snapped to the fine grid
        mix, val = summary(mod, th, pts, f)
        if a.r_range:
            r0, r1 = a.r_range
        else:
            qa, qb = quant(mix, [0.002, 0.998])
            r0, r1 = max(0.0, qa - 0.05), min(2.0, qb + 0.05)
        k0, k1 = int(np.floor(r0 / dr)), int(np.ceil(r1 / dr))
        grp = max(1, int(round((k1 - k0) / a.bins)))
        k1 = k0 + grp * int(np.ceil((k1 - k0) / grp))
        k1 = min(k1, len(rc))
        nb = (k1 - k0) // grp
        k1 = k0 + nb * grp
        E_ = R[k0:k1 + 1:grp]
        Ec = 0.5 * (E_[1:] + E_[:-1])

        def coarse_mass(dens):
            return (dens[k0:k1] * dr).reshape(nb, grp).sum(1)

        # delta-method band of the coarse densities and of (median, sigma_68/median)
        vec = lambda t: (lambda m_v: np.r_[coarse_mass(m_v[0]) / (grp * dr), m_v[1]])(summary(mod, t, pts, f))
        v0 = vec(th)
        err = np.zeros_like(v0)
        if not a.no_band:
            Jm = np.zeros((len(v0), len(th)))
            for i in range(len(th)):
                h = 1e-4 * max(1.0, abs(th[i]))
                e = np.zeros(len(th)); e[i] = h
                Jm[:, i] = (vec(th + e) - vec(th - e)) / (2 * h)
            err = np.sqrt(np.maximum(np.einsum("ip,pq,iq->i", Jm, cov, Jm), 0))
        dens_c, dens_e = v0[:nb], err[:nb]
        val_e = err[nb:]
        point = cond(mod.pdf_grid(th, np.exp([xbar]), rc))[0]
        m_in = mix > 1e-3 * mix.max()
        dpoint = float(np.max(np.abs(point[m_in] / mix[m_in] - 1)))

        # data in the plot bins, data median and the unbinned lnL check
        info = dict(eta_bin=ie, abs_eta=[e0, e1], pt_gen=[lo, hi], var=var, spectrum=spec_lab,
                    mean_lnpt=xbar, extrapolated=bool(extrap),
                    model=dict(median=float(val[0]), median_err=float(val_e[0]),
                               s68_over_median=float(val[1]), s68_over_median_err=float(val_e[1]),
                               max_rel_diff_point_vs_mixture=dpoint))
        # reference for the data comparison: the bin's mixture; for histogram data (wider than the
        # bin) the model mixed over the same fine pt bins as the data, so that the test is fair
        ref = mix
        if D is not None and D["kind"] == "hist":
            ref = (D["Wj"] / D["Wj"].sum()) @ cond(mod.pdf_grid(th, D["ptj"], rc))
        lnp = np.log(np.maximum(ref, 1e-300))
        Elp = float(np.sum(ref * lnp) * dr)
        sdlp = float(np.sqrt(max(np.sum(ref * (lnp - Elp) ** 2) * dr, 0.0)))
        if D is not None:
            if D["kind"] == "events":
                w, r = D["w"], D["r"]
                O, V = np.histogram(r, E_, weights=w)[0], np.histogram(r, E_, weights=w * w)[0]
                Win, W2 = w.sum(), (w * w).sum()
                ir = np.clip(((r - R[0]) / dr).astype(int), 0, len(rc) - 1)
                mlp = float(np.sum(w * lnp[ir]) / Win) if Win > 0 else np.nan
                mass = np.bincount(ir, w, len(rc))
                info["data"] = dict(n_events=int(len(w)), n_outside_r_range=D["n_out"])
            else:
                S, S2 = D["S"], D["S2"]
                O, V = coarse_mass(S / dr), coarse_mass(S2 / dr)
                Win, W2 = S.sum(), S2.sum()
                mlp = float(S @ lnp / Win)
                mass = S
                info["data"] = dict(enclosing_hist_bin=list(D["enclosing"]))
            neff = Win ** 2 / W2 if W2 > 0 else 0.0
            dmed = float(quant(mass / dr, [0.5])[0]) if Win > 0 else np.nan
            dmed_e = float(np.sqrt(0.25 / max(neff, 1)) / max(np.interp(dmed, rc, ref), 1e-12))
            zl = (mlp - Elp) / (sdlp / np.sqrt(neff)) if neff > 0 else np.nan
            info["data"].update(sum_w=float(Win), n_eff=float(neff), median=dmed, median_err=dmed_e,
                                mean_lnp=mlp, expected_lnp=Elp, expected_lnp_err=float(sdlp / np.sqrt(max(neff, 1))),
                                z_lnp=float(zl))
        rows.append(info)
        tag = "_sub%d" % a.subsample if a.subsample and D is not None and D["kind"] == "events" else ""

        # ---- independent local amplitude fit to the events of this bin
        LF = None
        if a.local_fit:
            if D is None or D["kind"] != "events" or Win <= 0:
                print("  --local-fit needs events in the bin (--files or --events); skipped")
            else:
                LF = local_fit(mod, th, cov, float(np.exp(xbar)), D)
                info["local_fit"] = {k: ([[float(z.real), float(z.imag)] for z in v] if np.iscomplexobj(v) else
                                         v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in LF.items() if k != "dens"}
                local_rows.setdefault(ie, []).append(dict(LF, lo=lo, hi=hi, lnpt=xbar))

        # ---- figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 6.8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2]))
        lab_e = "%.1f < |%s| < %.1f" % (e0, var, e1)
        for col, logy in enumerate((False, True)):
            ax, axr = axs[0, col], axs[1, col]
            ax.fill_between(Ec, dens_c - dens_e, dens_c + dens_e, step="mid", color=COL["glob_dens"], alpha=0.3, lw=0,
                            label="global model, $\\pm1\\sigma$ (fit covariance)")
            ax.plot(rc[k0:k1], mix[k0:k1], color=COL["glob_dens"], lw=1.6,
                    label="global model, mixture over bin (%s)" % spec_lab)
            ax.plot(rc[k0:k1], point[k0:k1], color="k", lw=0.9, ls="--",
                    label="$p(r\\,|\\,\\langle p_T\\rangle)$, max. rel. diff. %.1e" % dpoint)
            if LF is not None:
                ax.plot(rc[k0:k1], LF["dens"][k0:k1], color=COL["loc_dens"], lw=1.6,
                        label="independent amplitude fit to this bin only (%d par.)" % LF["n_par"])
            if D is not None and Win > 0:
                dw = E_[1] - E_[0]
                y, ey = O / (Win * dw), np.sqrt(V) / (Win * dw)
                m = O > 0
                ax.errorbar(Ec[m], y[m], ey[m], fmt="ko", ms=3, capsize=0, label=D["label"] + ", $n_{eff}$ = %.0f" % neff)
                if D["kind"] == "events" and len(D["r"]) <= 300 and not logy:   # few events: show each one
                    ax.plot(D["r"], np.zeros_like(D["r"]), "|", color="0.3", ms=12, mew=0.8, clip_on=False)
                # relative residual data/model - 1 per r bin with data (convention of
                # compare_global_cells.py), against the global model and (--local-fit) the local fit,
                # in the colours of the curves above; side by side within each r bin
                models = [(ref, COL["glob_dens"], "o", "data / global model $-$ 1")]
                if LF is not None:
                    models.append((LF["dens"], COL["loc_dens"], "s", "data / local fit $-$ 1"))
                for i_m, (dm, cl, mk, lb) in enumerate(models):
                    Ex = Win * coarse_mass(dm)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        res = np.where(m & (Ex > 0), O / Ex - 1, np.nan)
                    sh = (i_m - 0.5 * (len(models) - 1)) * 0.3 * dw
                    axr.plot(Ec + sh, res, mk, color=cl, mec="k", mew=0.4, ms=4, label=lb, zorder=3 + i_m)
                if not logy:
                    axr.legend(fontsize=7, loc="upper left", ncol=len(models), framealpha=0.85)
            axr.axhline(0, color="gray", lw=0.8)
            axr.set_ylim(-a.res_max, a.res_max); axr.grid(alpha=0.3)
            axr.set_ylabel("data/model $-$ 1" if D is None or D["kind"] == "events" else "data/model $-$ 1\n(model over hist. bin)")
            axr.set_xlabel("response $p_T^{rec}/p_T^{gen}$")
            ax.grid(alpha=0.3); ax.set_ylabel("density (conditional on $r \\in [0, 2)$)")
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(max(mix[k0:k1].min(), 1e-4 * mix.max()), 3 * mix.max())
            else:
                ax.set_ylim(bottom=0)
                ax.legend(fontsize=7.5, loc="upper left")
        txt_mod = ["model: median %.5f $\\pm$ %.1e" % (val[0], val_e[0]),
                   "model: $\\sigma_{68}$/median %.5f $\\pm$ %.1e" % (val[1], val_e[1])]
        if extrap:
            txt_mod.append("outside the fitted $p_T$ range: extrapolation")
        txt_dat = []
        if D is not None:
            dd = info["data"]
            if D["kind"] == "events":
                txt_dat.append("data in %s-%s GeV: %d events, $\\Sigma w$ = %.4g, $n_{eff}$ = %.1f"
                               % (ptf(lo), ptf(hi),dd["n_events"], dd["sum_w"], dd["n_eff"]))
            else:
                txt_dat.append("data (enclosing hist. bin): $\\Sigma w$ = %.4g, $n_{eff}$ = %.1f"
                               % (dd["sum_w"], dd["n_eff"]))
        if D is not None and Win > 0:
            txt_dat += ["data: median %.4f $\\pm$ %.4f" % (dd["median"], dd["median_err"]),
                       "$\\langle\\ln p\\rangle$: data %.3f, model %.3f $\\pm$ %.3f ($z$ = %.1f)"
                       % (dd["mean_lnp"], Elp, dd["expected_lnp_err"], dd["z_lnp"])]
        if LF is not None:
            txt_mod.append("local fit (%d par.): 2$\\Delta$lnL(local$-$global) = %.1f%s"
                           % (LF["n_par"], LF["two_dlnl_neff"], "" if LF["converged"] else ", NOT converged"))
        fig.suptitle("%s < $p_T^{gen}$ < %s GeV, %s  (global model, K=%d: evaluated, not refitted)"
                     % (ptf(lo), ptf(hi), lab_e, mod.K), fontsize=11)
        # summary in a strip below the panels (model under the left column, data under the right),
        # so it can never hide data, whatever the shape of the distribution or the statistics
        nl = max(len(txt_mod), len(txt_dat))
        fig.tight_layout(rect=(0, 0.025 + 0.02 * nl, 1, 1))
        box = dict(boxstyle="round", fc="white", lw=0.5)
        for ax_col, t in ((axs[1, 0], txt_mod), (axs[1, 1], txt_dat)):
            if t:
                fig.text(ax_col.get_position().x0, 0.012, "\n".join(t), va="bottom", ha="left", fontsize=8, bbox=box)
        name = "narrow_eta%d_pt%s-%s%s.png" % (ie, ptf(lo), ptf(hi), tag)
        fig.savefig(os.path.join(outdir, name), dpi=120)
        plt.close(fig)
        print("|%s| bin %d, %s-%s GeV: %s" % (var, ie, ptf(lo), ptf(hi), name), flush=True)

        # ---- coefficient figure: global c_n(pt) curves and the local fit of this bin
        if LF is not None:
            J = mod.J
            xg = np.linspace(mod.x0, mod.x1, 80)
            cgl, _, _, ere, eim = global_coefs(mod, th, cov, np.exp(xg))
            nc_ = int(np.ceil((J + 1) / 3))
            fig, axc = plt.subplots(nc_, 3, figsize=(12, 2.6 * nc_ + 0.8), sharex=True, squeeze=False)
            for n in range(J + 1):
                ax = axc.flat[n]
                for part, col, ls, e, nm in ((np.real, COL["glob_re"], "-", ere, "Re"), (np.imag, COL["glob_im"], "--", eim, "Im")):
                    yv = part(cgl[:, n])
                    ax.fill_between(xg, yv - e[:, n], yv + e[:, n], color=col, alpha=0.3, lw=0)
                    ax.plot(xg, yv, color=col, lw=1.6, ls=ls, label="global, " + nm)
                ax.axvspan(np.log(lo), np.log(hi), color="0.6", alpha=0.4, lw=0)
                ax.axvline(xbar, color="0.5", lw=0.6, ls=":")
                ax.errorbar(xbar - 0.03, LF["c"][n].real, LF["c_re_err"][n], fmt="o", color=COL["loc_re"],
                            mec="k", mew=0.6, ms=6, elinewidth=1.4, capsize=2.5, zorder=5, label="local fit, Re")
                ax.errorbar(xbar + 0.03, LF["c"][n].imag, LF["c_im_err"][n], fmt="s", color=COL["loc_im"],
                            mec="k", mew=0.6, ms=6, elinewidth=1.4, capsize=2.5, zorder=5, label="local fit, Im")
                ax.axhline(0, color="gray", lw=0.6)
                ax.set_ylim(-1.05, 1.05)                    # ||c|| = 1: |Re c_n|, |Im c_n| <= 1
                ax.set_title("$c_{%d}$" % n, fontsize=10); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
            for ax in axc.flat[J + 1:]:
                ax.set_visible(False)
            for ax in axc[-1]:
                ax.set_xlabel("ln($p_T^{gen}$ / GeV)", fontsize=9)
            axc.flat[0].legend(fontsize=7, loc="lower left")
            warn = ("\n$n_{eff}$ < number of parameters: the local fit is not determined by the data, its "
                    "(asymptotic) errors are not reliable" if LF["n_eff"] < LF["n_par"] else "")
            fig.suptitle("amplitudes $c_n$ (phase: $c_0$ real), %s: global model (curves, $\\pm1\\sigma$) and an independent "
                         "fit to the %d events in %s-%s GeV only\n($n_{eff}$ = %.1f, reference $\\mu$, $\\sigma$ fixed to the "
                         "global values at $\\langle p_T\\rangle$ = %.4g GeV; 2$\\Delta$lnL(local$-$global) = %.1f for %d par.)%s"
                         % (lab_e, info["data"]["n_events"], ptf(lo), ptf(hi), LF["n_eff"], np.exp(xbar),
                            LF["two_dlnl_neff"], LF["n_par"], warn), fontsize=10, color="k")
            fig.tight_layout(rect=(0, 0, 1, 0.92 if warn else 0.94))
            fig.savefig(os.path.join(outdir, name.replace(".png", "_coef.png")), dpi=110)
            plt.close(fig)

# ------------------------------------------------------------------ local fits against n_eff
for ie, LR in local_rows.items():
    if len(LR) < 2:
        continue
    J = len(LR[0]["c"]) - 1
    LR = sorted(LR, key=lambda q: q["n_eff"])
    ne = np.array([q["n_eff"] for q in LR])
    npar = LR[0]["n_par"]
    fig, axc = plt.subplots(int(np.ceil((J + 1) / 3)), 3, figsize=(12, 2.6 * np.ceil((J + 1) / 3) + 0.8),
                            sharex=True, squeeze=False)
    for n in range(J + 1):
        ax = axc.flat[n]
        for k, (part, cl, cg_, mk) in enumerate(((np.real, COL["loc_re"], COL["glob_re"], "o"),
                                                 (np.imag, COL["loc_im"], COL["glob_im"], "s"))):
            sh = 1.06 if k else 1 / 1.06                   # Re left, Im right of each n_eff
            yl = np.array([part(q["c"][n]) for q in LR])
            el = np.array([q["c_re_err" if k == 0 else "c_im_err"][n] for q in LR])
            yg = np.array([part(q["c_glob"][n]) for q in LR])
            eg = np.array([q["c_glob_re_err" if k == 0 else "c_glob_im_err"][n] for q in LR])
            ax.errorbar(ne * sh, yg, eg, fmt="_", color=cg_, ms=16, mew=2.5, capsize=0, zorder=3,
                        label="global at the bin's $p_T$, %s" % ("Re" if k == 0 else "Im"))
            ax.errorbar(ne * sh, yl, el, fmt=mk, color=cl, mec="k", mew=0.6, ms=5.5, elinewidth=1.4, capsize=2.5,
                        zorder=5, label="local fit, %s" % ("Re" if k == 0 else "Im"))
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xscale("log"); ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(min(ne.min(), npar) / 2, ne.max() * 2)
        ax.axvspan(ax.get_xlim()[0], npar, color="0.85", lw=0, zorder=0,
                   label="$n_{eff}$ < %d par.: local fit not determined" % npar)
        ax.set_title("$c_{%d}$" % n, fontsize=10); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    for ax in axc.flat[J + 1:]:
        ax.set_visible(False)
    for ax in axc[-1]:
        ax.set_xlabel("$n_{eff}$ of the bin", fontsize=9)
    axc.flat[0].legend(fontsize=7, loc="lower left")
    ea, eb = eta_range(FITS[ie])
    fig.suptitle("amplitudes $c_n$ (phase: $c_0$ real), %.1f < |%s| < %.1f: independent fit to each bin's events "
                 "(points) vs the global model at the same $p_T$ (bars)\nbins: %s GeV"
                 % (ea, var, eb, ", ".join("%s-%s" % (ptf(q["lo"]), ptf(q["hi"])) for q in LR)), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(outdir, "narrow_local_vs_neff_eta%d.png" % ie), dpi=110)
    plt.close(fig)
    print("|%s| bin %d: narrow_local_vs_neff_eta%d.png" % (var, ie, ie), flush=True)

# ------------------------------------------------------------------ text and json
L = ["# global model %s in narrow pt_gen bins; densities conditional on r in [0, 2)" % os.path.abspath(a.glob),
     "# med/s68: model median and sigma_68/median (+- delta method); data median; <ln p> test: z = (data - model)/err",
     "# 2dlnL(loc): 2 (lnL_local - lnL_global) in n_eff units of the independent local amplitude fit (--local-fit)",
     "%-9s %-17s %8s %9s %18s %18s %16s %7s %11s %s" % ("|eta|", "pt_gen", "events", "n_eff", "model median",
                                                        "model s68/med", "data median", "z(lnp)", "2dlnL(loc)", "note")]
for c in rows:
    d, m, lf = c.get("data", {}), c["model"], c.get("local_fit", {})
    L.append("%4.1f-%-4.1f %-17s %8s %9.1f %9.5f+-%.1e %9.5f+-%.1e %8.4f+-%.4f %7.2f %11s %s" % (
        c["abs_eta"][0], c["abs_eta"][1], "%s-%s" % (ptf(c["pt_gen"][0]), ptf(c["pt_gen"][1])), d.get("n_events", "hist" if d else "-"),
        d.get("n_eff", np.nan), m["median"], m["median_err"], m["s68_over_median"], m["s68_over_median_err"],
        d.get("median", np.nan), d.get("median_err", np.nan), d.get("z_lnp", np.nan),
        "%.1f/%d" % (lf["two_dlnl_neff"], lf["n_par"]) if lf else "-",
        " ".join(t for t in ("extrapolated" if c["extrapolated"] else "",
                             "local fit not converged" if lf and not lf["converged"] else "") if t)))
print("\n".join(L))
with open(os.path.join(outdir, "narrow_bins.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(outdir, "narrow_bins.json"), "w") as fh:
    json.dump(dict(global_fits=os.path.abspath(a.glob), bins=rows), fh, indent=1)
print("\noutputs in", outdir)
