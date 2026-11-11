#!/usr/bin/env python3
"""Compare the global (pt-smooth) amplitude fit of fit_response_global.py with the per-cell
amplitude and double-sided Crystal Ball (DSCB) fits of plot_response_all_bins.py.  No fitting:
every model is only evaluated.

    python3 scripts/compare_global_cells.py \
        --global ../../../figs/global_polynomial/global_fits.json \
        --amp-fits ../../../figs/plot_response_all_bins_amp/fits.json \
        --dscb-fits ../../../figs/plot_response_all_bins_crystal/fits.json

Inputs (all read-only; nothing is written next to them except the new --outdir):
  --global     global_fits.json of fit_response_global.py
  --hist       histograms2d.npz of the same run (default: next to --global): the data
  --amp-fits   fits.json of plot_response_all_bins.py with the amplitude fit (c_re, c_im, mu_ref, sigma_ref)
  --dscb-fits  fits.json of plot_response_all_bins.py with the DSCB fit ("dscb" entries); if it also
               holds "amplitude" entries, --amp-fits may be left out
Only unmerged cells (|eta| range = one |eta| bin of the global fit) are compared.

Common ground for all three models, the data of a cell = the fine pt bins of histograms2d.npz whose
mean pt_gen lies in the cell's pt_gen range, cross-section weighted (as in the cell fits), and
r in the histogram range [0, 2):
  * every density is taken conditional on r in [0, 2) (the DSCB was fitted that way; the
    amplitude densities lose < 1e-3 outside), normalised numerically on the 1000 fine r bins;
  * global model in a cell: p(r | pt_j) for every fine pt bin j, conditional per bin; the cell
    density is the mixture over j with the in-range data weight of each bin.  The cell models do
    not know pt inside the cell; the global model does, which is part of its advantage;
  * lnL = sum_k S_k log p(r_k) (S_k: summed weight in fine r bin k), quoted as
    n_eff * lnL / sum w ("n_eff*lnL") and as differences to the cell amplitude fit;
  * chi2 on --bins coarse r bins with data, variance max(sum w^2, E * wbar): E the model's
    expected weight, wbar = sum w^2 / sum w over the cell (Pearson-like: a bin the model fills but
    the data barely do cannot get a near-zero variance).
The cell fits were made on their own histograms (other pt binning at the cell edges), so their
lnL here differs slightly from the value stored in fits.json (shown in the table as a check).

Outputs in --outdir (default: <dir of --global>/compare_cells):
  comparison.txt / .json        per cell: n_eff, lnL of all models, chi2/bins, median, sigma68
  compare_eta<i>.png            median, sigma68/median, mean, standardised quantiles vs ln pt_gen:
                                global curve (+-1 sigma band), data, global mixture, cell amplitude,
                                cell DSCB; pull panels (model - data)/error below
  compare_lnl.png               Delta n_eff*lnL (global - amplitude, DSCB - amplitude) and chi2/bins
  cells/compare_eta<i>_pt<NN>.png   (--cell-plots) data and the three densities, residuals
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from response_models import BornModel, hermite_basis, log_normal

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", default="fig/global_receta/global_fits.json")
ap.add_argument("--hist", default=None, help="histograms2d.npz (default: next to --global)")
ap.add_argument("--amp-fits", default=None, help="fits.json with the cell amplitude fits")
ap.add_argument("--dscb-fits", default=None, help="fits.json with the cell DSCB fits")
ap.add_argument("--bins", type=int, default=100, help="coarse r bins for chi2 and the cell plots")
ap.add_argument("--cell-plots", action=argparse.BooleanOptionalAction, default=True)
ap.add_argument("--res-max", type=float, default=0.5, help="y range +-res-max of the residual panels")
ap.add_argument("--no-band", action="store_true", help="skip the delta-method band of the global curve")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

gdir = os.path.dirname(os.path.abspath(a.glob))
outdir = a.outdir or os.path.join(gdir, "compare_cells")
os.makedirs(outdir, exist_ok=True)

# ------------------------------------------------------------------ inputs (read only)
with open(a.glob) as fh:
    G = json.load(fh)
with np.load(a.hist or os.path.join(gdir, "histograms2d.npz")) as z:
    H = {k: z[k] for k in ("eta_edges", "lpt_edges", "r_edges", "sw", "sw2", "W", "SP", "var")}
var = str(H["var"])
ETA, R = H["eta_edges"], H["r_edges"]
rc, dr = 0.5 * (R[1:] + R[:-1]), R[1] - R[0]
nr = len(rc)
if nr % a.bins:
    raise SystemExit("--bins must divide the %d fine r bins" % nr)
grp = nr // a.bins
Rc = R[::grp]
rcc = 0.5 * (Rc[1:] + Rc[:-1])


def load_cells(path, kind):
    if not path:
        return []
    with open(path) as fh:
        out = []
        for e in json.load(fh)["fits"]:
            d = e if (kind == "amp" and "c_re" in e) else e.get("amplitude" if kind == "amp" else "dscb")
            if d is not None:
                out.append((tuple(e["pt_gen"]), tuple(e["abs_eta"]), d))
        return out


amp_list = load_cells(a.amp_fits, "amp") or load_cells(a.dscb_fits, "amp")
dscb_list = load_cells(a.dscb_fits, "dscb")
if not amp_list and not dscb_list:
    raise SystemExit("no cell fits: pass --amp-fits and/or --dscb-fits")


# ------------------------------------------------------------------ cell densities (as plot_response_all_bins.py)
def amp_pdf(d, r):
    c = np.asarray(d["c_re"]) + 1j * np.asarray(d["c_im"])
    if not (np.all(np.isfinite(c)) and np.isfinite(d["mu_ref"]) and d["sigma_ref"] > 0):
        return None
    m, s = d["mu_ref"], d["sigma_ref"]
    return np.exp(log_normal(r, m, s)) * np.abs(hermite_basis((r - m) / s, len(c) - 1) @ c) ** 2


def dscb_logf(z, aL, nL, aR, nR):
    out = -0.5 * z * z
    lo, hi = z < -aL, z > aR
    out[lo] = -0.5 * aL * aL - nL * np.log1p(-aL * (z[lo] + aL) / nL)
    out[hi] = -0.5 * aR * aR - nR * np.log1p(aR * (z[hi] - aR) / nR)
    return out


def dscb_pdf(d, r):
    p = [d[k] for k in ("mu", "sigma", "alpha_L", "n_L", "alpha_R", "n_R")]
    if not (np.all(np.isfinite(p)) and p[1] > 0):
        return None
    return np.exp(dscb_logf((r - p[0]) / p[1], *p[2:]))     # normalised below, on [0, 2)


def cond(p):
    """density on the fine r centres -> conditional on [0, 2) (rows or a vector)."""
    return p / (p.sum(-1, keepdims=True) * dr)


PQ = np.array([0.001, 0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99, 0.999])
ZG = np.array([-3.0902, -2.3263, -1.6449, -0.9945, 0.0, 0.9945, 1.6449, 2.3263, 3.0902]) / 0.9945
TAILS = (0, 1, 2, 6, 7, 8)
PANELS = ["median $Q_{50}$", "$\\sigma_{68}$ / median", "mean"] + \
         ["$(Q_{%g} - Q_{50})/\\sigma_{68}$" % (100 * PQ[i]) for i in TAILS]
GAUSS = [None, None, None] + [ZG[i] for i in TAILS]


def quantiles(mass):
    """mass per fine r bin (any normalisation) -> the PQ quantiles, linear inside the bins."""
    cdf = np.r_[0.0, np.cumsum(mass)]
    return np.interp(PQ * cdf[-1], cdf, R)


def derived(mass):
    Q = quantiles(mass)
    med, s68 = Q[4], 0.5 * (Q[5] - Q[3])
    return np.r_[med, s68 / med, mass @ rc / mass.sum(), [(Q[i] - med) / s68 for i in TAILS]]


# ------------------------------------------------------------------ per |eta| bin and cell
rows, per_eta = [], []
for f in G["fits"]:
    ie = f["eta_bin"]
    lo_e, hi_e = ETA[ie], ETA[ie + 1]
    mod, th = BornModel.from_dict(f["model"])
    cov = np.asarray(f["cov"])
    W = H["W"][ie]
    ok = W > 0
    ptj = np.where(ok, H["SP"][ie] / np.where(ok, W, 1.0), np.nan)
    sw, sw2 = H["sw"][ie], H["sw2"][ie]
    same = lambda ae: np.isclose(ae[0], lo_e) and np.isclose(ae[1], hi_e)
    amp_c = {pt: d for pt, ae, d in amp_list if same(ae)}
    cb_c = {pt: d for pt, ae, d in dscb_list if same(ae)}
    cells = []
    for pt in sorted(set(amp_c) | set(cb_c)):
        js = np.flatnonzero(ok & (ptj >= pt[0]) & (ptj < pt[1]) & (sw.sum(1) > 0))
        if len(js) == 0:
            continue
        S, S2 = sw[js].sum(0), sw2[js].sum(0)
        Win = S.sum()
        neff = Win ** 2 / S2.sum()
        Pj = cond(mod.pdf_grid(th, ptj[js], rc))                     # (len(js), nr), each row conditional
        Wj = sw[js].sum(1)
        mix = (Wj @ Pj) / Wj.sum()
        models = {"global": (mix, float(np.sum(sw[js] * np.log(np.maximum(Pj, 1e-300)))))}
        for name, dd, fn in (("amp", amp_c.get(pt), amp_pdf), ("dscb", cb_c.get(pt), dscb_pdf)):
            p = fn(dd, rc) if dd is not None else None
            if p is not None and p.sum() > 0:
                p = cond(p)
                models[name] = (p, float(S @ np.log(np.maximum(p, 1e-300))))
        # data summaries and errors (density of the global mixture at the quantile)
        dat = derived(S)
        Qd = quantiles(S)
        dens = np.interp(Qd, rc, mix)
        eQ = np.sqrt(PQ * (1 - PQ) / neff) / np.maximum(dens, 1e-12)
        med, s68 = dat[0], 0.5 * (Qd[5] - Qd[3])
        err = np.r_[eQ[4], 0.5 * np.hypot(eQ[5], eQ[3]) / med, np.sqrt(S2 @ (rc - dat[2]) ** 2) / Win,
                    [eQ[i] / s68 for i in TAILS]]
        # chi2 on the coarse bins
        O, V = S.reshape(a.bins, grp).sum(1), S2.reshape(a.bins, grp).sum(1)
        use = V > 0
        wbar = S2.sum() / Win
        res = dict(eta_bin=ie, abs_eta=[float(lo_e), float(hi_e)], pt_gen=list(pt), n_eff=float(neff), sum_w=float(Win),
                   lnpt=float(np.log(np.average(ptj[js], weights=Wj))), n_fine_pt_bins=int(len(js)),
                   extrap_frac=float(Wj[(np.log(ptj[js]) < mod.x0 - 1e-9) | (np.log(ptj[js]) > mod.x1 + 1e-9)].sum() / Wj.sum()),
                   data=dict(values=dat.tolist(), errors=err.tolist()), models={})
        for name, (p, lnl) in models.items():
            E = Win * (p * dr).reshape(a.bins, grp).sum(1)
            chi2 = float(np.sum((O[use] - E[use]) ** 2 / np.maximum(V[use], E[use] * wbar)))
            md = dict(lnL=lnl, neff_lnL=neff / Win * lnl, chi2=chi2, nbins=int(use.sum()),
                      values=derived(p).tolist())
            if name in ("amp", "dscb"):
                dd = (amp_c if name == "amp" else cb_c)[pt]
                md["lnL_per_w_stored"] = dd.get("lnL_per_w", dd.get("lnL_per_w_amp"))
            res["models"][name] = md
        M_ = res["models"]
        if "amp" in M_:
            for name in ("global", "dscb"):
                if name in M_:
                    res["dneff_lnL_%s_minus_amp" % name] = M_[name]["neff_lnL"] - M_["amp"]["neff_lnL"]
        res["_plot"] = dict(js=js, S=S, S2=S2, dens={k: v[0] for k, v in models.items()})
        cells.append(res)

    # global curve along pt with the delta-method band, conditional on [0, 2)
    ptg = np.exp(np.linspace(mod.x0, mod.x1, 60))
    curve = lambda t: np.array([derived(p) for p in cond(mod.pdf_grid(t, ptg, rc))]).T
    cur = curve(th)
    band = np.zeros_like(cur)
    if not a.no_band:
        Jm = np.zeros((cur.size, len(th)))
        for i in range(len(th)):
            h = 1e-4 * max(1.0, abs(th[i]))
            e = np.zeros(len(th)); e[i] = h
            Jm[:, i] = ((curve(th + e) - curve(th - e)) / (2 * h)).ravel()
        band = np.sqrt(np.maximum(np.einsum("ip,pq,iq->i", Jm, cov, Jm), 0)).reshape(cur.shape)
    per_eta.append(dict(ie=ie, lab="%.2f < |%s| < %.2f" % (lo_e, var, hi_e), K=mod.K, ptg=ptg, cur=cur, band=band,
                        cells=cells))
    rows += cells
    print("|%s| bin %d: %d cells compared" % (var, ie, len(cells)), flush=True)

# ------------------------------------------------------------------ text and json
lab_m = {"global": "global", "amp": "cell amp", "dscb": "cell DSCB"}
L = ["# global pt-smooth amplitude fit vs the cell fits of plot_response_all_bins.py, on the data of",
     "# histograms2d.npz (cross-section weights), densities conditional on r in [%g, %g)" % (R[0], R[-1]),
     "# n_eff*lnL = n_eff/sum w * sum_k S_k log p(r_k);  dG, dCB: global, DSCB minus cell amplitude (> 0: better)",
     "# chi2/nb: chi2 with variance max(sum w^2, E wbar) on %d coarse r bins / number of bins with data" % a.bins,
     "# stored: lnL/w of the amplitude fit in fits.json (its own histogram) vs here",
     "%-11s %-12s %10s %10s %10s %18s %9s %9s %9s %8s %8s %8s %6s" % (
         "|%s|" % var, "pt_gen", "n_eff", "dG", "dCB", "amp lnL/w here|st", "chi2 G", "chi2 amp", "chi2 CB",
         "med dat", "med G", "s68/m G", "extr")]
for c in rows:
    M_ = c["models"]
    g = lambda n, k: M_[n][k] if n in M_ else np.nan
    cn = lambda n: g(n, "chi2") / g(n, "nbins") if n in M_ else np.nan
    st = M_.get("amp", {}).get("lnL_per_w_stored")
    L.append("%4.2f-%-6.2f %5.0f-%-6.0f %10.0f %10.1f %10.1f %8.4f|%-9s %9.2f %9.2f %9.2f %8.4f %8.4f %8.4f %5.0f%%" % (
        c["abs_eta"][0], c["abs_eta"][1], c["pt_gen"][0], c["pt_gen"][1], c["n_eff"],
        c.get("dneff_lnL_global_minus_amp", np.nan), c.get("dneff_lnL_dscb_minus_amp", np.nan),
        g("amp", "lnL") / c["sum_w"] if "amp" in M_ else np.nan, "%.4f" % st if st is not None else "-",
        cn("global"), cn("amp"), cn("dscb"), c["data"]["values"][0], g("global", "values")[0],
        g("global", "values")[1], 100 * c["extrap_frac"]))
tot = {k: np.nansum([c.get(k, np.nan) for c in rows]) for k in ("dneff_lnL_global_minus_amp", "dneff_lnL_dscb_minus_amp")}
L.append("# sum over cells: dG %.1f, dCB %.1f" % (tot["dneff_lnL_global_minus_amp"], tot["dneff_lnL_dscb_minus_amp"]))
print("\n".join(L))
with open(os.path.join(outdir, "comparison.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(outdir, "comparison.json"), "w") as fh:
    json.dump(dict(var=var, panels=PANELS, global_fits=os.path.abspath(a.glob),
                   amp_fits=a.amp_fits and os.path.abspath(a.amp_fits),
                   dscb_fits=a.dscb_fits and os.path.abspath(a.dscb_fits),
                   cells=[{k: v for k, v in c.items() if k != "_plot"} for c in rows]), fh, indent=1)

# ------------------------------------------------------------------ plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STY = {"global": dict(color="C3", marker="s", mfc="none", label="global, mixture over cell"),
       "amp": dict(color="C0", marker="^", label="cell amplitude"),
       "dscb": dict(color="C2", marker="D", ms=4, label="cell DSCB")}
OFF = {"global": -0.025, "amp": 0.0, "dscb": 0.025}

for E in per_eta:
    cells = E["cells"]
    if not cells:
        continue
    x = np.array([c["lnpt"] for c in cells])
    dv = np.array([c["data"]["values"] for c in cells])
    de = np.array([c["data"]["errors"] for c in cells])
    fig = plt.figure(figsize=(13.5, 11.5))
    outer = fig.add_gridspec(3, 3, hspace=0.32, wspace=0.27)
    xg = np.log(E["ptg"])
    for p in range(9):
        inner = outer[p // 3, p % 3].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(inner[0]); axp = fig.add_subplot(inner[1], sharex=ax)
        ax.fill_between(xg, E["cur"][p] - E["band"][p], E["cur"][p] + E["band"][p], color="C3", alpha=0.25, lw=0)
        ax.plot(xg, E["cur"][p], color="C3", lw=1.2, label="global fit (K=%d)" % E["K"])
        ax.errorbar(x, dv[:, p], de[:, p], fmt="ko", ms=3, capsize=1.5, label="data", zorder=5)
        for name, st in STY.items():
            v = np.array([c["models"][name]["values"][p] if name in c["models"] else np.nan for c in cells])
            ax.plot(x + OFF[name], v, ls="none", ms=st.get("ms", 4.5), **{k: s for k, s in st.items() if k != "ms"})
            axp.plot(x + OFF[name], (v - dv[:, p]) / de[:, p], ls="none", ms=3.5,
                     **{k: s for k, s in st.items() if k not in ("ms", "label")})
        if GAUSS[p] is not None:
            ax.axhline(GAUSS[p], color="gray", ls="--", lw=0.8)
        axp.axhline(0, color="gray", lw=0.8)
        for v in (-2, 2):
            axp.axhline(v, color="gray", lw=0.6, ls=":")
        ax.set_title(PANELS[p], fontsize=10)
        ax.grid(alpha=0.3); axp.grid(alpha=0.3)
        ax.tick_params(labelsize=8); axp.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), visible=False)
        if p % 3 == 0:
            axp.set_ylabel("pull", fontsize=8)
        if p >= 6:
            axp.set_xlabel("ln($p_T^{gen}$ / GeV)")
        if p == 0:
            ax.legend(fontsize=7)
    fig.suptitle("global vs cell fits, %s  (r in [%g, %g); dashed: Gaussian; pull = (model - data)/error)"
                 % (E["lab"], R[0], R[-1]), y=0.995)
    fig.savefig(os.path.join(outdir, "compare_eta%d.png" % E["ie"]), dpi=115, bbox_inches="tight")
    plt.close(fig)

# --- lnL and chi2 overview
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for k, E in enumerate(per_eta):
    cells = E["cells"]
    if not cells:
        continue
    x = np.array([c["lnpt"] for c in cells])
    col = plt.cm.viridis(k / max(len(per_eta) - 1, 1))
    for ax, key in zip(axes[:2], ("dneff_lnL_global_minus_amp", "dneff_lnL_dscb_minus_amp")):
        ax.plot(x, [c.get(key, np.nan) for c in cells], "o-", color=col, ms=3.5, lw=1, label=E["lab"])
    for name, mk in (("global", "s"), ("amp", "^"), ("dscb", "D")):
        y = [c["models"][name]["chi2"] / c["models"][name]["nbins"] if name in c["models"] else np.nan for c in cells]
        axes[2].plot(x, y, mk, color=col, ms=4, mfc="none" if name == "global" else col)
for ax, t in zip(axes, ("global - cell amplitude", "cell DSCB - cell amplitude", "$\\chi^2$ / bins")):
    ax.set_title(t); ax.grid(alpha=0.3); ax.set_xlabel("ln($p_T^{gen}$ / GeV)")
for ax in axes[:2]:
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("$\\Delta\\, n_{eff} \\ln L$")
axes[2].set_yscale("log")
axes[2].plot([], [], "ks", mfc="none", label="global"); axes[2].plot([], [], "k^", label="cell amplitude")
axes[2].plot([], [], "kD", label="cell DSCB"); axes[2].legend(fontsize=7)
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "compare_lnl.png"), dpi=120); plt.close(fig)

# --- the cells: data and the three densities
if a.cell_plots:
    celldir = os.path.join(outdir, "cells")
    os.makedirs(celldir, exist_ok=True)
    names = {"global": ("C3", "-"), "amp": ("C0", "--"), "dscb": ("C2", ":")}
    for E in per_eta:
        for ic, c in enumerate(E["cells"]):
            P = c["_plot"]
            S, S2 = P["S"].reshape(a.bins, grp).sum(1), P["S2"].reshape(a.bins, grp).sum(1)
            dRc = Rc[1] - Rc[0]
            y, ey = S / (c["sum_w"] * dRc), np.sqrt(S2) / (c["sum_w"] * dRc)
            fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
            for col_, logy in enumerate((False, True)):
                ax, axr = axs[0, col_], axs[1, col_]
                m = y > 0
                ax.errorbar(rcc[m], y[m], ey[m], fmt="ko", ms=2.5, capsize=0, label="data")
                for name, dens in P["dens"].items():
                    cl, ls = names[name]
                    M_ = c["models"][name]
                    ax.plot(rc, dens, color=cl, ls=ls, lw=1.3,
                            label="%s: $\\chi^2$/bins %.2f" % (lab_m[name], M_["chi2"] / M_["nbins"]))
                    pb = dens.reshape(a.bins, grp).mean(1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        axr.plot(rcc[m], y[m] / pb[m] - 1, color=cl, ls="none", marker="o", ms=2.5)
                axr.axhline(0, color="gray", lw=0.8)
                axr.set_ylim(-a.res_max, a.res_max); axr.grid(alpha=0.3)
                axr.set_xlabel("response $p_T^{rec}/p_T^{gen}$"); axr.set_ylabel("data/model - 1")
                if logy:
                    ax.set_yscale("log"); ax.set_ylim(max(y[m].min() * 0.3, 1e-6) if m.any() else 1e-6, None)
                else:
                    ax.legend(fontsize=8)
                ax.grid(alpha=0.3); ax.set_ylabel("density")
            dG, dCB = c.get("dneff_lnL_global_minus_amp", np.nan), c.get("dneff_lnL_dscb_minus_amp", np.nan)
            fig.suptitle("%s, %g < $p_T^{gen}$ < %g GeV, $n_{eff}$ = %.0f;  $\\Delta n_{eff}\\ln L$ vs cell "
                         "amplitude: global %.1f, DSCB %.1f" % (E["lab"], c["pt_gen"][0], c["pt_gen"][1],
                                                                 c["n_eff"], dG, dCB), fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(celldir, "compare_eta%d_pt%02d.png" % (E["ie"], ic)), dpi=100)
            plt.close(fig)

print("\noutputs in", outdir)
