#!/usr/bin/env python3
"""Quantiles and tails of the response: data against the network and the global fit(s), per
(|eta| slice) x (pt_gen bin), on the same rows.

    python3 scripts/quantiles_nn_global.py --model fig/nn_pt_eta_full2/best.pt --global K3=fig/global_K3/global_fits.json
    python3 scripts/quantiles_nn_global.py --model ... --global K3=... --eta-edges 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2

The rows are the network's validation rows (read again with the training's own arguments; --files
overrides, then all rows of those files are used).  In every cell the three distributions are
compared on exactly the same population:

  data      weighted quantiles Q_p of r of the cell's rows (all rows, also r outside [0, 2))
  network   quantiles of the mixture  F(r) = sum_i w_i F_NN(r | x_i) / sum_i w_i  over the cell's
            rows, every row with its own conditions (exact pt and |eta|)
  global    the same mixture of F_global(r | pt_i, |eta| bin of row i)

so differences in the cell's pt or |eta| distribution cannot fake a disagreement, and the
|eta| slices may be finer than the global fit's bins (--eta-edges): the network follows the exact
|eta|, the global fit only its bin.  For the mixtures, at most --model-rows rows per cell are
drawn with probability ~ |w| (then weight sign(w)): the same expectation, little noise.

Quantile levels p = 0.001 0.01 0.05 0.16 0.5 0.84 0.95 0.99 0.999 (as fit_response_global.py).

Outputs in --outdir (default <dir of --model>/quantiles):
  quantiles_eta<k>.png  median, sigma_68/median and the standardised tail quantiles
                        (Q_p - Q_50)/sigma_68 vs ln pt: data (points, statistical errors),
                        network (red), global (blue), Gaussian value (dashed)
  tails.png             tail calibration: the data's fraction below the model's Q_p, divided by
                        p (lower tails), or above it divided by 1 - p (upper tails).  1 = the
                        model puts the right probability into that tail; gray: +-1 sigma statistical
  accuracy_map.png      per (|eta|, pt) cell and model: bias of the median and of sigma_68 against
                        the data (numbers in %, colour = pull) and the chi^2/ndf of the tail
                        calibration: where a model differs from the data beyond statistics
  quantiles.txt/.json   all numbers per cell; per |eta| slice and tail level the chi^2/ndf of the
                        tail-calibration pulls; per |eta| slice the accuracy summary
Inputs are only read.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from jernn import data as jd                                          # noqa: E402
from jernn.diffusion import cdf_from_a, density_coefficients, pdf_from_a                    # noqa: E402
from jernn.features import Cut, Feature                               # noqa: E402
from jernn.model import Predictor                                     # noqa: E402
from response_models import BornModel                                 # noqa: E402

PQ = np.array([0.001, 0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99, 0.999])
ZG = np.array([-3.0902, -2.3263, -1.6449, -0.9945, 0.0, 0.9945, 1.6449, 2.3263, 3.0902]) / 0.9945
TAILS = (0, 1, 2, 6, 7, 8)                                            # indices of the tail levels in PQ

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="checkpoint of train_nn.py")
ap.add_argument("--global", dest="glob", nargs="*", default=[], help="global_fits.json, or label=path (several)")
ap.add_argument("--files", nargs="+", default=None, help="other files (default: the training's, validation split)")
ap.add_argument("--pt-edges", nargs="+", type=float,
                default=[97, 120, 140, 170, 200, 240, 280, 330, 400, 470, 550, 650, 750, 870, 1000, 1200, 1400,
                         1700, 2000, 2600, 3500, 6500])
ap.add_argument("--eta-edges", nargs="+", type=float, default=None,
                help="|eta| slices (default: the global fit's bins, else 0 0.5 ... 2.5 3 5)")
ap.add_argument("--eta-var", default=None, help="|eta| column (default: the global fit's 'var', else receta)")
ap.add_argument("--min-neff", type=float, default=1000, help="cells with fewer effective rows are not shown")
ap.add_argument("--model-rows", type=int, default=5000, help="rows per cell for the model mixtures")
ap.add_argument("--max-rows", type=int, default=None, help="random subset of the evaluation rows")
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

P = Predictor.load(a.model, a.device)
info, A = P.info, P.info["args"]
outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(a.model)), "quantiles")
os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(a.seed)

GL = []
for s in a.glob:
    lab, _, path = s.rpartition("=")
    with open(path) as fh:
        G = json.load(fh)
    fits = sorted(G["fits"], key=lambda f: f["eta_bin"])
    GL.append(dict(label=lab or os.path.basename(os.path.dirname(os.path.abspath(path))), var=G.get("var", "receta"),
                   edges=np.array([f["abs_eta"][0] for f in fits] + [fits[-1]["abs_eta"][1]]),
                   models=[BornModel.from_dict(f["model"]) for f in fits]))
eta_var = a.eta_var or (GL[0]["var"] if GL else "receta")
ETA = np.asarray(a.eta_edges if a.eta_edges else (GL[0]["edges"] if GL else [0, 0.5, 1, 1.5, 2, 2.5, 3, 5]), float)
PT = np.asarray(a.pt_edges, float)

# ------------------------------------------------------------------ rows (read only)
extra = [Feature("genpt:log"), Feature("%s:abs" % eta_var)]
D = jd.read(a.files or info["data"]["files"], P.target, P.conds + extra, [Cut(s) for s in info["cuts"]],
            weight=A["weight"], fraction=A["fraction"] if a.files is None else 1.0,
            max_rows=A["max_rows"] if a.files is None else None, seed=A["seed"])
if a.files is None:
    _, rows = jd.split(len(D["y"]), A["val_frac"], A["seed"])
else:
    rows = np.arange(len(D["y"]))
if a.max_rows and len(rows) > a.max_rows:
    rows = np.sort(rng.choice(rows, a.max_rows, replace=False))
nc = len(P.conds)
y = D["y"][rows].astype(np.float64)
Xn = D["X"][rows, :nc]
lpt, aeta = D["X"][rows, nc].astype(np.float64), D["X"][rows, nc + 1].astype(np.float64)
w = D["w"][rows]
del D
print("%d rows (%s), n_eff %.0f" % (len(y), "validation split" if a.files is None else "all rows", jd.n_eff(w)),
      flush=True)
ie = np.searchsorted(ETA, aeta, side="right") - 1
ip = np.searchsorted(np.log(PT), lpt, side="right") - 1

# ------------------------------------------------------------------ mixtures (closed-form CDF)
def model_rows(mu, sig, c, s):
    """A mixture's rows with their density coefficients computed once: (mu, sigma, a, s)."""
    return mu, sig, density_coefficients(c)[:, None, :], s


def mix_cdf(r, M):
    """F(r) = sum_i s_i F_i(r) / sum_i s_i for model rows M = (mu, sigma, a, s), r (m,)."""
    mu, sig, A, s = M
    return s @ cdf_from_a((r[None, :] - mu[:, None]) / sig[:, None], A) / s.sum()


def mix_pdf(r, M):
    mu, sig, A, s = M
    return s @ (pdf_from_a((r[None, :] - mu[:, None]) / sig[:, None], A) / sig[:, None]) / s.sum()


def mix_quantiles(M, bisect=14, newton=3):
    """Q_p of the mixture, all levels at once: bisection on its exact CDF to bracket (4/2^14 in r),
    then Newton steps with its exact density (quadratic convergence), kept inside the bracket."""
    lo, hi = np.full(len(PQ), -1.0), np.full(len(PQ), 3.0)
    for _ in range(bisect):
        mid = 0.5 * (lo + hi)
        below = mix_cdf(mid, M) < PQ
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    r = 0.5 * (lo + hi)
    for _ in range(newton):
        r = np.clip(r - (mix_cdf(r, M) - PQ) / np.maximum(mix_pdf(r, M), 1e-12), lo, hi)
    return r


def weighted_quantiles(r, ww):
    o = np.argsort(r)
    cw = np.cumsum(ww[o])
    cw = cw / cw[-1]
    return np.array([r[o][min(np.searchsorted(cw, p), len(r) - 1)] for p in PQ])


def frac_below(r, ww, q):
    return np.array([ww[r < x].sum() for x in q]) / ww.sum()


def derived(Q):
    med, s68 = Q[4], 0.5 * (Q[5] - Q[3])
    return np.r_[med, s68 / med, [(Q[i] - med) / s68 for i in TAILS]]


PANELS = ["median $Q_{50}$", "$\\sigma_{68}$ / median"] + ["$(Q_{%g} - Q_{50})/\\sigma_{68}$" % (100 * PQ[i]) for i in TAILS]
GAUSS = [None, None] + [ZG[i] for i in TAILS]

# ------------------------------------------------------------------ cells
cells = []
for k in range(len(ETA) - 1):
    for j in range(len(PT) - 1):
        m = np.flatnonzero((ie == k) & (ip == j))
        if len(m) < 20:
            continue
        ww = w[m]
        ne = jd.n_eff(ww)
        if ne < a.min_neff:
            continue
        # data
        Qd = weighted_quantiles(y[m], ww)
        # model rows: drawn ~ |w|, weight sign(w)
        pw = np.abs(ww) / np.abs(ww).sum()
        sub = m[rng.choice(len(m), min(a.model_rows, len(m)), replace=True, p=pw)] if len(m) > a.model_rows else m
        s = np.sign(w[sub]) if len(m) > a.model_rows else w[sub]
        models = {}
        mu, sig, c = P.params(Xn[sub])
        models["network"] = model_rows(mu, sig, c, s)
        for g in GL:
            ig = np.searchsorted(g["edges"], aeta[sub], side="right") - 1
            if np.any((ig < 0) | (ig >= len(g["models"]))):
                continue
            parts, ok = [np.empty(len(sub)), np.empty(len(sub))], True
            cg_all = None
            for b in np.unique(ig):
                mod, th = g["models"][b]
                mb = ig == b
                if np.mean((lpt[sub][mb] < mod.x0 - 1e-9) | (lpt[sub][mb] > mod.x1 + 1e-9)) > 0.01:
                    ok = False
                    break
                cg, mug, sgg = mod.coefficients(th, np.exp(lpt[sub][mb]))
                if cg_all is None:
                    cg_all = np.zeros((len(sub), cg.shape[1]), complex)
                parts[0][mb], parts[1][mb], cg_all[mb] = mug, sgg, cg
            if ok:
                models[g["label"]] = model_rows(parts[0], parts[1], cg_all, s)
        rec = dict(eta=[float(ETA[k]), float(ETA[k + 1])], pt=[float(PT[j]), float(PT[j + 1])], eta_bin=k,
                   lnpt=float(np.average(lpt[m], weights=np.abs(ww))), rows=int(len(m)), n_eff=float(ne),
                   data=dict(Q=Qd.tolist()), models={})
        # statistical error of the data quantiles: binomial in p, through the model density at Q_p
        fd = np.maximum(mix_pdf(Qd, models["network"]), 1e-9)
        eQ = np.sqrt(PQ * (1 - PQ) / ne) / fd
        med, s68 = Qd[4], 0.5 * (Qd[5] - Qd[3])
        rec["data"]["derived"] = derived(Qd).tolist()
        rec["data"]["err"] = np.r_[eQ[4], 0.5 * np.hypot(eQ[5], eQ[3]) / med, [eQ[i] / s68 for i in TAILS]].tolist()
        for lab, M in models.items():
            Qm = mix_quantiles(M)
            fb = frac_below(y[m], ww, Qm)                     # data fraction below the model's Q_p
            tail = np.where(PQ < 0.5, fb / PQ, (1 - fb) / (1 - PQ))
            etail = np.where(PQ < 0.5, np.sqrt(PQ * (1 - PQ) / ne) / PQ, np.sqrt(PQ * (1 - PQ) / ne) / (1 - PQ))
            sm = 0.5 * (Qm[5] - Qm[3])
            es68 = 0.5 * np.hypot(eQ[5], eQ[3])
            tp = (tail - 1) / etail
            rec["models"][lab] = dict(Q=Qm.tolist(), derived=derived(Qm).tolist(), tail_ratio=tail.tolist(),
                                      tail_err=etail.tolist(), tail_pull=tp.tolist(),
                                      # accuracy against the data: relative bias and pull of the median and
                                      # of sigma_68, chi2/ndf of the tail-calibration pulls (6 tail levels)
                                      median_bias=float(Qm[4] / med - 1), median_pull=float((Qm[4] - med) / eQ[4]),
                                      width_bias=float(sm / s68 - 1), width_pull=float((sm - s68) / es68),
                                      tail_chi2=float(np.mean(tp[list(TAILS)] ** 2)))
        cells.append(rec)
    print("|%s| %.2f-%.2f done (%d cells)" % (eta_var, ETA[k], ETA[k + 1], sum(c["eta_bin"] == k for c in cells)),
          flush=True)

labels = ["network"] + [g["label"] for g in GL]

# ------------------------------------------------------------------ text summary
L = ["# quantiles of r: data vs network (%s) and global fit(s), validation rows%s" %
     (os.path.abspath(a.model), "" if a.files is None else " of --files"),
     "# tail calibration: R = (data fraction below the model's Q_p)/p for p < 0.5, (fraction above)/(1-p) else;",
     "# pull = (R - 1)/sigma_R (binomial).  chi2/ndf over the pt cells of each |eta| slice; ~1: calibrated tail",
     ""]
L.append("%-11s %-9s" % ("|eta|", "model") + "".join("  p=%-6g" % PQ[i] for i in TAILS) + "   (chi2/ndf | mean R)")
summ = {}
for k in range(len(ETA) - 1):
    cc = [c for c in cells if c["eta_bin"] == k]
    if not cc:
        continue
    for lab in labels:
        cl = [c for c in cc if lab in c["models"]]
        if not cl:
            continue
        pulls = np.array([c["models"][lab]["tail_pull"] for c in cl])
        ratios = np.array([c["models"][lab]["tail_ratio"] for c in cl])
        chi = [float(np.mean(pulls[:, i] ** 2)) for i in TAILS]
        mr = [float(np.average(ratios[:, i], weights=[c["n_eff"] for c in cl])) for i in TAILS]
        summ.setdefault("%.2f-%.2f" % (ETA[k], ETA[k + 1]), {})[lab] = dict(chi2ndf=chi, mean_ratio=mr, ncells=len(cl))
        L.append("%4.2f-%-6.2f %-9s" % (ETA[k], ETA[k + 1], lab[:9]) +
                 "".join("  %5.1f|%4.2f" % (x, r) for x, r in zip(chi, mr)))
L += ["", "# accuracy against the data per |eta| slice: chi2/ndf of the median and sigma_68 pulls over the pt cells,",
      "# n_eff-weighted mean |relative bias| of median and sigma_68 (%), mean tail chi2/ndf, worst cell",
      "%-11s %-9s %9s %9s %8s %8s %9s   %s" % ("|eta|", "model", "med chi2", "wid chi2", "|dmed|%", "|dwid|%",
                                             "tail chi2", "worst cell (tail chi2)")]
for k in range(len(ETA) - 1):
    cc = [c for c in cells if c["eta_bin"] == k]
    for lab in labels:
        cl = [c for c in cc if lab in c["models"]]
        if not cl:
            continue
        acc = {key: np.array([c["models"][lab][key] for c in cl])
               for key in ("median_pull", "width_pull", "median_bias", "width_bias", "tail_chi2")}
        ne_ = np.array([c["n_eff"] for c in cl])
        worst = cl[int(np.argmax(acc["tail_chi2"]))]
        L.append("%4.2f-%-6.2f %-9s %9.1f %9.1f %8.2f %8.2f %9.1f   pt %.0f-%.0f (%.1f)" % (
            ETA[k], ETA[k + 1], lab[:9], np.mean(acc["median_pull"] ** 2), np.mean(acc["width_pull"] ** 2),
            100 * np.average(np.abs(acc["median_bias"]), weights=ne_),
            100 * np.average(np.abs(acc["width_bias"]), weights=ne_), np.mean(acc["tail_chi2"]),
            worst["pt"][0], worst["pt"][1], worst["models"][lab]["tail_chi2"]))
print("\n".join(L))
with open(os.path.join(outdir, "quantiles.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(outdir, "quantiles.json"), "w") as fh:
    json.dump(dict(model=os.path.abspath(a.model), PQ=PQ.tolist(), eta_edges=ETA.tolist(), pt_edges=PT.tolist(),
                   eta_var=eta_var, cells=cells, summary=summ), fh, indent=1)

# ------------------------------------------------------------------ plots
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

COL = {"network": "C3"}
for i, g in enumerate(GL):
    COL[g["label"]] = "C%d" % (0 if i == 0 else i + 1)
for k in range(len(ETA) - 1):
    cc = [c for c in cells if c["eta_bin"] == k]
    if not cc:
        continue
    x = np.array([c["lnpt"] for c in cc])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for p, ax in enumerate(axes.flat):
        dv = np.array([c["data"]["derived"][p] for c in cc])
        de = np.array([c["data"]["err"][p] for c in cc])
        for n, lab in enumerate(labels):
            sel = [c for c in cc if lab in c["models"]]
            if sel:
                ax.plot([c["lnpt"] for c in sel], [c["models"][lab]["derived"][p] for c in sel], "-",
                        color=COL[lab], lw=1.4, label=lab)
        ax.errorbar(x, dv, de, fmt="ko", ms=3, capsize=1.5, label="data")
        if GAUSS[p] is not None:
            ax.axhline(GAUSS[p], color="gray", ls="--", lw=0.8)
        ax.set_title(PANELS[p], fontsize=10); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("ln($p_T^{gen}$ / GeV)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("response quantiles, %.2f < |%s| < %.2f  (mixtures over the same rows; dashed: Gaussian)"
                 % (ETA[k], eta_var, ETA[k + 1]))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(outdir, "quantiles_eta%d.png" % k), dpi=115); plt.close(fig)

ks = sorted({c["eta_bin"] for c in cells})
fig, axes = plt.subplots(len(ks), len(TAILS), figsize=(2.9 * len(TAILS), 1.9 * len(ks) + 0.6), sharex=True,
                         squeeze=False)
for r_, k in enumerate(ks):
    cc = [c for c in cells if c["eta_bin"] == k]
    x = np.array([c["lnpt"] for c in cc])
    for col, i in enumerate(TAILS):
        ax = axes[r_, col]
        e = np.array([c["models"]["network"]["tail_err"][i] for c in cc])
        ax.fill_between(x, 1 - e, 1 + e, color="gray", alpha=0.25, lw=0)
        for lab in labels:
            sel = [c for c in cc if lab in c["models"]]
            if sel:
                ax.plot([c["lnpt"] for c in sel], [c["models"][lab]["tail_ratio"][i] for c in sel], "o-", ms=2.5,
                        lw=1, color=COL[lab], label=lab)
        ax.axhline(1, color="k", lw=0.6)
        ax.set_ylim(0.3, 2.2); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        if r_ == 0:
            ax.set_title(("below $Q_{%g}$ / p" if PQ[i] < 0.5 else "above $Q_{%g}$ / (1-p)") % (100 * PQ[i]), fontsize=9)
        if col == 0:
            ax.set_ylabel("|%s| %.1f-%.1f" % (eta_var, ETA[k], ETA[k + 1]), fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("ln $p_T^{gen}$", fontsize=8)
axes[0, 0].legend(fontsize=7)
fig.suptitle("tail calibration: data fraction beyond the model's quantile / nominal (1 = correct; gray: $\\pm1\\sigma$)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(outdir, "tails.png"), dpi=110); plt.close(fig)

# accuracy maps over (|eta|, pt): rows = models; median and width bias (colour: pull), tail chi2/ndf
from matplotlib.colors import LogNorm, TwoSlopeNorm                 # noqa: E402
fig, axes = plt.subplots(len(labels), 3, figsize=(19, 3.4 * len(labels) + 0.6), squeeze=False)
MAPS = (("median_pull", "median_bias", "median: (model - data)/data"),
        ("width_pull", "width_bias", "$\\sigma_{68}$: (model - data)/data"),
        ("tail_chi2", None, "tails: $\\chi^2$/ndf of the 6 tail pulls"))
for r_, lab in enumerate(labels):
    for col, (key, vkey, title) in enumerate(MAPS):
        ax = axes[r_, col]
        Z = np.full((len(ETA) - 1, len(PT) - 1), np.nan)
        T = {}
        for c in cells:
            if lab not in c["models"]:
                continue
            i, j = c["eta_bin"], int(np.searchsorted(PT, c["pt"][0]))
            Z[i, j] = c["models"][lab][key]
            T[(i, j)] = ("%+.1f" % (100 * c["models"][lab][vkey])) if vkey else ("%.0f" % c["models"][lab][key])
        if vkey:
            mesh = ax.pcolormesh(np.log(PT), np.arange(len(ETA)), np.clip(Z, -10, 10), cmap="RdBu_r",
                                 norm=TwoSlopeNorm(0, -10, 10))
            fig.colorbar(mesh, ax=ax, pad=0.01).set_label("pull (clipped at $\\pm$10)", fontsize=7)
        else:
            mesh = ax.pcolormesh(np.log(PT), np.arange(len(ETA)), np.clip(Z, 0.3, 300), cmap="viridis",
                                 norm=LogNorm(0.3, 300))
            fig.colorbar(mesh, ax=ax, pad=0.01).set_label("$\\chi^2$/ndf", fontsize=7)
        for (i, j), t in T.items():
            ax.text(0.5 * (np.log(PT[j]) + np.log(PT[j + 1])), i + 0.5, t, ha="center", va="center", fontsize=5.5)
        ax.set_yticks(np.arange(len(ETA)))
        ax.set_yticklabels(["%g" % e for e in ETA], fontsize=7)
        ax.set_xlabel("ln($p_T^{gen}$ / GeV)", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_ylabel("|%s|" % eta_var, fontsize=8)
        ax.set_title("%s: %s%s" % (lab, title, "  (numbers in %)" if vkey else ""), fontsize=9)
fig.suptitle("accuracy against the data per cell (colour: significance; pull ~0, chi2/ndf ~1 = within statistics)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(outdir, "accuracy_map.png"), dpi=120); plt.close(fig)
print("outputs in", outdir)
