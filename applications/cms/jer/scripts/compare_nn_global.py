#!/usr/bin/env python3
"""The network against the global (pt-smooth) fit(s), row by row on the same held-out rows.

    python3 scripts/compare_nn_global.py --model fig/nn_pt_eta/best.pt --global fig/global_K3/global_fits.json
    python3 scripts/compare_nn_global.py --model ... --global K2=fig/global_receta/global_fits.json K3=fig/global_K3/global_fits.json

For every validation row i of the network (the rows are read again with the training's own
arguments, so the split is the same; --files overrides, then all rows are used):

    d_i = ln p_NN(r_i | x_i) - ln p_global(r_i | pt_i, |eta| bin of row i)

The network sees each row's own conditions (e.g. the exact |eta|), the global fit the |eta| bin;
that difference in information is part of what is compared.  Rows outside the censoring range of
the network (its y_range, default [0, 2)) enter both models as ln P(outside), so both are scored
on the same likelihood.  The global density is evaluated exactly per row (no histogram).

Summaries in cells of (|eta| bin of the global fit) x (--pt-edges), as in compare_global_cells.py:

    Delta n_eff lnL = n_eff / sum w * sum_i w_i d_i        (> 0: the network is better)
    error           = n_eff / sum w * sqrt(sum_i w_i^2 (d_i - dbar)^2)

with w the rows' own weights (cross sections, or 1 for unweighted rows), not the balancing of the
training.  The error is statistical (the rows are a sample); it does not include that the global
fit was fitted on these rows too (an in-sample advantage of O(its 58-76 parameters), negligible
against the row counts here).  Rows outside a global fit's ln pt range are extrapolations of the
polynomial: reported separately, not in the totals.

Outputs in --outdir (default <dir of --model>/compare_global):
  compare_nn_global.txt / .json   per cell and per |eta| bin: rows, n_eff, Delta n_eff lnL +- error,
                                  significance, and the same per global fit if several
  compare_nn_global.png           Delta n_eff lnL vs ln pt per |eta| bin (symlog), with errors
  pit_nn_global.png               PIT of the network and of the global fit per |eta| bin
Inputs are only read.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from jernn import data as jd                                          # noqa: E402
from jernn.diffusion import log_tail, noised_logpdf                    # noqa: E402
from jernn.features import Cut, Feature                               # noqa: E402
from jernn.model import Predictor                                     # noqa: E402
from response_models import BornModel                                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="checkpoint of train_nn.py")
ap.add_argument("--global", dest="glob", nargs="+", required=True, help="global_fits.json, or label=path (several)")
ap.add_argument("--files", nargs="+", default=None, help="other files (default: the training's, validation split)")
ap.add_argument("--pt-edges", nargs="+", type=float,
                default=[97, 140, 200, 280, 400, 550, 750, 1000, 1400, 2000, 3500, 6500])
ap.add_argument("--eta-var", default=None, help="|eta| column of the global fit (default: its 'var')")
ap.add_argument("--max-rows", type=int, default=None, help="random subset of the evaluation rows")
ap.add_argument("--pit-rows", type=int, default=300000, help="rows used for the PIT plot")
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

P = Predictor.load(a.model, a.device)
info, A = P.info, P.info["args"]
outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(a.model)), "compare_global")
os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(a.seed)

GL = []
for s in a.glob:
    lab, _, path = s.rpartition("=")
    with open(path) as fh:
        G = json.load(fh)
    fits = sorted(G["fits"], key=lambda f: f["eta_bin"])
    GL.append(dict(label=lab or os.path.basename(os.path.dirname(os.path.abspath(path))), path=os.path.abspath(path),
                   var=G.get("var", "receta"), fits=fits, models=[BornModel.from_dict(f["model"]) for f in fits]))
eta_var = a.eta_var or GL[0]["var"]
ETA = np.array([f["abs_eta"][0] for f in GL[0]["fits"]] + [GL[0]["fits"][-1]["abs_eta"][1]])
for g in GL[1:]:
    if not np.allclose([f["abs_eta"][0] for f in g["fits"]], ETA[:-1]):
        raise SystemExit("the global fits have different |eta| bins")

# ------------------------------------------------------------------ rows (read only)
# the network's conditions first (so the rows and the split are those of the training), then
# ln pt and |eta| for the global fit as two extra columns
extra = [Feature("genpt:log"), Feature("%s:abs" % eta_var)]
D = jd.read(a.files or info["data"]["files"], P.target, P.conds + extra, [Cut(s) for s in info["cuts"]],
            weight=A["weight"], fraction=A["fraction"], max_rows=A["max_rows"], seed=A["seed"])
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
print("%d rows (%s), n_eff %.0f" % (len(y), "validation split" if a.files is None else "all rows", jd.n_eff(w)), flush=True)

ie = np.searchsorted(ETA, aeta, side="right") - 1
ok_eta = (ie >= 0) & (ie < len(ETA) - 1)
yr = P.model.y_range
lo_r, hi_r = yr if yr is not None else (-np.inf, np.inf)


# ------------------------------------------------------------------ ln p of both models per row
def lnp_born(yv, mu, sig, c):
    """ln p with the network's censoring, for numpy Born parameters per row."""
    out = noised_logpdf(yv, mu, sig, c, 0.0)
    for side, m in ((-1, yv < lo_r), (+1, yv >= hi_r)):
        if m.any():
            zb = ((lo_r if side < 0 else hi_r) - mu[m]) / sig[m]
            out[m] = log_tail(zb, c[m], side)
    return out


ln_nn = P.log_prob(y.astype(np.float32), Xn.astype(np.float32)).astype(np.float64)
for g in GL:
    lg = np.full(len(y), np.nan)
    ext = np.zeros(len(y), bool)
    for k, (mod, th) in enumerate(g["models"]):
        m = np.flatnonzero(ok_eta & (ie == k))
        if len(m) == 0:
            continue
        ext[m] = (lpt[m] < mod.x0 - 1e-9) | (lpt[m] > mod.x1 + 1e-9)
        for s in np.array_split(m, max(1, len(m) // 200000)):
            c, mu, sig = mod.coefficients(th, np.exp(lpt[s]))
            lg[s] = lnp_born(y[s], mu, sig, c)
    g["lnp"], g["extrap"] = lg, ext
    print("%s: ln p evaluated (%d rows outside its ln pt range)" % (g["label"], ext.sum()), flush=True)


# ------------------------------------------------------------------ summaries
def summary(m, d):
    ww = w[m]
    sw = np.abs(ww).sum()
    if sw <= 0 or m.sum() < 2:
        return None
    ne = jd.n_eff(ww)
    dbar = (ww * d[m]).sum() / ww.sum()
    tot = ne / sw * (ww * d[m]).sum()
    err = ne / sw * np.sqrt((ww ** 2 * (d[m] - dbar) ** 2).sum())
    return dict(rows=int(m.sum()), n_eff=float(ne), dlnl=float(tot), err=float(err), signif=float(tot / max(err, 1e-300)))


PT = np.log(np.asarray(a.pt_edges))
ip = np.searchsorted(PT, lpt, side="right") - 1
res = {}
L = ["# network %s vs global fit(s), row by row on the same rows" % os.path.abspath(a.model),
     "# network: target %s, conditions %s; censoring range %s" % (info["target"], ", ".join(f.spec for f in P.conds), yr),
     "# Delta n_eff lnL = n_eff/sum w * sum w (ln p_NN - ln p_global)  (> 0: network better), +- statistical error",
     "# rows outside the global fit's ln pt range (extrapolation) are listed but not in the totals"]
for g in GL:
    d = ln_nn - g["lnp"]
    base = ok_eta & np.isfinite(d)
    inr = base & ~g["extrap"]
    cells, per_eta = [], []
    L.append("\n== %s (%s)" % (g["label"], g["path"]))
    L.append("%-11s %-12s %9s %10s %14s %8s" % ("|eta|", "pt_gen", "rows", "n_eff", "dn_eff lnL", "signif"))
    for k in range(len(ETA) - 1):
        for j in range(len(PT) - 1):
            s = summary(inr & (ie == k) & (ip == j), d)
            if s is None:
                continue
            s.update(eta_bin=k, abs_eta=[float(ETA[k]), float(ETA[k + 1])], pt=[a.pt_edges[j], a.pt_edges[j + 1]],
                     lnpt=float(np.average(lpt[inr & (ie == k) & (ip == j)])))
            cells.append(s)
            L.append("%4.2f-%-6.2f %5.0f-%-6.0f %9d %10.0f %8.1f+-%-5.1f %7.1f" % (
                ETA[k], ETA[k + 1], a.pt_edges[j], a.pt_edges[j + 1], s["rows"], s["n_eff"], s["dlnl"], s["err"],
                s["signif"]))
        s = summary(inr & (ie == k), d)
        if s is not None:
            s.update(eta_bin=k, abs_eta=[float(ETA[k]), float(ETA[k + 1])])
            per_eta.append(s)
    tot = summary(inr, d)
    ext = summary(base & g["extrap"], d)
    L.append("per |eta| bin:")
    for s in per_eta:
        L.append("  %4.2f-%-6.2f %9d rows  n_eff %10.0f  dn_eff lnL %10.1f +- %6.1f  (%.1f sigma)" % (
            *s["abs_eta"], s["rows"], s["n_eff"], s["dlnl"], s["err"], s["signif"]))
    if tot:
        L.append("TOTAL %s: dn_eff lnL %.1f +- %.1f (%.1f sigma), %d rows; network worse in %d of %d cells"
                 % (g["label"], tot["dlnl"], tot["err"], tot["signif"], tot["rows"],
                    sum(c["dlnl"] < 0 for c in cells), len(cells)))
    if ext:
        L.append("extrapolated rows (not in the total): %d, dn_eff lnL %.1f +- %.1f" % (ext["rows"], ext["dlnl"], ext["err"]))
    res[g["label"]] = dict(path=g["path"], cells=cells, per_eta=per_eta, total=tot, extrapolated=ext)
print("\n".join(L))
with open(os.path.join(outdir, "compare_nn_global.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(outdir, "compare_nn_global.json"), "w") as fh:
    json.dump(dict(model=os.path.abspath(a.model), pt_edges=a.pt_edges, eta_edges=ETA.tolist(), results=res), fh, indent=1)

# ------------------------------------------------------------------ plots
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

fig, axes = plt.subplots(1, len(GL), figsize=(7.5 * len(GL), 4.8), squeeze=False)
for ax, g in zip(axes[0], GL):
    cells = res[g["label"]]["cells"]
    for k in range(len(ETA) - 1):
        cc = [c for c in cells if c["eta_bin"] == k]
        if not cc:
            continue
        col = plt.cm.viridis(k / max(len(ETA) - 2, 1))
        ax.errorbar([c["lnpt"] for c in cc], [c["dlnl"] for c in cc], [c["err"] for c in cc], fmt="o-", ms=3.5,
                    lw=1, capsize=1.5, color=col, label="%.1f < |%s| < %.1f" % (ETA[k], eta_var, ETA[k + 1]))
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("ln($p_T^{gen}$ / GeV)"); ax.set_ylabel("$\\Delta\\, n_{eff} \\ln L$  (network $-$ global)")
    t = res[g["label"]]["total"]
    ax.set_title("network vs %s: total %.0f $\\pm$ %.0f" % (g["label"], t["dlnl"], t["err"]) if t else g["label"], fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "compare_nn_global.png"), dpi=115); plt.close(fig)

# PIT of both models per |eta| bin (rows inside the censoring range)
sel = np.flatnonzero(ok_eta & (y >= lo_r) & (y < hi_r) & ~GL[0]["extrap"])
if len(sel) > a.pit_rows:
    sel = np.sort(rng.choice(sel, a.pit_rows, replace=False))
u_nn = P.pit(y[sel], Xn[sel])
u_g = np.empty(len(sel))
z = P.ZGRID
for k, (mod, th) in enumerate(GL[0]["models"]):
    m = np.flatnonzero(ie[sel] == k)
    if len(m) == 0:
        continue
    c, mu, sig = mod.coefficients(th, np.exp(lpt[sel][m]))
    F = P._cdf_z(c)
    zi = np.clip((y[sel][m] - mu) / sig, z[0], z[-1])
    j = np.clip(np.searchsorted(z, zi) - 1, 0, len(z) - 2)
    t = (zi - z[j]) / (z[j + 1] - z[j])
    r = np.arange(len(m))
    u_g[m] = F[r, j] * (1 - t) + F[r, j + 1] * t
ne = len(ETA) - 1
fig, axes = plt.subplots(1, ne, figsize=(2.6 * ne, 2.8), sharey=True)
edges = np.linspace(0, 1, 21)
for k, ax in enumerate(np.atleast_1d(axes)):
    m = ie[sel] == k
    if m.sum() < 50:
        continue
    ww = np.abs(w[sel][m])
    for u, col, lab in ((u_nn, "C3", "network"), (u_g, "C0", GL[0]["label"])):
        h = np.histogram(u[m], edges, weights=ww)[0] / ww.sum() * 20
        ax.step(edges, np.r_[h, h[-1]], where="post", color=col, label=lab)
    nf = jd.n_eff(w[sel][m])
    ax.fill_between([0, 1], 1 - np.sqrt(20 / nf), 1 + np.sqrt(20 / nf), color="gray", alpha=0.2, lw=0)
    ax.set_title("|%s| %.1f-%.1f" % (eta_var, ETA[k], ETA[k + 1]), fontsize=8)
    ax.set_ylim(0.5, 1.5); ax.tick_params(labelsize=7); ax.set_xlabel("PIT", fontsize=8)
np.atleast_1d(axes)[0].legend(fontsize=7)
fig.suptitle("PIT per |%s| bin (flat = calibrated; gray: $\\pm1\\sigma$ statistical)" % eta_var, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "pit_nn_global.png"), dpi=110); plt.close(fig)
print("outputs in", outdir)
