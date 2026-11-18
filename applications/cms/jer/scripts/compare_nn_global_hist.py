#!/usr/bin/env python3
"""The network against the global fit(s) on the binned data histograms2d.npz: for a computer
without the event export (the row-by-row version is compare_nn_global.py).

    python3 scripts/compare_nn_global_hist.py --model fig/nn_pt_eta/best.pt \
        --global K3=fig/global_K3/global_fits.json --hist fig/global_receta/histograms2d.npz

Data: every fine bin (|eta| bin e, ln pt bin j, r bin k) of the histograms, with its summed weight
S_ejk (cross sections), the mean pt_j of the bin, and the weight below/above the r range.  These
are ALL rows of the export: the network saw a random --fraction of them in training (10 % for
fig/nn_pt_eta, of which 90 % for training), the global fits saw all of them.

Densities on the 1000 fine r centres (the r bins are 0.002 wide, much finer than any structure,
so the binned score is the unbinned one for all practical purposes):
  * global: p(r | pt_j) of the fit of |eta| bin e;
  * network: the histograms do not keep each jet's |eta|, so the network is mixed over --eta-nodes
    |eta| points inside the bin, p_NN(r | pt_j, e) = sum_i pi_i p_NN(r | pt_j, eta_i), in two ways:
      uniform   pi_i = 1/nodes.  The true |eta| distribution is not flat (in the forward bins the
                jets crowd at the low edge), so this is pessimistic for the network.
      fitted    pi fitted to the data per cell of --pt-edges (EM; nodes - 1 numbers per cell
                against millions of effective jets).  Optimistic: it also absorbs some mismodelling.
    Both discard what the network knows about |eta| inside the bin; the row-by-row comparison on
    the cluster (compare_nn_global.py) has neither problem and lies between the two.
Both models: rows below/above the r range enter as ln P(outside) (the network's censoring rule).

Per (|eta| bin) x (--pt-edges cell), as compare_nn_global.py:
    Delta n_eff lnL = n_eff / sum w * sum_jk S_jk d_jk,   d = ln p_NN - ln p_global,
    error = n_eff / sum w * sqrt(sum_jk S2_jk (d_jk - dbar)^2)
Extrapolated pt bins (outside a global fit's ln pt range) are listed separately.

Outputs in --outdir (default <dir of --model>/compare_global_hist): compare_nn_global_hist.txt/.json,
compare_nn_global_hist.png (Delta vs ln pt), curves_hist.png (median and sigma_68/median per |eta|
bin: data, network, global, in the cells), pit_hist.png (binned PIT per |eta| bin).
Inputs are only read.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from jernn.diffusion import log_tail, noised_logpdf                    # noqa: E402
from jernn.model import Predictor                                     # noqa: E402
from response_models import BornModel                                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--global", dest="glob", nargs="+", required=True, help="global_fits.json, or label=path")
ap.add_argument("--hist", required=True, help="histograms2d.npz")
ap.add_argument("--pt-edges", nargs="+", type=float,
                default=[97, 140, 200, 280, 400, 550, 750, 1000, 1400, 2000, 3500, 6500])
ap.add_argument("--eta-nodes", type=int, default=10, help="|eta| points of the network mixture per bin")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

P = Predictor.load(a.model)
names = [f.spec for f in P.conds]
outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(a.model)), "compare_global_hist")
os.makedirs(outdir, exist_ok=True)
with np.load(a.hist) as z:
    H = {k: z[k] for k in z.files}
var = str(H["var"])
ETA, LPT, R = H["eta_edges"], H["lpt_edges"], H["r_edges"]
rc = 0.5 * (R[1:] + R[:-1])
dr = R[1] - R[0]
lo_r, hi_r = R[0], R[-1]
yr = P.model.y_range
if yr is None or not np.allclose(yr, (lo_r, hi_r)):
    print("note: network censoring range %s, histogram range [%g, %g): scored on the histogram range" % (yr, lo_r, hi_r))
need = {"genpt:log", "%s:abs" % var}
if set(names) != need:
    raise SystemExit("this binned comparison needs the conditions exactly %s (network: %s)" % (sorted(need), names))
i_pt, i_eta = names.index("genpt:log"), names.index("%s:abs" % var)

GL = []
for s in a.glob:
    lab, _, path = s.rpartition("=")
    with open(path) as fh:
        G = json.load(fh)
    fits = sorted(G["fits"], key=lambda f: f["eta_bin"])
    if not np.allclose([f["abs_eta"][0] for f in fits] + [fits[-1]["abs_eta"][1]], ETA):
        raise SystemExit("%s: |eta| bins differ from the histograms" % path)
    GL.append(dict(label=lab or os.path.basename(os.path.dirname(os.path.abspath(path))), path=os.path.abspath(path),
                   models=[BornModel.from_dict(f["model"]) for f in fits]))


def lnp_row_grid(mu, sig, c):
    """(rows, nr) ln p on the fine r centres and (rows,) ln P below / above the range."""
    lp = noised_logpdf(rc[None, :], mu[:, None], sig[:, None], c, 0.0)
    return lp, log_tail((lo_r - mu) / sig, c, -1), log_tail((hi_r - mu) / sig, c, +1)


def logmeanexp(x, axis):
    m = x.max(axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.mean(np.exp(x - m), axis))


def fit_mix(lp, llo, lhi, S, Slo, Shi, iters=200):
    """EM for the |eta| node weights pi (one set per cell): maximises
    sum_jk S_jk ln sum_i pi_i p_i(r_k | pt_j) (+ tails).  lp (nj, nodes, nr), llo/lhi (nj, nodes)."""
    nn = lp.shape[1]
    lpi = np.full(nn, -np.log(nn))
    for _ in range(iters):
        a_ = lpi[None, :, None] + lp
        lm = logmeanexp(a_, 1) + np.log(nn)                     # ln sum_i pi_i p_i
        r = np.einsum("jk,jik->i", S, np.exp(a_ - lm[:, None, :]))
        for lt, St in ((llo, Slo), (lhi, Shi)):
            at = lpi[None, :] + lt
            r += (St[:, None] * np.exp(at - (logmeanexp(at, 1) + np.log(nn))[:, None])).sum(0)
        new = np.log(np.maximum(r / r.sum(), 1e-300))
        if np.max(np.abs(np.exp(new) - np.exp(lpi))) < 1e-7:
            lpi = new
            break
        lpi = new
    return lpi


ne_, M = len(ETA) - 1, len(LPT) - 1
PT = np.log(np.asarray(a.pt_edges))
MODES = ("uniform", "fitted")
res = {(g["label"], md): dict(cells=[]) for g in GL for md in MODES}
store = []                                                # per |eta| bin: arrays for plots
for e in range(ne_):
    W = H["W"][e]
    ok = W > 0
    js = np.flatnonzero(ok & (H["sw"][e].sum(1) + H["Wlo"][e] + H["Whi"][e] > 0))
    ptj = H["SP"][e][js] / W[js]
    S, S2 = H["sw"][e][js], H["sw2"][e][js]
    Slo, Shi = H["Wlo"][e][js], H["Whi"][e][js]
    S2lo, S2hi = H["W2lo"][e][js], H["W2hi"][e][js]
    cell = np.searchsorted(PT, np.log(ptj), side="right") - 1
    # network at |eta| nodes (midpoints of equal sub-intervals of the bin)
    nn = a.eta_nodes
    en = ETA[e] + (np.arange(nn) + 0.5) / nn * (ETA[e + 1] - ETA[e])
    X = np.zeros((len(js) * nn, 2))
    X[:, i_pt] = np.repeat(np.log(ptj), nn)
    X[:, i_eta] = np.tile(en, len(js))
    mu, sig, c = P.params(X.astype(np.float32))
    lpN, lloN, lhiN = lnp_row_grid(mu, sig, c)
    lpN, lloN, lhiN = lpN.reshape(len(js), nn, -1), lloN.reshape(len(js), nn), lhiN.reshape(len(js), nn)
    net = {"uniform": (logmeanexp(lpN, 1), logmeanexp(lloN, 1), logmeanexp(lhiN, 1))}
    # fitted |eta| weights, one set per cell of --pt-edges
    lpF, lloF, lhiF = np.empty_like(net["uniform"][0]), np.empty(len(js)), np.empty(len(js))
    pis = {}
    for k in np.unique(cell):
        m = cell == k
        lpi = fit_mix(lpN[m], lloN[m], lhiN[m], S[m], Slo[m], Shi[m])
        pis[int(k)] = np.exp(lpi).round(3).tolist()
        lpF[m] = logmeanexp(lpi[None, :, None] + lpN[m], 1) + np.log(nn)
        lloF[m] = logmeanexp(lpi[None, :] + lloN[m], 1) + np.log(nn)
        lhiF[m] = logmeanexp(lpi[None, :] + lhiN[m], 1) + np.log(nn)
    net["fitted"] = (lpF, lloF, lhiF)
    rec = dict(e=e, ptj=ptj, S=S, net={md: v[0] for md, v in net.items()}, pis=pis, glob={})
    for g in GL:
        mod, th = g["models"][e]
        cg, mug, sgg = mod.coefficients(th, ptj)
        lpg, llog, lhig = lnp_row_grid(mug, sgg, cg)
        ext = (np.log(ptj) < mod.x0 - 1e-9) | (np.log(ptj) > mod.x1 + 1e-9)
        rec["glob"][g["label"]] = dict(lp=lpg, ext=ext)
        for md in MODES:
            lpn, llon, lhin = net[md]
            d, dlo, dhi = lpn - lpg, llon - llog, lhin - lhig

            def summ(m):
                sw = S[m].sum() + Slo[m].sum() + Shi[m].sum()
                sw2 = S2[m].sum() + S2lo[m].sum() + S2hi[m].sum()
                if sw <= 0:
                    return None
                ne = sw * sw / sw2
                tot = (S[m] * d[m]).sum() + (Slo[m] * dlo[m]).sum() + (Shi[m] * dhi[m]).sum()
                db = tot / sw
                var_ = ((S2[m] * (d[m] - db) ** 2).sum() + (S2lo[m] * (dlo[m] - db) ** 2).sum()
                        + (S2hi[m] * (dhi[m] - db) ** 2).sum())
                return dict(n_eff=float(ne), dlnl=float(ne / sw * tot), err=float(ne / sw * np.sqrt(var_)),
                            lnpt=float(np.average(np.log(ptj[m]), weights=S[m].sum(1) + Slo[m] + Shi[m])))

            for k in range(len(PT) - 1):
                for is_ext in (False, True):
                    m = (cell == k) & (ext == is_ext)
                    s = summ(m) if m.any() else None
                    if s:
                        s.update(eta_bin=e, abs_eta=[float(ETA[e]), float(ETA[e + 1])],
                                 pt=[a.pt_edges[k], a.pt_edges[k + 1]], extrapolated=is_ext,
                                 eta_weights=pis.get(k) if md == "fitted" else None)
                        res[(g["label"], md)]["cells"].append(s)
    store.append(rec)
    print("|%s| bin %d done" % (var, e), flush=True)

L = ["# network %s vs global fit(s) on the binned data %s" % (os.path.abspath(a.model), os.path.abspath(a.hist)),
     "# the network is mixed over %d |%s| nodes inside each bin: 'uniform' (equal weights: pessimistic for the"
     % (a.eta_nodes, var),
     "# network) and 'fitted' (node weights fitted per cell to the data: optimistic).  The row-by-row result",
     "# (compare_nn_global.py on the cluster) lies in between.",
     "# Delta n_eff lnL = n_eff/sum w * sum S (ln p_NN - ln p_global) per cell (> 0: network better), +- statistical;",
     "# bin and total values are sums over cells (as in compare_global_cells.py); extrapolated cells not in the sums"]
summary = {}
for g in GL:
    L.append("\n== %s (%s)" % (g["label"], g["path"]))
    cu = {(c_["eta_bin"], c_["pt"][0], c_["extrapolated"]): c_ for c_ in res[(g["label"], "uniform")]["cells"]}
    cf = {(c_["eta_bin"], c_["pt"][0], c_["extrapolated"]): c_ for c_ in res[(g["label"], "fitted")]["cells"]}
    L.append("%-11s %-12s %10s %18s %18s" % ("|%s|" % var, "pt_gen", "n_eff", "uniform |eta|", "fitted |eta|"))
    for key, u_ in cu.items():
        f_ = cf[key]
        L.append("%4.2f-%-6.2f %5.0f-%-6.0f %10.0f %10.1f+-%-7.1f %10.1f+-%-7.1f%s" % (
            *u_["abs_eta"], *u_["pt"], u_["n_eff"], u_["dlnl"], u_["err"], f_["dlnl"], f_["err"],
            "  extrapolated" if key[2] else ""))
    L.append("per |%s| bin (sum over cells in the fitted pt range):         uniform |eta|          fitted |eta|" % var)
    summary[g["label"]] = {}
    for md in MODES:
        cells = [c_ for c_ in res[(g["label"], md)]["cells"] if not c_["extrapolated"]]
        per = []
        for e in range(ne_):
            ce = [c_ for c_ in cells if c_["eta_bin"] == e]
            per.append((sum(c_["dlnl"] for c_ in ce), np.sqrt(sum(c_["err"] ** 2 for c_ in ce)),
                        sum(c_["dlnl"] < 0 for c_ in ce), len(ce)))
        summary[g["label"]][md] = dict(per_eta=per, total=(sum(p[0] for p in per), float(np.sqrt(sum(p[1] ** 2 for p in per)))),
                                       worse=sum(p[2] for p in per), cells=sum(p[3] for p in per))
    for e in range(ne_):
        pu, pf = summary[g["label"]]["uniform"]["per_eta"][e], summary[g["label"]]["fitted"]["per_eta"][e]
        L.append("  %4.2f-%-6.2f %41.1f +- %6.1f %12.1f +- %6.1f   (network worse in %d / %d of %d cells)" % (
            ETA[e], ETA[e + 1], pu[0], pu[1], pf[0], pf[1], pu[2], pf[2], pu[3]))
    for md in MODES:
        sm = summary[g["label"]][md]
        L.append("TOTAL %s, %s |%s|: %.1f +- %.1f; network worse in %d of %d cells" % (
            g["label"], md, var, sm["total"][0], sm["total"][1], sm["worse"], sm["cells"]))
L.append("\n# fitted |%s| node weights per cell (nodes = midpoints of %d equal sub-intervals):" % (var, a.eta_nodes))
for rec in store:
    for k, pi in rec["pis"].items():
        L.append("  |%s| %.1f-%.1f  pt %5.0f-%-5.0f  %s" % (var, ETA[rec["e"]], ETA[rec["e"] + 1], a.pt_edges[k],
                                                        a.pt_edges[k + 1], " ".join("%.2f" % v for v in pi)))
print("\n".join(L))
with open(os.path.join(outdir, "compare_nn_global_hist.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(outdir, "compare_nn_global_hist.json"), "w") as fh:
    json.dump(dict(model=os.path.abspath(a.model), hist=os.path.abspath(a.hist), eta_nodes=a.eta_nodes,
                   results={"%s|%s" % k: v for k, v in res.items()}, summary=summary), fh, indent=1)

# ------------------------------------------------------------------ plots
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

fig, axes = plt.subplots(len(MODES), len(GL), figsize=(7.5 * len(GL), 4.6 * len(MODES)), squeeze=False)
for row, md in enumerate(MODES):
    for ax, g in zip(axes[row], GL):
        cells = [c_ for c_ in res[(g["label"], md)]["cells"] if not c_["extrapolated"]]
        for e in range(ne_):
            cc = [c_ for c_ in cells if c_["eta_bin"] == e]
            col = plt.cm.viridis(e / max(ne_ - 1, 1))
            ax.errorbar([c_["lnpt"] for c_ in cc], [c_["dlnl"] for c_ in cc], [c_["err"] for c_ in cc], fmt="o-",
                        ms=3.5, lw=1, capsize=1.5, color=col, label="%.1f < |%s| < %.1f" % (ETA[e], var, ETA[e + 1]))
        ax.axhline(0, color="gray", lw=0.8); ax.set_yscale("symlog", linthresh=10)
        t = summary[g["label"]][md]["total"]
        ax.set_title("network (%s |%s|) vs %s: total %.0f $\\pm$ %.0f" % (md, var, g["label"], *t), fontsize=10)
        ax.set_xlabel("ln($p_T^{gen}$ / GeV)"); ax.set_ylabel("$\\Delta\\, n_{eff}\\ln L$  (network $-$ global)")
        ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "compare_nn_global_hist.png"), dpi=110); plt.close(fig)


def qs(mass):
    cdf = np.r_[0.0, np.cumsum(mass)]
    q = np.interp(np.array([0.16, 0.5, 0.84]) * cdf[-1], cdf, R)
    return q[1], 0.5 * (q[2] - q[0]) / q[1]


g0 = GL[0]
fig, axes = plt.subplots(2, ne_, figsize=(3.0 * ne_, 5.6), sharex=False)
pfig, paxes = plt.subplots(1, ne_, figsize=(2.6 * ne_, 2.8), sharey=True)
edges = np.linspace(0, 1, 21)
for rec in store:
    e = rec["e"]
    cell = np.searchsorted(PT, np.log(rec["ptj"]), side="right") - 1
    ext = rec["glob"][g0["label"]]["ext"]
    norm = lambda p: p / p.sum(1, keepdims=True)                      # noqa: E731  conditional on the r range
    pn = {md: norm(np.exp(lp)) for md, lp in rec["net"].items()}
    out = {k: [] for k in ("x", "dat", "uniform", "fitted", "gl")}
    for k in np.unique(cell[~ext]):
        m = (cell == k) & ~ext
        Wj = rec["S"][m].sum(1)
        if Wj.sum() <= 0:
            continue
        out["x"].append(np.average(np.log(rec["ptj"][m]), weights=Wj))
        out["dat"].append(qs(rec["S"][m].sum(0)))
        for md in MODES:
            out[md].append(qs(Wj @ pn[md][m]))
        out["gl"].append({lab: qs(Wj @ norm(np.exp(gg["lp"][m]))) for lab, gg in rec["glob"].items()})
    x = np.array(out["x"])
    for row in (0, 1):
        ax = axes[row, e]
        ax.plot(x, [v[row] for v in out["dat"]], "ko", ms=3.5, label="data")
        ax.plot(x, [v[row] for v in out["uniform"]], "s-", color="C3", mfc="none", ms=4, lw=0.8, label="network, uniform |eta|")
        ax.plot(x, [v[row] for v in out["fitted"]], "s:", color="C1", ms=3, lw=0.8, label="network, fitted |eta|")
        for i, g in enumerate(GL):
            ax.plot(x, [v[g["label"]][row] for v in out["gl"]], "^--", color=("C0", "C2", "C4")[i % 3], ms=3.5, lw=0.8,
                    label=g["label"])
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        if row == 1:
            ax.set_xlabel("ln $p_T^{gen}$", fontsize=8)
    axes[0, e].set_title("|%s| %.1f-%.1f" % (var, ETA[e], ETA[e + 1]), fontsize=9)
    # binned PIT: every fine r bin at the model CDF of its centre
    for lp, col, lab in ((rec["net"]["uniform"], "C3", "network, uniform |eta|"),
                         (rec["net"]["fitted"], "C1", "network, fitted |eta|"),
                         (rec["glob"][g0["label"]]["lp"], "C0", g0["label"])):
        p = np.exp(lp)[~ext]
        F = np.cumsum(p, 1) * dr - 0.5 * p * dr
        F /= (p.sum(1, keepdims=True) * dr)
        Sx = rec["S"][~ext]
        h = np.histogram(F.ravel(), edges, weights=Sx.ravel())[0]
        h = h / h.sum() * 20
        paxes[e].step(edges, np.r_[h, h[-1]], where="post", color=col, label=lab)
    paxes[e].set_title("|%s| %.1f-%.1f" % (var, ETA[e], ETA[e + 1]), fontsize=8)
    paxes[e].set_ylim(0.8, 1.2); paxes[e].tick_params(labelsize=7); paxes[e].set_xlabel("PIT", fontsize=8)
axes[0, 0].set_ylabel("median"); axes[1, 0].set_ylabel("$\\sigma_{68}$ / median"); axes[0, 0].legend(fontsize=7)
fig.suptitle("data vs network (mixed over |%s| in the bin) vs global, cells of --pt-edges, r in [%g, %g)" % (var, lo_r, hi_r), fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "curves_hist.png"), dpi=110); plt.close(fig)
paxes[0].legend(fontsize=7)
pfig.suptitle("binned PIT per |%s| bin (fitted pt range; flat = calibrated)" % var, fontsize=9)
pfig.tight_layout(); pfig.savefig(os.path.join(outdir, "pit_hist.png"), dpi=110); plt.close(pfig)
print("outputs in", outdir)
