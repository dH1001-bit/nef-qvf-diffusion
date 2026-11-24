#!/usr/bin/env python3
"""Quantiles and tails of the response: data against the network and the global fit(s), per
(|eta| slice) x (pt_gen bin), on the same rows.

    python3 scripts/quantiles_nn_global.py --model fig/nn_pt_eta_full2/best.pt --global K3=fig/global_K3/global_fits.json
    python3 scripts/quantiles_nn_global.py --model ... --global K3=... --eta-edges 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2
    python3 scripts/quantiles_nn_global.py --model J4=fig/nn_J4/best.pt J8=fig/nn_J8/best.pt J12=fig/nn_J12/best.pt \
        --global K3=fig/global_K3/global_fits.json --outdir fig/nn_J_scan        # several networks (J or lambda scan)

Several networks (--model label=path ...): they must have been trained on the same rows (same
files, cuts, fraction, seed, validation fraction, conditions); all are evaluated on the same
validation rows, the first is the reference of the likelihood comparison.

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
                        tail-calibration pulls; per |eta| slice the accuracy summary; the paired
                        likelihood of the networks; the wiggle counts
  likelihood.png        networks: n_eff * (lnL - lnL of the first network) per |eta| slice, row by row
                        on the same rows, with the paired statistical error
  wiggles.png           per model and |eta| slice: secondary maxima of p(r | x) per row (>1 % in ln p),
                        and the bumps of the cell mixtures that the data contradict / confirm

Wiggle check (as for the global fits): in every cell the model mixture is binned in r (0.01) and
compared with the cell's data histogram.  For every secondary maximum of the binned mixture (>5 %
above the dip towards the main peak) the rise dip -> bump (3 bins each) is compared, model and
data, in units of the data's statistical error: "visible" = model rise > 1 sigma, "confirmed" =
data rise > 2 sigma, "contradicted" = data - model < -2 sigma.
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
ap.add_argument("--model", nargs="+", required=True,
                help="checkpoint(s) of train_nn.py, path or label=path; the first is the likelihood reference")
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
ap.add_argument("--wiggle-rows", type=int, default=20, help="rows per cell for counting secondary maxima")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

NETS = {}
for s_ in a.model:
    lab, _, path = s_.rpartition("=")
    lab = lab or ("network" if len(a.model) == 1 else os.path.basename(os.path.dirname(os.path.abspath(path))))
    if lab in NETS:
        raise SystemExit("--model: label %s twice" % lab)
    NETS[lab] = Predictor.load(path, a.device)
NETLAB = list(NETS)
P = NETS[NETLAB[0]]
info, A = P.info, P.info["args"]
SAME = ("files", "fraction", "max_rows", "seed", "val_frac", "weight")
for lab, Pn in NETS.items():
    An = Pn.info["args"]
    bad = [k for k in SAME if a.files is None and An.get(k) != A.get(k)]
    if Pn.info["target"] != info["target"] or Pn.info["conds"] != info["conds"] or Pn.info["cuts"] != info["cuts"]:
        bad.append("target/conds/cuts")
    if bad:
        raise SystemExit("--model %s: trained on other rows than %s (%s): the comparison would not be on the same "
                         "rows" % (lab, NETLAB[0], ", ".join(bad)))
outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(a.model[0].rpartition("=")[2])), "quantiles")
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

# ------------------------------------------------------------------ paired likelihood of the networks
LP = {lab: Pn.log_prob(y, Xn).astype(np.float64) for lab, Pn in NETS.items()}
LIK = {}
for k in list(range(len(ETA) - 1)) + [-1]:
    m = np.flatnonzero((ie == k) if k >= 0 else (ie >= 0) & (ie < len(ETA) - 1))
    if len(m) == 0:
        continue
    ww = w[m]; W = ww.sum(); ne = jd.n_eff(ww)
    key = "%.2f-%.2f" % (ETA[k], ETA[k + 1]) if k >= 0 else "all"
    LIK[key] = dict(n_eff=float(ne), rows=int(len(m)), nets={})
    for lab in NETLAB:
        d = LP[lab][m] - LP[NETLAB[0]][m]
        md = float(ww @ d / W)
        err = float(np.sqrt(np.sum(ww ** 2 * (d - md) ** 2)) / abs(W))
        LIK[key]["nets"][lab] = dict(lnl_per_event=float(ww @ LP[lab][m] / W), dlnl_per_event=md, err=err,
                                     dlnl_neff=ne * md, err_neff=ne * err)
del LP

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


RW = np.linspace(0.0, 2.0, 8001)                                       # grid for secondary maxima
EW = np.linspace(0.0, 2.0, 201)                                        # 0.01 bins for the data check


def count_maxima(p):
    """secondary maxima of each row of p (on RW) that rise > 1 % in ln p above the lowest point
    between them and the main peak."""
    lp = np.log(np.maximum(p, 1e-300))
    n = 0
    for row in lp:
        i0 = int(np.argmax(row))
        pk = np.flatnonzero((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:])) + 1
        for i in pk:
            if i != i0 and row[i] - row[min(i, i0):max(i, i0) + 1].min() > 0.01:
                n += 1
    return n


def wiggles(M, yc, wc):
    """secondary maxima per row (first --wiggle-rows model rows of the cell) and the data check of
    the bumps of the cell's binned mixture: list of (model rise, data rise) in data sigma."""
    mu, sig, A_, s_ = M
    nr = min(a.wiggle_rows, len(mu))
    p = pdf_from_a((RW[None, :] - mu[:nr, None]) / sig[:nr, None], A_[:nr]) / sig[:nr, None]
    nmax = count_maxima(p)
    W = wc.sum()
    pred = np.diff(mix_cdf(EW, M)) * W
    sw = np.histogram(yc, EW, weights=wc)[0]
    sw2 = np.histogram(yc, EW, weights=wc * wc)[0]
    wbar = (wc * wc).sum() / W
    pk = int(np.argmax(pred))
    cases = []
    for i in range(2, len(pred) - 2):
        if i == pk or not (pred[i] > pred[i - 1] and pred[i] >= pred[i + 1]):
            continue
        lo, hi = sorted((i, pk))
        d = lo + int(np.argmin(pred[lo:hi + 1]))
        if pred[i] / max(pred[d], 1e-300) < 1.05:
            continue
        win = lambda c_: slice(max(c_ - 1, 0), c_ + 2)                # noqa: E731
        pb, pd = pred[win(i)].sum(), pred[win(d)].sum()
        db, dd = sw[win(i)].sum(), sw[win(d)].sum()
        e = np.hypot(np.sqrt(max(sw2[win(i)].sum(), pb * wbar)), np.sqrt(max(sw2[win(d)].sum(), pd * wbar)))
        cases.append([float((pb - pd) / e), float((db - dd) / e)])
    return dict(maxima_per_row=nmax / max(nr, 1), bumps=cases)


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
        for lab, Pn in NETS.items():
            mu, sig, c = Pn.params(Xn[sub])
            models[lab] = model_rows(mu, sig, c, s)
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
        fd = np.maximum(mix_pdf(Qd, models[NETLAB[0]]), 1e-9)
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
                                      tail_chi2=float(np.mean(tp[list(TAILS)] ** 2)),
                                      wiggles=wiggles(M, y[m], ww))
        cells.append(rec)
    print("|%s| %.2f-%.2f done (%d cells)" % (eta_var, ETA[k], ETA[k + 1], sum(c["eta_bin"] == k for c in cells)),
          flush=True)

labels = NETLAB + [g["label"] for g in GL]

# ------------------------------------------------------------------ text summary
L = ["# quantiles of r: data vs network(s) (%s) and global fit(s), validation rows%s" %
     (", ".join(a.model), "" if a.files is None else " of --files"),
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
L += ["", "# likelihood of the networks, row by row on the same rows: n_eff * (lnL - lnL[%s]) per |eta| slice" % NETLAB[0],
      "# (+- paired statistical error; positive: better than %s), and lnL per event" % NETLAB[0],
      "%-11s %12s" % ("|eta|", "n_eff") + "".join(" %22s" % l[:22] for l in NETLAB)]
for key, v in LIK.items():
    L.append("%-11s %12.4g" % (key, v["n_eff"]) + "".join(
        " %11.1f +- %7.1f" % (v["nets"][l]["dlnl_neff"], v["nets"][l]["err_neff"]) for l in NETLAB))
L.append("%-11s %12s" % ("lnL/event", "") + "".join(" %22.6f" % LIK["all"]["nets"][l]["lnl_per_event"] for l in NETLAB))
L += ["", "# wiggles: secondary maxima of p(r|x) per row (cell mean); bumps of the cell mixtures: all | visible (model",
      "# rise > 1 sigma) | confirmed by the data (data rise > 2 sigma) | contradicted (data - model < -2 sigma)",
      "%-11s %-9s %9s %7s %8s %10s %13s" % ("|eta|", "model", "max/row", "bumps", "visible", "confirmed", "contradicted")]
WIG = {}
for k in list(range(len(ETA) - 1)) + [-1]:
    cc = [c for c in cells if c["eta_bin"] == k or k < 0]
    if not cc:
        continue
    key = "%.2f-%.2f" % (ETA[k], ETA[k + 1]) if k >= 0 else "all"
    for lab in labels:
        cl = [c for c in cc if lab in c["models"]]
        if not cl:
            continue
        b = np.array([x for c in cl for x in c["models"][lab]["wiggles"]["bumps"]]).reshape(-1, 2)
        vis = b[:, 0] > 1
        r_ = dict(maxima_per_row=float(np.mean([c["models"][lab]["wiggles"]["maxima_per_row"] for c in cl])),
                  bumps=int(len(b)), visible=int(vis.sum()), confirmed=int(np.sum(b[vis, 1] > 2)),
                  contradicted=int(np.sum(b[vis, 1] - b[vis, 0] < -2)))
        WIG.setdefault(key, {})[lab] = r_
        L.append("%-11s %-9s %9.2f %7d %8d %10d %13d" % (key, lab[:9], r_["maxima_per_row"], r_["bumps"], r_["visible"],
                                                        r_["confirmed"], r_["contradicted"]))
print("\n".join(L))
with open(os.path.join(outdir, "quantiles.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(outdir, "quantiles.json"), "w") as fh:
    json.dump(dict(models={l: os.path.abspath(s_.rpartition("=")[2]) for l, s_ in zip(NETLAB, a.model)}, PQ=PQ.tolist(),
                   eta_edges=ETA.tolist(), pt_edges=PT.tolist(), eta_var=eta_var, cells=cells, summary=summ,
                   likelihood=LIK, wiggles=WIG), fh, indent=1)

# ------------------------------------------------------------------ plots
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

if len(NETLAB) == 1:
    COL = {NETLAB[0]: "C3"}
else:                                                  # a scan: ordered colours
    COL = {l: plt.cm.plasma(x) for l, x in zip(NETLAB, np.linspace(0.05, 0.85, len(NETLAB)))}
for i, g in enumerate(GL):
    COL[g["label"]] = "C%d" % (0 if i == 0 else i + 1) if len(NETLAB) == 1 else "C%d" % (2 * i)
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
        e = np.array([c["models"][NETLAB[0]]["tail_err"][i] for c in cc])
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

# likelihood of the networks vs model (x: the order of --model), one line per |eta| slice
keys = [k_ for k_ in LIK if k_ != "all"]
if len(NETLAB) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(len(NETLAB))
    for n, key in enumerate(keys + ["all"]):
        v = LIK[key]["nets"]
        dv = np.array([v[l]["dlnl_neff"] for l in NETLAB]); de = np.array([v[l]["err_neff"] for l in NETLAB])
        ax = axes[1] if key == "all" else axes[0]
        ax.errorbar(xs + (0.03 * (n - len(keys) / 2) if key != "all" else 0), dv, de, fmt="o-", ms=3, capsize=2,
                    color="k" if key == "all" else plt.cm.viridis(n / max(len(keys) - 1, 1)), label="|eta| " + key)
    for ax, t in zip(axes, ("per |%s| slice" % eta_var, "all rows")):
        ax.axhline(0, color="gray", lw=0.7); ax.set_xticks(xs); ax.set_xticklabels(NETLAB)
        ax.set_ylabel("n_eff (lnL - lnL[%s])" % NETLAB[0]); ax.set_title(t); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("likelihood of the networks on the same validation rows (paired errors)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "likelihood.png"), dpi=115); plt.close(fig)

# wiggles per model and |eta| slice
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
xs = np.arange(len(labels))
for n, key in enumerate(keys + ["all"]):
    if key not in WIG:
        continue
    col = "k" if key == "all" else plt.cm.viridis(n / max(len(keys) - 1, 1))
    for ax, q in zip(axes, ("maxima_per_row", "visible", "contradicted")):
        ax.plot(xs, [WIG[key].get(l, {}).get(q, np.nan) for l in labels], "o-", ms=3, color=col,
                lw=2 if key == "all" else 1, label="|eta| " + key)
for ax, t in zip(axes, ("secondary maxima of p(r|x) per row", "visible bumps (model rise > 1 sigma)",
                        "bumps contradicted by the data (< -2 sigma)")):
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8); ax.set_title(t, fontsize=10); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("tail wiggles per model (bumps confirmed by the data: %s)"
             % ", ".join("%s %d" % (l, WIG["all"][l]["confirmed"]) for l in labels if l in WIG.get("all", {})), fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "wiggles.png"), dpi=115); plt.close(fig)
print("outputs in", outdir)
