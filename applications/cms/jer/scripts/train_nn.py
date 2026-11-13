#!/usr/bin/env python3
"""Train the neural conditional density p(y | x) of jernn on unbinned rows.

    python3 scripts/train_nn.py --files "/scratch-cbe/users/robert.schoefbeck/Jets/jer_export_UL18/jer_UL18_HT*.npz" \
        --cond genpt:log receta:abs --outdir fig/nn_pt_eta
    python3 scripts/train_nn.py --files ... --cond genpt:log receta:abs rho nVtx --outdir fig/nn_full   # more variables
    python3 scripts/train_nn.py --files data.npz --weight none ...                                       # unweighted rows

Inputs are only read.  Every choice is an argument; a new condition is one more --cond spec (see
jernn/features.py: ``column[:transform]``, transforms id, log, abs, log1p; derived columns such as
``response`` = recpt / genpt).

Loss: weighted mean negative log-likelihood per unit weight of the Born density (jernn/model.py),
rows outside --y-range censored (ln P(outside)), plus a small penalty that holds ||c|| near 1.

Weights.  --weight names the column; with --weight none, or files without it (real data), every
row has weight 1.  With cross-section weights the lowest pt dominates.  --balance i --balance-power
alpha reweights along condition i so that every quantile bin of it carries a total weight
proportional to its n_eff^alpha (alpha = 1: as fit_response_global.py's default --pt-weighting
neff; for unit weights alpha = 1 changes nothing).  This only moves where the network spends its
capacity: the optimum is the true p(y | x) for any reweighting in x.

Batches.  --sampling weighted (default) draws rows with probability proportional to |w| (times
the balance factor) and gives them weight sign(w): the same expected gradient as uniform batches
with weights, with far less noise when the weights span orders of magnitude (HT slices).
--sampling uniform: plain shuffled epochs with weights.  The validation loss is always the exact
weighted mean over the validation rows.

Outputs in --outdir: best.pt (lowest validation loss), last.pt, history.json, loss.png,
summary.txt.  Load with jernn.model.Predictor.load(path).
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch                                                         # noqa: E402

from jernn import data as jd                                         # noqa: E402
from jernn.features import Cut, Feature                              # noqa: E402
from jernn.model import BornNet, loss_fn, save                       # noqa: E402

ap = argparse.ArgumentParser()
g = ap.add_argument_group("data")
g.add_argument("--files", nargs="+", required=True, help="npz files or globs (quote globs)")
g.add_argument("--target", default="response", help="spec of y")
g.add_argument("--cond", nargs="+", default=["genpt:log", "receta:abs"], help="specs of the conditions x")
g.add_argument("--cut", nargs="*", default=["dR<0.4", "genpt>=97"], help="row cuts, e.g. dR<0.4 'abs(receta)<5'")
g.add_argument("--weight", default="weight", help="weight column, or 'none' (all rows weight 1)")
g.add_argument("--fraction", type=float, default=1.0, help="keep this random fraction of the rows of every file")
g.add_argument("--max-rows", type=int, default=None, help="stop reading after this many rows (quick looks only)")
g.add_argument("--y-range", nargs="*", type=float, default=[0.0, 2.0],
               help="censoring range LO HI of y (rows outside enter as ln P(outside)); empty: no censoring")
g.add_argument("--val-frac", type=float, default=0.1)
g.add_argument("--balance", type=int, default=0, help="index of the condition to balance along; -1: off")
g.add_argument("--balance-power", type=float, default=1.0, help="alpha: bin totals ~ n_eff^alpha")
g.add_argument("--balance-bins", type=int, default=100)
g = ap.add_argument_group("model")
g.add_argument("--J", type=int, default=8, help="highest Hermite degree of the amplitude")
g.add_argument("--width", type=int, default=128)
g.add_argument("--depth", type=int, default=4)
g.add_argument("--norm-penalty", type=float, default=1e-3, help="holds ||c|| near 1 (gauge)")
g.add_argument("--ref-penalty", type=float, default=0.0, help="lambda * (|c_1|^2 + |c_2|^2)/||c||^2 (optional gauge)")
g = ap.add_argument_group("training")
g.add_argument("--sampling", default="weighted", choices=("weighted", "uniform"))
g.add_argument("--batch", type=int, default=8192)
g.add_argument("--epochs", type=int, default=40)
g.add_argument("--steps-per-epoch", type=int, default=None,
               help="batches per epoch (default: rows / batch); an epoch ends with a validation pass")
g.add_argument("--lr", type=float, default=2e-3)
g.add_argument("--weight-decay", type=float, default=0.0)
g.add_argument("--patience", type=int, default=6, help="stop after this many epochs without improvement")
g.add_argument("--seed", type=int, default=0)
g.add_argument("--device", default="auto", help="auto, cpu, cuda")
g.add_argument("--threads", type=int, default=None)
g.add_argument("--outdir", default="fig/nn")
a = ap.parse_args()

os.makedirs(a.outdir, exist_ok=True)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
if a.threads:
    torch.set_num_threads(a.threads)
dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                   ("cpu" if a.device == "auto" else a.device))
print("device:", dev, flush=True)

# ------------------------------------------------------------------ data (read only)
target, conds, cuts = Feature(a.target), [Feature(s) for s in a.cond], [Cut(s) for s in a.cut]
t0 = time.time()
D = jd.read(a.files, target, conds, cuts, weight=a.weight, fraction=a.fraction, max_rows=a.max_rows, seed=a.seed)
meta = D["meta"]
print("rows %d, n_eff %.0f, weights: %s, %.0f s" % (meta["n_rows"], meta["n_eff"], meta["weight"], time.time() - t0),
      flush=True)
if meta["n_rows"] < 1000:
    raise SystemExit("too few rows")
tr, va = jd.split(meta["n_rows"], a.val_frac, a.seed)

w_fit = D["w"].copy()
if a.balance >= 0 and conds:
    # balance factors from the training rows' bins, applied to all rows (validation uses the same loss)
    gfac = jd.balance(D["X"][:, a.balance], D["w"], tr, a.balance_bins, a.balance_power)
    w_fit = D["w"] * gfac
    print("balanced along %s (alpha %g): factors %.3g ... %.3g" % (conds[a.balance].label, a.balance_power,
                                                                  gfac[gfac > 0].min(), gfac.max()), flush=True)

# reference location and scale of y (initial Gaussian of the network)
y = D["y"]
q16, q50, q84 = np.quantile(y[tr], [0.16, 0.5, 0.84])
X = D["X"]
x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-6
model = BornNet(len(conds), J=a.J, width=a.width, depth=a.depth, x_mean=x_mean, x_std=x_std,
                y_loc=float(q50), y_scale=float(max(0.5 * (q84 - q16), 1e-3)),
                y_range=tuple(a.y_range) if len(a.y_range) == 2 else None).to(dev)
print("parameters: %d" % sum(p.numel() for p in model.parameters()), flush=True)

# tensors: on the device if they fit (a few GB), else on the CPU and moved per batch
def as_t(v, dtype=torch.float32):
    return torch.as_tensor(v, dtype=dtype)

on_dev = dev.type == "cpu" or (len(y) * (2 + len(conds)) * 4 < 0.5 * torch.cuda.get_device_properties(dev).total_memory)
put = (lambda t: t.to(dev)) if on_dev else (lambda t: t)
Yt, Xt, Wt = put(as_t(y)), put(as_t(X)), put(as_t(w_fit))
tr_t, va_t = put(torch.as_tensor(tr)), put(torch.as_tensor(va))
if a.sampling == "weighted":
    pw = np.abs(w_fit[tr]).astype(np.float64)
    cdf_t = put(torch.as_tensor(np.cumsum(pw) / pw.sum()))
    sign_t = put(as_t(np.sign(w_fit)))

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
steps = a.steps_per_epoch or max(1, len(tr) // a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps, pct_start=0.05)
gen = torch.Generator(device=cdf_t.device if a.sampling == "weighted" else "cpu").manual_seed(a.seed)


def batch_indices(step, perm):
    if a.sampling == "weighted":
        u = torch.rand(a.batch, generator=gen, device=cdf_t.device, dtype=torch.float64)
        return tr_t[torch.clamp(torch.searchsorted(cdf_t, u), max=len(tr) - 1)]
    return perm[step * a.batch:(step + 1) * a.batch]


@torch.no_grad()
def validate():
    model.eval()
    s_lp, s_w = 0.0, 0.0
    for i in range(0, len(va), 1 << 17):
        idx = va_t[i:i + (1 << 17)]
        yy, xx, ww = Yt[idx].to(dev), Xt[idx].to(dev), Wt[idx].to(dev)
        lp, _ = model.log_prob(yy, xx)
        s_lp += float((ww * lp).sum()); s_w += float(ww.abs().sum())
    model.train()
    return -s_lp / s_w


hist = dict(epoch=[], train_nll=[], val_nll=[], lr=[], seconds=[])
best, bad = np.inf, 0
info = dict(target=target.spec, conds=[f.spec for f in conds], cuts=[c.spec for c in cuts], data=meta,
            args=vars(a), y_quantiles=[float(q16), float(q50), float(q84)])
v0 = validate()
print("start: validation nll %.6f" % v0, flush=True)
t_train = time.time()
for ep in range(a.epochs):
    te = time.time()
    perm = tr_t[torch.randperm(len(tr), generator=torch.Generator().manual_seed(a.seed + ep)).to(tr_t.device)] \
        if a.sampling == "uniform" else None
    run, nrun = 0.0, 0
    for st in range(steps):
        idx = batch_indices(st, perm)
        if len(idx) == 0:
            break
        yy, xx = Yt[idx].to(dev, non_blocking=True), Xt[idx].to(dev, non_blocking=True)
        ww = (sign_t[idx] if a.sampling == "weighted" else Wt[idx]).to(dev, non_blocking=True)
        loss, nll = loss_fn(model, yy, xx, ww, a.norm_penalty, a.ref_penalty)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
        run += float(nll); nrun += 1
    v = validate()
    hist["epoch"].append(ep + 1); hist["train_nll"].append(run / max(nrun, 1)); hist["val_nll"].append(v)
    hist["lr"].append(sched.get_last_lr()[0]); hist["seconds"].append(time.time() - te)
    flag = ""
    if v < best - 1e-6:
        best, bad, flag = v, 0, "  *"
        save(os.path.join(a.outdir, "best.pt"), model, dict(info, epoch=ep + 1, val_nll=v))
    else:
        bad += 1
    print("epoch %3d  train %.6f  val %.6f  lr %.2e  %5.0f s%s" % (ep + 1, hist["train_nll"][-1], v,
                                                                  hist["lr"][-1], hist["seconds"][-1], flag), flush=True)
    with open(os.path.join(a.outdir, "history.json"), "w") as fh:
        json.dump(hist, fh, indent=1)
    if bad >= a.patience:
        print("no improvement for %d epochs: stop" % a.patience)
        break
save(os.path.join(a.outdir, "last.pt"), model, dict(info, epoch=hist["epoch"][-1], val_nll=hist["val_nll"][-1]))

neff_va = jd.n_eff(D["w"][va])
L = ["# train_nn.py %s" % " ".join(sys.argv[1:]),
     "target %s | conditions %s | cuts %s | weights %s" % (target.spec, ", ".join(f.spec for f in conds),
                                                          ", ".join(c.spec for c in cuts), meta["weight"]),
     "rows %d (train %d, validation %d), n_eff %.0f (validation %.0f)" % (meta["n_rows"], len(tr), len(va),
                                                                        meta["n_eff"], neff_va),
     "parameters %d, J %d, width %d, depth %d" % (sum(p.numel() for p in model.parameters()), a.J, a.width, a.depth),
     "validation nll per unit weight: start %.6f, best %.6f (epoch %d); n_eff(val) * gain = %.1f"
     % (v0, best, hist["epoch"][int(np.argmin(hist["val_nll"]))], neff_va * (v0 - best)),
     "training time %.0f s" % (time.time() - t_train)]
print("\n".join(L))
with open(os.path.join(a.outdir, "summary.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(hist["epoch"], hist["train_nll"], "o-", ms=3, label="train (batch mean)")
ax.plot(hist["epoch"], hist["val_nll"], "s-", ms=3, label="validation")
ax.set_xlabel("epoch"); ax.set_ylabel("nll per unit weight"); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "loss.png"), dpi=110)
print("outputs in", a.outdir)
