"""Figures (matplotlib, optional dependency)."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from .width import nsc_sigma
from .gaussian import log_window_mass



def _save(fig, path):
    """Save the figure; a .pdf path also gets a .png beside it (for web indexes)."""
    fig.savefig(path)
    if path.endswith(".pdf"):
        fig.savefig(path[:-4] + ".png", dpi=150)

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def response_bin(rows, full, core, title="", path=None, bins=100, rng=(0.0, 2.0)):
    """The starter figure: weighted response histogram on log y with the
    full-range and core Gaussians, both normalised to the weighted count."""
    plt = _mpl()
    r, w = rows.response, rows.weight
    h, e = np.histogram(r, bins=bins, range=rng, weights=w)
    h2, _ = np.histogram(r, bins=bins, range=rng, weights=w * w)
    c = 0.5 * (e[1:] + e[:-1]); dx = e[1] - e[0]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.errorbar(c, h, np.sqrt(h2), fmt="k.", ms=4, lw=0.8, label="simulation, weighted")
    x = np.linspace(rng[0], rng[1], 800)
    sw = w.sum()
    ax.plot(x, sw * dx * norm.pdf(x, full["mu"], full["sigma"]), "-", color="C3",
            label=r"Gaussian, full range ($\sigma$ = %.4f)" % full["sigma"])
    a, b = core.extra["window"]
    mass = np.exp(log_window_mass((a - core["mu"]) / core["sigma"], (b - core["mu"]) / core["sigma"]))
    inwin = ((r > a) & (r < b))
    ax.plot(x, w[inwin].sum() / mass * dx * norm.pdf(x, core["mu"], core["sigma"]), "--", color="C0",
            label=r"Gaussian, $\pm%g\sigma$ core ($\sigma$ = %.4f)" % (core.extra["k"], core["sigma"]))
    ax.axvspan(a, b, color="C0", alpha=0.07, lw=0)
    ax.set_yscale("log"); ax.set_ylim(max(h.max() * 1e-6, 1e-12), h.max() * 3)
    ax.set_xlabel(r"$p_T^{\rm rec}/p_T^{\rm gen}$"); ax.set_ylabel("weighted jets / bin")
    ax.set_title(title, fontsize=10); ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    if path: _save(fig, path)
    return fig


def sigma_vs_pt(points, curves, title="", path=None):
    """Per-bin widths with NSC curves; ``curves`` is {label: (N, S, C)}."""
    plt = _mpl()
    pt = np.array([p["pt"] for p in points]); s = np.array([p["fit"]["sigma"] for p in points])
    e = np.array([p["fit"].error("sigma") for p in points])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(pt, s, e, fmt="ko", ms=4, label="per-bin Gaussian")
    x = np.geomspace(pt.min() * 0.8, pt.max() * 1.2, 300)
    for i, (lab, th) in enumerate(curves.items()):
        ax.plot(x, nsc_sigma(x, *th), "-", color="C%d" % i, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$p_T^{\rm gen}$ [GeV]"); ax.set_ylabel(r"$\sigma(p_T^{\rm rec}/p_T^{\rm gen})$")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    if path: _save(fig, path)
    return fig


def pull_vs_pt(rows, theta, pt_edges, title="", path=None, k=None):
    """Weighted mean and RMS of the standardised residual z = (r - mu)/sigma(pt)
    per pt bin: a flat 0 and 1 is a Gaussian response with the fitted NSC."""
    plt = _mpl()
    mu, N, S, C = theta
    z = (rows.response - mu) / nsc_sigma(rows.genpt, N, S, C)
    w = rows.weight
    idx = np.digitize(rows.genpt, pt_edges) - 1
    cs, m1, m2 = [], [], []
    for i in range(len(pt_edges) - 1):
        m = idx == i
        if k is not None: m &= np.abs(z) < k
        if m.sum() < 20: continue
        ww = w[m]; zz = z[m]
        cs.append(np.average(rows.genpt[m], weights=ww))
        m1.append(np.average(zz, weights=ww)); m2.append(np.sqrt(np.average(zz * zz, weights=ww)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cs, m1, "o-", label="mean of z"); ax.plot(cs, m2, "s-", label="rms of z")
    ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="k", lw=0.5)
    ax.set_xscale("log"); ax.set_xlabel(r"$p_T^{\rm gen}$ [GeV]"); ax.set_ylabel(r"$z=(r-\mu)/\sigma(p_T)$")
    ax.set_title(title, fontsize=10); ax.legend(frameon=False)
    fig.tight_layout()
    if path: _save(fig, path)
    return fig


def efficiency(centres, eff, n, title="", path=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    err = np.sqrt(np.maximum(eff * (1 - eff) / np.maximum(n, 1), 0))
    ax.errorbar(centres, eff, err, fmt="ko", ms=4)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$p_T^{\rm gen}$ [GeV]"); ax.set_ylabel("matching efficiency")
    ax.set_title(title, fontsize=10); fig.tight_layout()
    if path: _save(fig, path)
    return fig
