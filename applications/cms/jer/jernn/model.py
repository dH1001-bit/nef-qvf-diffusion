"""The network: conditions x -> Born density p(y | x), and loading a trained one.

    p(y | x) = N(y; mu(x), sigma(x)) |sum_{n<=J} c_n(x) phi_n(z)|^2 / ||c(x)||^2,   z = (y - mu)/sigma

The same family as the cell and global fits (response_models.BornModel), with (mu, sigma, c) now
given by a multilayer perceptron of the conditions instead of polynomials in ln pt per |eta| bin.
The density is normalised exactly for every x, so the unbinned likelihood needs no histogram, and
a new condition is one more input column.

Censoring (``y_range``): rows with y outside [lo, hi) enter with ln P(y < lo | x) or
ln P(y >= hi | x), as the tail terms of the global fit; the tails are not modelled pointwise.
These probabilities are integrated with Gauss-Legendre nodes in z over 25 units beyond the
boundary, in log space (no underflow far out).

Gauge: the overall scale and phase of c do not change p.  The scale is held near ||c|| = 1 by a
small penalty; the phase is left free.  (mu, sigma) and (c_1, c_2) can trade, as in the global fit:
single parameters are not physics, quantiles and moments are.

The network takes the RAW transformed conditions (e.g. ln pt, |eta|); the standardisation is a
buffer of the model, so a checkpoint is self-contained.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

LOG2PI = math.log(2 * math.pi)
TAIL_NODES, TAIL_LEN = 64, 25.0


def hermite(z, J):
    """phi_0 ... phi_J (last axis), orthonormal under N(0, 1)."""
    out = [torch.ones_like(z)]
    if J >= 1:
        out.append(z)
    for n in range(1, J):
        out.append((z * out[n] - math.sqrt(n) * out[n - 1]) / math.sqrt(n + 1))
    return torch.stack(out, -1)


class BornNet(nn.Module):
    def __init__(self, n_cond, J=8, width=128, depth=4, x_mean=None, x_std=None, y_loc=1.0, y_scale=0.1,
                 y_range=None):
        super().__init__()
        self.n_cond, self.J, self.width, self.depth = int(n_cond), int(J), int(width), int(depth)
        self.register_buffer("x_mean", torch.as_tensor(np.zeros(n_cond) if x_mean is None else x_mean,
                                                       dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(np.ones(n_cond) if x_std is None else x_std,
                                                      dtype=torch.float32))
        self.y_loc, self.y_scale = float(y_loc), float(y_scale)
        self.y_range = None if y_range is None else (float(y_range[0]), float(y_range[1]))
        layers, d = [], n_cond
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2 + 2 * (J + 1))
        with torch.no_grad():                    # start: a Gaussian at (y_loc, y_scale) for every x
            self.head.weight.normal_(0.0, 1e-3)
            self.head.bias.zero_()
            self.head.bias[2] = 1.0              # Re c_0
            self.head.bias[2 + J + 2:] = 0.01    # Im c_1..c_J: at real c the gradient in Im c vanishes
        x, w = np.polynomial.legendre.leggauss(TAIL_NODES)
        self.register_buffer("gl_x", torch.as_tensor(x, dtype=torch.float32))
        self.register_buffer("gl_lw", torch.as_tensor(np.log(w * TAIL_LEN / 2), dtype=torch.float32))

    def config(self):
        return dict(n_cond=self.n_cond, J=self.J, width=self.width, depth=self.depth, y_loc=self.y_loc,
                    y_scale=self.y_scale, y_range=self.y_range)

    # --- parameters of the density
    def forward(self, X):
        o = self.head(self.body((X - self.x_mean) / self.x_std))
        J = self.J
        mu = self.y_loc + self.y_scale * o[:, 0]
        sig = self.y_scale * torch.exp(torch.clamp(o[:, 1], -8.0, 5.0))
        return mu, sig, o[:, 2:3 + J], o[:, 3 + J:]

    # --- log density
    def _log_shape(self, z, cre, cim, lnorm):
        """ln N(z) + ln |h(z)|^2 - ln ||c||^2, z (..., ) with c broadcast over the leading axis."""
        Phi = hermite(z, self.J)
        if z.dim() == 2:                        # (rows, nodes)
            cre, cim = cre[:, None, :], cim[:, None, :]
            lnorm = lnorm[:, None]
        hr, hi = (Phi * cre).sum(-1), (Phi * cim).sum(-1)
        return -0.5 * z * z - 0.5 * LOG2PI + torch.log(hr * hr + hi * hi + 1e-30) - lnorm

    def _log_tail(self, zb, side, cre, cim, lnorm):
        """ln P(z < zb) (side -1) or ln P(z >= zb) (side +1)."""
        a = zb - TAIL_LEN if side < 0 else zb
        z = a[:, None] + (self.gl_x[None, :] + 1) * (TAIL_LEN / 2)
        return torch.logsumexp(self._log_shape(z, cre, cim, lnorm) + self.gl_lw[None, :], -1)

    def log_prob(self, y, X, params=None):
        """ln p(y | X) per row; censored rows (outside y_range) get ln P(outside)."""
        mu, sig, cre, cim = self(X) if params is None else params
        lnorm = torch.log((cre * cre + cim * cim).sum(-1))
        z = (y - mu) / sig
        lp = self._log_shape(z, cre, cim, lnorm) - torch.log(sig)
        if self.y_range is not None:
            lo, hi = self.y_range
            for side, m in ((-1, y < lo), (+1, y >= hi)):
                if bool(m.any()):
                    idx = torch.nonzero(m).squeeze(1)
                    zb = ((lo if side < 0 else hi) - mu[idx]) / sig[idx]
                    lp = lp.index_put((idx,), self._log_tail(zb, side, cre[idx], cim[idx], lnorm[idx]))
        return lp, lnorm


def loss_fn(model, y, X, w, norm_penalty=1e-3, ref_penalty=0.0):
    """Weighted mean negative log-likelihood (per unit |weight|) plus the gauge penalties."""
    params = model(X)
    lp, lnorm = model.log_prob(y, X, params)
    nll = -(w * lp).sum() / w.abs().sum()
    loss = nll + norm_penalty * (lnorm * lnorm).mean()
    if ref_penalty > 0:
        cre, cim = params[2], params[3]
        c12 = (cre[:, 1:3] ** 2 + cim[:, 1:3] ** 2).sum(-1) / torch.exp(lnorm)
        loss = loss + ref_penalty * c12.mean()
    return loss, nll


# ------------------------------------------------------------------ checkpoints
def save(path, model, info):
    torch.save(dict(kind="jernn-born", version=1, config=model.config(), state=model.state_dict(), info=info), path)


class Predictor:
    """A trained network for evaluation, numpy in and out.

        P = Predictor.load("fig/nn/best.pt")
        mu, sigma, c = P.params(X)              # X: (n, d) raw transformed conditions, in P.conds order
        lp = P.log_prob(y, X)
        X = P.make_X(genpt=300.0, receta=0.3)   # by column name (raw values; transforms applied)
    """

    def __init__(self, model, info, device="cpu"):
        self.model, self.info, self.device = model.eval(), info, device
        from .features import Feature
        self.target = Feature(info["target"])
        self.conds = [Feature(s) for s in info["conds"]]

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck.get("kind") != "jernn-born":
            raise ValueError("%s is not a jernn checkpoint" % path)
        cfg = dict(ck["config"])
        m = BornNet(**cfg)
        m.load_state_dict(ck["state"])
        return cls(m.to(device), ck["info"], device)

    def make_X(self, **raw):
        """Rows of conditions from raw column values (scalars or arrays, broadcast)."""
        cols = []
        for f in self.conds:
            if f.column not in raw:
                raise KeyError("condition %r needs %s=..." % (f.spec, f.column))
            cols.append(f(lambda n, v=raw[f.column]: np.asarray(v, float), {f.column}))
        return np.stack(np.broadcast_arrays(*cols), -1).reshape(-1, len(self.conds))

    @torch.no_grad()
    def params(self, X, chunk=1 << 18):
        out = [[], [], []]
        for i in range(0, len(X), chunk):
            mu, sig, cre, cim = self.model(torch.as_tensor(X[i:i + chunk], dtype=torch.float32, device=self.device))
            c = (cre + 1j * cim).cpu().numpy().astype(complex)
            out[0].append(mu.cpu().numpy()); out[1].append(sig.cpu().numpy())
            out[2].append(c / np.linalg.norm(c, axis=1, keepdims=True))
        return tuple(np.concatenate(o) for o in out)

    @torch.no_grad()
    def log_prob(self, y, X, chunk=1 << 18):
        out = []
        for i in range(0, len(X), chunk):
            yy = torch.as_tensor(y[i:i + chunk], dtype=torch.float32, device=self.device)
            xx = torch.as_tensor(X[i:i + chunk], dtype=torch.float32, device=self.device)
            out.append(self.model.log_prob(yy, xx)[0].cpu().numpy())
        return np.concatenate(out)

    def pdf_grid(self, X, ygrid):
        """p(y | x) for every row of X (rows) on a common y grid (columns), not censored."""
        from .diffusion import noised_pdf
        mu, sig, c = self.params(X)
        return noised_pdf(ygrid[None, :], mu[:, None], sig[:, None], c, 0.0)

    # CDF in z = (y - mu)/sigma on a common grid: (rows, nz), chunked
    ZGRID = np.linspace(-12.0, 12.0, 1601)

    def _cdf_z(self, c, chunk=256):
        from .diffusion import noised_pdf
        z = self.ZGRID
        F = np.empty((len(c), len(z)))
        for i in range(0, len(c), chunk):
            p = noised_pdf(z[None, :], 0.0, 1.0, c[i:i + chunk], 0.0)
            Fi = np.concatenate([np.zeros((len(p), 1)), np.cumsum(0.5 * (p[:, 1:] + p[:, :-1]) * np.diff(z), 1)], 1)
            F[i:i + chunk] = Fi / Fi[:, -1:]
        return F

    def _invert(self, F, u):
        """per row: z with F(z) = u (linear inside the grid cell)."""
        z = self.ZGRID
        j = np.clip((F < u[:, None]).sum(1) - 1, 0, len(z) - 2)
        r = np.arange(len(F))
        f0, f1 = F[r, j], F[r, j + 1]
        return z[j] + (z[j + 1] - z[j]) * np.clip((u - f0) / np.maximum(f1 - f0, 1e-300), 0, 1)

    def quantiles(self, X, q):
        """(rows, len(q)) quantiles of p(y | x)."""
        mu, sig, c = self.params(X)
        F = self._cdf_z(c)
        return np.stack([mu + sig * self._invert(F, np.full(len(mu), qq)) for qq in np.atleast_1d(q)], 1)

    def pit(self, y, X):
        """F(y | x) per row: uniform on [0, 1] if the network is calibrated."""
        mu, sig, c = self.params(X)
        F = self._cdf_z(c)
        z = self.ZGRID
        zi = np.clip((np.asarray(y) - mu) / sig, z[0], z[-1])
        j = np.clip(np.searchsorted(z, zi) - 1, 0, len(z) - 2)
        r = np.arange(len(zi))
        t = (zi - z[j]) / (z[j + 1] - z[j])
        return F[r, j] * (1 - t) + F[r, j + 1] * t

    def sample(self, X, rng=None):
        """One draw of y per row of X (inverse CDF on the grid)."""
        rng = np.random.default_rng(rng)
        mu, sig, c = self.params(X)
        return mu + sig * self._invert(self._cdf_z(c), rng.random(len(mu)))
