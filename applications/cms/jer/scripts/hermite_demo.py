#!/usr/bin/env python3
"""Hermite polynomials: the orthogonal polynomial system (OPS) of the normal
law, and how a non-Gaussian response is built on it.  A walk-through, not a
fit: every step prints what it checks, and the fit is left to you (section 6).

Go through it line by line:

    . /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh && conda activate hephy-ml-gpu-claude
    cd ~/ML/nef-qvf-diffusion/applications/cms/jer
    ipython -i scripts/hermite_demo.py            # or, inside ipython:  %run scripts/hermite_demo.py

Everything is in the module scope, so after the run each variable (phi, c,
R, rows, moments, ...) is yours to inspect.  Optional argument: the rows file
for section 5 (default: the prescale-1000 campaign sample, see README).

Sections
  1. the basis   phi_n(x) = He_n((x - mu)/sigma) / sqrt(n!),  orthonormal under N(mu, sigma^2)
  2. the three-term recurrence (Jacobi coefficients) and Clenshaw evaluation
  3. the Born model  p = p_ref h^2,  h = sum_n c_n phi_n,  ||c|| = 1:  normalisation and moments
  4. a density with a low-side tail, a sample from it, and the log-likelihood -- evaluated, not fitted
  5. the JER rows: standardised response and its weighted moments <phi_n>, with errors
  6. your task
"""
import os
import sys

import numpy as np
from scipy.special import eval_hermitenorm, factorial

# --------------------------------------------------------------------------
# 0. imports: the nefqvf package (repository src/) and the jerfit package
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
APP = os.path.abspath(os.path.join(HERE, ".."))                  # applications/cms/jer
REPO = os.path.abspath(os.path.join(APP, "..", "..", ".."))        # the repository root
for p in (APP, os.path.join(REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from nefqvf import Normal, NormalParams, product_matrices, ratio_coefficients   # noqa: E402
from jerfit import load, select, fit_bin, fit_core                              # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def banner(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# --------------------------------------------------------------------------
# 1. the basis
# --------------------------------------------------------------------------
banner("1. the orthonormal Hermite basis of N(mu, sigma^2)")

mu, sigma = 1.0, 0.07                     # a JER-like reference: response 1, width 7%
ref = NormalParams(mu, sigma)             # the reference law; nefqvf calls its parameters 'params'
K = 6                                     # highest degree we look at
x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1001)

# Normal.basis evaluates phi_0 ... phi_K at every x; the LAST axis is the degree.
phi = Normal.basis(x, K, ref)             # shape (1001, K+1)
print("phi.shape =", phi.shape)

# The convention, checked against scipy: probabilists' Hermite He_n of the
# standardised variable, divided by sqrt(n!) so that the basis is orthoNORMAL.
z = (x - mu) / sigma
He = np.stack([eval_hermitenorm(n, z) / np.sqrt(factorial(n)) for n in range(K + 1)], axis=-1)
print("max |phi_n - He_n(z)/sqrt(n!)| = %.1e" % np.abs(phi - He).max())
print("phi_0 = 1, phi_1 = z, phi_2 = (z^2 - 1)/sqrt 2, phi_3 = (z^3 - 3z)/sqrt 6 :",
      np.allclose(phi[:, 1], z), np.allclose(phi[:, 2], (z**2 - 1) / np.sqrt(2)),
      np.allclose(phi[:, 3], (z**3 - 3 * z) / np.sqrt(6)))

# Orthonormality:  E_ref[phi_m phi_n] = delta_mn.  The expectation under N(mu, sigma^2)
# is a Gauss-Hermite quadrature (probabilists' nodes, weights normalised to 1),
# exact for polynomials of degree < 2 * (number of nodes).
zq, wq = np.polynomial.hermite_e.hermegauss(40)
wq = wq / wq.sum()
xq = mu + sigma * zq
B = Normal.basis(xq, K, ref)                         # (40, K+1)
gram = (B * wq[:, None]).T @ B                       # [m, n] = sum_q w_q phi_m(x_q) phi_n(x_q)
print("Gram matrix - identity: max |.| = %.1e" % np.abs(gram - np.eye(K + 1)).max())

# --------------------------------------------------------------------------
# 2. the recurrence, and how to evaluate an expansion
# --------------------------------------------------------------------------
banner("2. the three-term recurrence and Clenshaw evaluation")

# Every OPS satisfies  x phi_n(x) = a_{n+1} phi_{n+1}(x) + b_n phi_n(x) + a_n phi_{n-1}(x).
# For the normal law  a_n = sigma sqrt(n),  b_n = mu.  nefqvf returns them in the
# 'positive off-diagonal' convention (a_n > 0); the same two sequences define
# every family's basis, which is why one code path serves all six families.
n = np.arange(K + 2)
a, b = Normal.jacobi_coefficients(n, ref)
print("a_n =", a, " (= sigma sqrt(n))")
print("b_n =", b, " (= mu)")
m = 3
lhs = x * phi[:, m]
rhs = a[m + 1] * Normal.basis(x, m + 1, ref)[:, m + 1] + b[m] * phi[:, m] + a[m] * phi[:, m - 1]
print("recurrence at n = %d: max |lhs - rhs| = %.1e" % (m, np.abs(lhs - rhs).max()))

# An expansion h(x) = sum_n c_n phi_n(x).  Two ways to evaluate it:
c_demo = np.array([1.0, 0.0, 0.1, -0.2, 0.1, 0.0, 0.0])
h_materialised = phi @ c_demo                        # build the basis, contract
h_clenshaw = Normal.basis_dot(x, c_demo, ref)        # Clenshaw recurrence, never builds the basis
print("basis @ c  vs  basis_dot: max |difference| = %.1e" % np.abs(h_materialised - h_clenshaw).max())
print("use basis_dot in a likelihood: it is the stable path and does not allocate (n_rows x K) memory")

# --------------------------------------------------------------------------
# 3. the Born model: p = p_ref h^2
# --------------------------------------------------------------------------
banner("3. the Born model  p(x) = p_ref(x) h(x)^2,  h = sum c_n phi_n")

# Why the square: p_ref h^2 is non-negative for ANY real c, and its integral is
#     int p_ref h^2 = sum_mn c_m c_n E_ref[phi_m phi_n] = sum_n c_n^2 = ||c||^2,
# so normalisation is the unit sphere ||c|| = 1 and nothing else.  c = e_0
# gives h = 1 and p = p_ref: the Gaussian is the point (1, 0, 0, ...) of the sphere.
c = c_demo / np.linalg.norm(c_demo)
p_ref = Normal.prob(x, ref)
h = Normal.basis_dot(x, c, ref)
p = p_ref * h**2
# the integral by quadrature (h^2 has degree 2K < 80, exact):
print("int p dx = sum_q w_q h(x_q)^2 = %.12f   (||c||^2 = %.12f)"
      % (np.sum(wq * Normal.basis_dot(xq, c, ref) ** 2), c @ c))

# Moments.  The expectations of the basis functions under p are quadratic in c:
#     E_p[phi_k] = sum_mn c_m c_n Lambda_mnk,     Lambda_mnk = E_ref[phi_m phi_n phi_k],
# and nefqvf builds Lambda exactly from (a_n, b_n) -- no quadrature anywhere.
# product_matrices returns Phi[k] = Lambda[:K+1, :K+1, k] for k = 0 ... 2K,
# ratio_coefficients returns R_k(c) = c^T Phi[k] c.  These R_k are what a fit
# must reproduce from the data (section 5 measures them).
Phi = product_matrices(Normal, ref, K)               # (2K+1, K+1, K+1)
R = ratio_coefficients(c, Phi)                       # (2K+1,)
B2 = Normal.basis(xq, 2 * K, ref)
R_quad = (wq * Normal.basis_dot(xq, c, ref) ** 2) @ B2
print("R_k = E_p[phi_k], k = 0 ... %d:" % (2 * K))
print("  from Lambda    :", R)
print("  by quadrature  :", R_quad)
print("  R_0 = 1 is the normalisation; R_1 = (E_p[x] - mu)/sigma = %.4f is the mean shift in units of sigma" % R[1])
print("  R_2 = (Var_p[x]/sigma^2 + R_1^2 - 1)/sqrt 2 = %.4f  ->  Var_p[x]/sigma^2 = %.4f"
      % (R[2], np.sqrt(2) * R[2] - R[1] ** 2 + 1))
print("  R_k for k > K is not zero: h^2 has degree 2K.  Degrees 1 and 2 are mu and sigma in disguise;")
print("  a non-Gaussian SHAPE starts at degree 3 (skewness) and 4 (tails).")

# --------------------------------------------------------------------------
# 4. a density with a low-side tail; a sample; the log-likelihood (evaluated only)
# --------------------------------------------------------------------------
banner("4. a low-side tail from degrees 3 and 4; a sample; the weighted log-likelihood")

# A negative degree-3 coefficient skews towards low x (the cross term 2 c_0 c_3 phi_3
# is positive where phi_3 < 0, i.e. at negative z), a positive degree-4 one adds
# symmetric tails.  Normalise and look at the result on a log scale.
c_true = np.array([1.0, 0.0, 0.0, -0.12, 0.08, 0.0, 0.0])
c_true /= np.linalg.norm(c_true)
h_true = Normal.basis_dot(x, c_true, ref)
p_true = p_ref * h_true**2
R_true = ratio_coefficients(c_true, Phi)
print("c_true =", c_true)
print("R_k(c_true) =", R_true[: K + 1])
print("p/p_ref at z = -3: %.1f,  at z = +3: %.1f,  at z = -4: %.0f" % tuple(
    (h_true**2)[np.searchsorted(z, v)] for v in (-3.0, 3.0, -4.0)))

# A sample.  p is a density we can tabulate, so sample it by the inverse CDF on
# a fine grid: F(x) = int_-inf^x p, then x = F^-1(u) for uniform u.  (Rejection
# from the reference with h^2/max h^2 would be exact too, but h^2 is a polynomial
# and its maximum over the reference's range makes the acceptance tiny.)
rng = np.random.default_rng(1)
xg = mu + sigma * np.linspace(-12, 12, 240_001)
pg = Normal.prob(xg, ref) * Normal.basis_dot(xg, c_true, ref) ** 2
cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pg[1:] + pg[:-1]) * np.diff(xg))])
cdf /= cdf[-1]
n_toy = 200_000
r_toy = np.interp(rng.uniform(size=n_toy), cdf, xg)
w_toy = rng.gamma(2.0, 0.5, n_toy)                  # arbitrary positive weights, mean 1, as the rows have
print("sample: %d events by inverse CDF; weighted mean %.4f (R_1 says %.4f), n_eff %.0f"
      % (n_toy, (w_toy @ r_toy) / w_toy.sum(), mu + sigma * R_true[1], w_toy.sum() ** 2 / (w_toy**2).sum()))

# The data side of the moment identity: the weighted sample means of phi_k
# estimate R_k.  Errors from the weights (ratio estimator):
#     var(<phi>) = sum w_i^2 (phi_i - <phi>)^2 / (sum w_i)^2.
def weighted_moments(r, w, params, k_max):
    """<phi_k>_w and its error for k = 0 ... k_max."""
    B = Normal.basis(r, k_max, params)
    mean = (w @ B) / w.sum()
    err = np.sqrt(((w[:, None] ** 2) * (B - mean) ** 2).sum(axis=0)) / w.sum()
    return mean, err

m_toy, e_toy = weighted_moments(r_toy, w_toy, ref, 2 * K)
print("k    R_k(c_true)   <phi_k>_w +- err      pull")
for k in range(1, 9):
    print("%-4d %10.4f   %10.4f +- %.4f  %6.1f" % (k, R_true[k], m_toy[k], e_toy[k], (m_toy[k] - R_true[k]) / e_toy[k]))

p_approx = []

for i in range(K):
    r_approx, e_approx = weighted_moments(r_toy, w_toy, ref, 2*i)
    p_approx.append(p_ref*Normal.basis_dot(x, r_approx, ref))

# The weighted log-likelihood of the Born model.  Per row:
#     log p(r_i) = log N(r_i; mu, sigma) + 2 log |h(r_i)|,      h = basis_dot(r_i, c/||c||)
# This is the function a fit maximises over (c_active, mu, sigma).  Here it is
# only EVALUATED: at the truth, and at the Gaussian point c = e_0.
def loglik(c, mu, sigma, r, w):
    """sum_i w_i log p(r_i) for the Born model with amplitude c (normalised inside)."""
    c = np.asarray(c, float)
    c = c / np.linalg.norm(c)
    params = NormalParams(mu, sigma)
    h = Normal.basis_dot(r, c, params)
    with np.errstate(divide="ignore"):                 # h = 0 at a root of the amplitude: log p = -inf there
        return np.sum(w * (Normal.log_prob(r, params) + 2.0 * np.log(np.abs(h))))

e0 = np.eye(K + 1)[0]
ll_true = loglik(c_true, mu, sigma, r_toy, w_toy)
ll_gauss = loglik(e0, mu, sigma, r_toy, w_toy)
print("log L(c_true) - log L(Gaussian) = %.1f  for sum w = %.0f  (%.4f per unit weight)"
      % (ll_true - ll_gauss, w_toy.sum(), (ll_true - ll_gauss) / w_toy.sum()))
print("note: the Gaussian point is compared at the TRUE mu, sigma; a fitted Gaussian would")
print("      absorb part of the difference into its width -- that is the full-range sigma of the README.")

# --------------------------------------------------------------------------
# 5. the JER rows: standardised response and its moments
# --------------------------------------------------------------------------
banner("5. the JER rows, 300 <= pt_gen < 500 GeV: what the Gaussian cannot reproduce")

candidates = [
    sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".npz") else None,
    "/scratch-cbe/users/robert.schoefbeck/jer-unbinned/campaign_UL18_absy0-0.5_ps1000.npz",
    os.path.expanduser("~/CMS/jer-unbinned/out/UL18/campaign_UL18_absy0-0.5_ps1000.npz"),
]
rows_file = next((f for f in candidates if f and os.path.exists(f)), None)
if rows_file is None:
    print("no rows file found (pass one as the first argument); skipping section 5")
    rows = bin_rows = None
else:
    rows = load(rows_file)
    bin_rows = select(rows, genpt=(300.0, 500.0), absy=(0.0, 0.5), max_dr=0.4, response=(0.0, 2.0))
    r, w = bin_rows.response, bin_rows.weight
    print("%s\n%d jets in the bin, sum w = %.4g pb, n_eff = %.0f" % (rows_file, len(r), w.sum(), bin_rows.n_eff))

    # Two Gaussian references: the full-range MLE and the +-1.5 sigma core fit (the
    # campaign convention).  jerfit.fit_bin / fit_core are the same fits as fit_response.py.
    g_full = fit_bin(r, w)
    g_core = fit_core(r, w, k=1.5)
    print("Gaussian MLE, full range: mu = %.4f  sigma = %.4f" % (g_full["mu"], g_full["sigma"]))
    print("Gaussian, +-1.5 sigma core: mu = %.4f  sigma = %.4f" % (g_core["mu"], g_core["sigma"]))

    # The weighted moments <phi_k> of the response under each reference.  Under the
    # full-range MLE, <phi_1> = <phi_2> = 0 by construction (the MLE equations ARE
    # those two moment conditions); degree 3 onwards is the shape information.
    # Under the core reference the low degrees are non-zero too: they say how the
    # full distribution differs from the core Gaussian.
    print("\nk   <phi_k> under full-range MLE        under core Gaussian")
    moments = {}
    for name, g in (("full", g_full), ("core", g_core)):
        moments[name] = weighted_moments(r, w, NormalParams(g["mu"], g["sigma"]), 2 * K)
    for k in range(1, K + 1):
        mf, ef = moments["full"][0][k], moments["full"][1][k]
        mc, ec = moments["core"][0][k], moments["core"][1][k]
        print("%-3d %9.4f +- %.4f (%6.1f sigma)   %9.4f +- %.4f (%6.1f sigma)"
              % (k, mf, ef, mf / ef, mc, ec, mc / ec))
    print("\nthese <phi_k> are the R_k a Born amplitude has to reproduce -- section 3 says how R_k depends on c.")
    print("the Gaussian predicts every one of them to be zero.  The error grows fast with k: phi_k(z) ~ z^k,")
    print("so a high moment is whatever the farthest few jets do.  Moments are the check; the fit is the likelihood.")

# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
try:
    import matplotlib
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(os.getcwd(), "fig", "hermite_demo")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(5):
        ax.plot(z, phi[:, k], label=r"$\varphi_%d$" % k)
    ax.set_xlabel(r"$z = (x-\mu)/\sigma$"); ax.set_ylim(-4, 4); ax.legend(ncol=3)
    ax.set_title("orthonormal Hermite basis of N(0,1)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "basis.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(z, p_ref * sigma, label=r"reference $N(\mu,\sigma^2)$")
    ax.semilogy(z, p_true * sigma, label=r"$p_{\rm ref}\,h^2$, $c_3=%.2f$, $c_4=%.2f$" % (c_true[3], c_true[4]))
    hist, edges = np.histogram((r_toy - mu) / sigma, bins=80, range=(-5, 5), weights=w_toy, density=True)
    ax.semilogy(z, p_approx * sigma, label=r"approximated")
    ax.step(edges[:-1], hist, where="post", lw=0.8, color="k", label="sample (weighted)")
    ax.set_ylim(1e-6, 1); ax.set_xlabel(r"$z$"); ax.legend(); ax.set_title("a low-side tail from degrees 3 and 4")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "tail.png"), dpi=150)

    for i in range(K):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(z, p_ref * sigma, label=r"reference $N(\mu,\sigma^2)$")
        ax.semilogy(z, p_true * sigma, label=r"$p_{\rm ref}\,h^2$, $c_3=%.2f$, $c_4=%.2f$" % (c_true[3], c_true[4]))
        hist, edges = np.histogram((r_toy - mu) / sigma, bins=80, range=(-5, 5), weights=w_toy, density=True)
        ax.semilogy(z, p_approx[i] * sigma, label=r"approximated with Hermite of %i th degree" % i)
        ax.step(edges[:-1], hist, where="post", lw=0.8, color="k", label="sample (weighted)")
        ax.set_ylim(1e-6, 1); ax.set_xlabel(r"$z$"); ax.legend(); ax.set_title("a low-side tail from degrees 3 and 4")
        file_name = "approx_Hermite_" + str(i) + ".png" 
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, file_name), dpi=150)

    if bin_rows is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        zz_data = (r - g_core["mu"]) / g_core["sigma"]
        hist, edges = np.histogram(zz_data, bins=100, range=(-8, 6), weights=w, density=True)
        ax.step(edges[:-1], hist, where="post", color="k", label="jets, 300-500 GeV (weighted)")
        zg = np.linspace(-8, 6, 500)
        ax.semilogy(zg, np.exp(-0.5 * zg**2) / np.sqrt(2 * np.pi), label="core Gaussian")
        s = g_full["sigma"] / g_core["sigma"]; d = (g_full["mu"] - g_core["mu"]) / g_core["sigma"]
        ax.semilogy(zg, np.exp(-0.5 * ((zg - d) / s) ** 2) / np.sqrt(2 * np.pi) / s, label="full-range Gaussian MLE")
        ax.set_ylim(1e-5, 1); ax.set_xlabel(r"$(r-\mu_{\rm core})/\sigma_{\rm core}$"); ax.legend()
        ax.set_title("the response and the two Gaussians")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "response.png"), dpi=150)
    print("\nfigures in", fig_dir)
except ImportError:
    print("\n(no matplotlib: figures skipped)")

# --------------------------------------------------------------------------
# 6. your task
# --------------------------------------------------------------------------
banner("6. your task")
print("""
Fit the Born model to the JER rows with ACTIVE degrees for the non-Gaussian
tails.  The pieces are all above; putting them together is the work.

  * The model:  p(r) = N(r; mu, sigma) h(r)^2,  h = sum_n c_n phi_n(r; mu, sigma),  ||c|| = 1.
    Decide which degrees are active.  Degrees 1 and 2 duplicate mu and sigma, so
    either fix c_1 = c_2 = 0 and fit (mu, sigma, c_3, c_4, ...) or fix (mu, sigma)
    and fit all c_n; not both.  Start with {3, 4}, then {3, 4, 5, 6}.
  * The objective: the weighted log-likelihood loglik() above, maximised with
    scipy.optimize.minimize.  The sphere: normalise c inside the objective (as
    loglik does) or parametrise it by angles.  Roots of h make log p = -inf at
    any row sitting exactly on them; the optimiser avoids such points by itself,
    but watch for it.
  * Errors: the likelihood is weighted, so use the sandwich covariance
    (jerfit/mle.py does this for the Gaussian fits; reuse it).
  * Checks, in this order: (i) the Gaussian limit -- with all c_n fixed to zero
    you must reproduce fit_bin exactly; (ii) the fitted R_k(c) against the
    measured <phi_k> of section 5, pull by pull; (iii) the log-likelihood gain
    over the Gaussian, and how it saturates with the number of active degrees;
    (iv) the response histogram on a log scale with p_ref h^2 drawn over it.
  * Then let mu and sigma follow the NSC form in pt:  jerfit/nsc.py takes any
    model with  names, bounds, init(rows), window(theta, pt), logpdf(theta, r, pt, a, b);
    logpdf is  log N(r; mu, sigma(pt)) + 2 log |h(r)|.  GaussianResponse in
    jerfit/gaussian.py is the reference implementation to copy from.
""")
