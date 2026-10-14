"""Unbinned jet-energy-resolution fit.

Depends on numpy and scipy only.  Input rows come from the one Darwin
dependent script in ``export/`` or from ``jerfit.synthetic``; both write the
schema documented in ``jerfit.rows``.
"""
from .rows import Rows, load, concatenate, select, flatten, matching_efficiency
from .width import nsc_sigma, NSC_NAMES
from .gaussian import GaussianResponse, fit_bin, fit_core
from .nsc import fit_nsc, FitResult
