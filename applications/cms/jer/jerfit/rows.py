"""The per-jet row schema shared by the export and the fit.

A row is one reconstructed jet (accepted: jet weight > 0) paired with its
nearest generator jet.  Files are ``.npz`` with the arrays

    response    pt_rec(corrected) / pt_gen
    genpt       generator-jet pt                                 [GeV]
    geny        generator-jet rapidity
    recpt       JES-corrected reconstructed pt                   [GeV]
    recy        reconstructed rapidity (of the RAW p4, as Darwin does)
    weight      genEvent.Weight() * jet.Weight()   -- a cross section in pb
    dr          Delta R(rec, nearest gen)
    dr2         Delta R(rec, second-nearest gen); inf if none

and the scalars ``era``, ``slice``, ``source`` (strings), ``n_events``,
``sumw_events`` (events read, sum of their generator weights), plus the
generator-jet table for the matching efficiency

    g_pt, g_y   every generator jet above the export threshold
    g_dr        Delta R to the nearest ACCEPTED reconstructed jet; inf if none

Rejected jets (``jet.Weight() == 0``) are never written.  The export keeps
every pair up to a loose radius so the matching radius is an analysis cut
(``select(..., max_dr=...)``), not an export choice.

The campaign's own flat export (0070 ``export_jer_rows.py``, files
``jer_<era>_HT<slice>_<k>.npz``) is read by the same ``load``: its columns
``genpt geneta geny recpt receta recy dR area rho trpu nVtx weight`` are taken
over, ``dR`` is renamed ``dr``, ``response = recpt/genpt`` is computed, and
the scalars ``n_events_read`` / ``sum_weights`` become ``n_events`` /
``sumw_events``.  It has no ``dr2`` (so ``isolate_dr2`` is unavailable) and no
generator-jet table (so no matching efficiency); it carries pile-up columns
that this export does not.
"""
from __future__ import annotations
import glob
import os
import numpy as np

PAIR_COLUMNS = ("response", "genpt", "geny", "recpt", "recy", "weight", "dr", "dr2")
GEN_COLUMNS = ("g_pt", "g_y", "g_dr")


class Rows:
    """A column store.  ``r.response`` etc. are numpy arrays of equal length;
    ``r.gen`` is the generator-jet table (dict of arrays) or None."""

    def __init__(self, cols, gen=None, meta=None):
        n = {len(v) for v in cols.values()}
        if len(n) != 1:
            raise ValueError("unequal column lengths: %r" % {k: len(v) for k, v in cols.items()})
        self._cols = {k: np.asarray(v) for k, v in cols.items()}
        self.gen = None if gen is None else {k: np.asarray(v) for k, v in gen.items()}
        self.meta = dict(meta or {})

    def __getattr__(self, name):
        try:
            return self._cols[name]
        except KeyError:
            raise AttributeError(name)

    def __len__(self):
        return len(next(iter(self._cols.values())))

    def __getitem__(self, mask):
        return Rows({k: v[mask] for k, v in self._cols.items()}, gen=self.gen, meta=self.meta)

    @property
    def columns(self):
        return tuple(self._cols)

    @property
    def sumw(self):
        return float(self.weight.sum())

    @property
    def n_eff(self):
        """Kish effective sample size (sum w)^2 / sum w^2."""
        w = self.weight
        return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0

    def save(self, path):
        out = dict(self._cols)
        if self.gen is not None:
            out.update(self.gen)
        for k, v in self.meta.items():
            out[k] = np.asarray(v)
        np.savez_compressed(path, **out)


CAMPAIGN_COLUMNS = {"dR": "dr"}
CAMPAIGN_META = {"n_events_read": "n_events", "sum_weights": "sumw_events"}


def load(path) -> Rows:
    """Load one npz written by the export, by Rows.save, or by the campaign's
    flat export (recognised by its ``dR`` column; see the module docstring)."""
    cols, gen, meta = {}, {}, {}
    with np.load(path, allow_pickle=False) as f:
        campaign = "response" not in f.files and "dR" in f.files
        if "response" not in f.files and not campaign:
            raise ValueError("%s has neither a 'response' nor a 'dR' column" % path)
        n = len(f["recpt" if campaign else "response"])
        for k in f.files:
            v = f[k]
            if k.startswith("g_"):
                gen[k] = v
            elif v.ndim == 0:
                meta[CAMPAIGN_META.get(k, k) if campaign else k] = v[()]
            elif v.ndim == 1 and len(v) == n:
                cols[CAMPAIGN_COLUMNS.get(k, k) if campaign else k] = v   # schema columns plus any extra per-row column
            else:
                meta[k] = v
    if campaign:
        cols["response"] = cols["recpt"].astype(np.float64) / cols["genpt"].astype(np.float64)
        meta.setdefault("source", "campaign flat export (0070 export_jer_rows.py)")
    gen = gen or None
    return Rows(cols, gen=gen, meta=meta)


def concatenate(paths_or_rows) -> Rows:
    """Concatenate files or Rows.  Weights are cross sections, so slices and
    files simply add; nothing is rescaled.  Per-file metadata is kept as
    ``meta['files']`` (list of dicts) and integer ``file_id`` / ``slice_id`` /
    ``era_id`` columns are added, with lookup tables in meta."""
    items = []
    for p in paths_or_rows:
        if isinstance(p, Rows):
            items.append(p)
        else:
            items.extend(load(q) for q in sorted(glob.glob(p)) or [p])
    if not items:
        raise ValueError("nothing to concatenate")
    eras, slices = [], []
    cols = {k: [] for k in items[0].columns}
    gen = {k: [] for k in GEN_COLUMNS} if items[0].gen is not None else None
    file_id, slice_id, era_id, files = [], [], [], []
    sumw_events = 0.0
    n_events = 0
    for i, r in enumerate(items):
        era = str(r.meta.get("era", "")); sl = str(r.meta.get("slice", ""))
        if era not in eras: eras.append(era)
        if sl not in slices: slices.append(sl)
        for k in cols: cols[k].append(getattr(r, k))
        if gen is not None and r.gen is not None:
            for k in gen: gen[k].append(r.gen[k])
        n = len(r)
        file_id.append(np.full(n, i, np.int32))
        slice_id.append(np.full(n, slices.index(sl), np.int16))
        era_id.append(np.full(n, eras.index(era), np.int16))
        files.append({k: (v.item() if getattr(v, "ndim", 1) == 0 else getattr(v, "tolist", lambda: v)())
                      for k, v in r.meta.items()})
        sumw_events += float(r.meta.get("sumw_events", 0.0))
        n_events += int(r.meta.get("n_events", 0))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["file_id"] = np.concatenate(file_id)
    out["slice_id"] = np.concatenate(slice_id)
    out["era_id"] = np.concatenate(era_id)
    gen_out = None if gen is None else {k: np.concatenate(v) for k, v in gen.items()}
    meta = dict(files=files, eras=eras, slices=slices,
                sumw_events=sumw_events, n_events=n_events)
    return Rows(out, gen=gen_out, meta=meta)


def select(rows: Rows, genpt=(None, None), absy=(None, None), max_dr=0.4,
           isolate_dr2=None, response=(None, None)) -> Rows:
    """The analysis cuts.  ``genpt``/``absy``/``response`` are (lo, hi) with
    None for open; ``max_dr`` is the matching radius (JME: R/2); ``isolate_dr2``
    demands the second-nearest generator jet further than that."""
    m = np.ones(len(rows), bool)
    if max_dr is not None:
        m &= rows.dr < max_dr
    lo, hi = genpt
    if lo is not None: m &= rows.genpt >= lo
    if hi is not None: m &= rows.genpt < hi
    lo, hi = absy
    ay = np.abs(rows.recy)
    if lo is not None: m &= ay >= lo
    if hi is not None: m &= ay < hi
    lo, hi = response
    if lo is not None: m &= rows.response >= lo
    if hi is not None: m &= rows.response < hi
    if isolate_dr2 is not None:
        if "dr2" not in rows.columns:
            raise ValueError("isolate_dr2 needs the 'dr2' column, which the campaign's flat export does not carry")
        m &= rows.dr2 > isolate_dr2
    return rows[m]


def matching_efficiency(rows: Rows, pt_edges, max_dr=0.4, absy_max=None):
    """Fraction of generator jets with an accepted reconstructed jet within
    max_dr, per generator-pt bin (unweighted counts; the export's gen table).
    Returns (centres, efficiency, n_gen)."""
    g = rows.gen
    if g is None:
        raise ValueError("rows carry no generator-jet table")
    pt, y, dr = g["g_pt"], g["g_y"], g["g_dr"]
    m = np.ones(len(pt), bool)
    if absy_max is not None:
        m &= np.abs(y) < absy_max
    pt_edges = np.asarray(pt_edges, float)
    idx = np.digitize(pt[m], pt_edges) - 1
    ok = (idx >= 0) & (idx < len(pt_edges) - 1)
    n = np.bincount(idx[ok], minlength=len(pt_edges) - 1)
    k = np.bincount(idx[ok], weights=(dr[m][ok] < max_dr), minlength=len(pt_edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = k / n
    return 0.5 * (pt_edges[1:] + pt_edges[:-1]), eff, n


def flatten(rows: Rows, n_bins=40, pt_range=None) -> Rows:
    """Divide each weight by the mean weight of its generator-pt bin (bins
    equal in log pt), so every pt region carries its own statistical power.

    Why this is legitimate: the fit models the CONDITIONAL law p(r | pt).
    Multiplying the weights by any positive function g(pt) leaves the
    weighted-likelihood estimator consistent -- it only changes how the pt
    regions are balanced -- while the relative weights of rows at the same
    pt (which encode the stitching of the H_T slices and the sample
    composition) are kept.  Cross-section weights fall like pt^-5, so without
    this the global likelihood is the lowest-pt rows and the constant term is
    unconstrained; the per-bin convention does the same renormalisation
    implicitly, one bin at a time.  The flattened weights are still weights:
    the sandwich errors remain valid."""
    pt = rows.genpt
    lo, hi = pt_range or (pt.min(), pt.max() * (1 + 1e-9))
    edges = np.geomspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pt, edges) - 1, 0, n_bins - 1)
    sw = np.bincount(idx, weights=rows.weight, minlength=n_bins)
    n = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_w = np.where(n > 0, sw / np.maximum(n, 1), 1.0)
    cols = {k: getattr(rows, k) for k in rows.columns}
    cols["weight"] = rows.weight / mean_w[idx]
    cols["weight_xsec"] = rows.weight if "weight_xsec" not in rows.columns else rows.weight_xsec
    out = Rows(cols, gen=rows.gen, meta=dict(rows.meta, flattened=dict(n_bins=n_bins, edges=edges.tolist())))
    return out
