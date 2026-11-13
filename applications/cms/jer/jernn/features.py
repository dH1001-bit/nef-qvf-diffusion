"""Variables of the network, given as short text specs, so that a new variable is a new spec on
the command line and not a code change.

A spec is ``column[:transform]``:

    genpt:log      ln(genpt)
    receta:abs     |receta|
    rho            rho as it is
    response       a derived column (recpt / genpt), see DERIVED

The target (the variable whose conditional density is learned) and every condition use the same
specs.  A column that is missing in a file is looked up under its aliases (the two exports name
the matching radius dR and dr) and then in DERIVED; anything else is an error naming the
columns that the file does have.

Cuts are specs too: ``column op value`` or ``abs(column) op value`` with op one of
< <= > >= == !=, e.g. ``dR<0.4``, ``genpt>=97``, ``abs(receta)<5``.
"""
from __future__ import annotations

import re

import numpy as np

ALIASES = {"dR": ("dr",), "dr": ("dR",)}

# derived columns: name -> (columns needed, function of a dict of those columns).  A new derived
# variable (e.g. a dijet balance for real data) is one entry here.
DERIVED = {
    "response": (("recpt", "genpt"), lambda c: c["recpt"] / c["genpt"]),
}

TRANSFORMS = {
    "id": (lambda v: v, lambda v: v),
    "log": (np.log, np.exp),
    "abs": (np.abs, None),
    "log1p": (np.log1p, np.expm1),
}


def resolve(name, available):
    """Which stored column(s) give ``name``: (kind, payload) with kind 'col' or 'derived'."""
    if name in available:
        return "col", name
    for alt in ALIASES.get(name, ()):
        if alt in available:
            return "col", alt
    if name in DERIVED:
        need, _ = DERIVED[name]
        missing = [n for n in need if resolve_or_none(n, available) is None]
        if not missing:
            return "derived", name
    raise KeyError("column %r not found; available: %s" % (name, ", ".join(sorted(available))))


def resolve_or_none(name, available):
    try:
        return resolve(name, available)
    except KeyError:
        return None


def raw_column(name, getter, available):
    """The raw values of ``name`` (stored, aliased or derived) as float64."""
    kind, what = resolve(name, available)
    if kind == "col":
        return np.asarray(getter(what), dtype=np.float64)
    need, fn = DERIVED[what]
    return np.asarray(fn({n: raw_column(n, getter, available) for n in need}), dtype=np.float64)


class Feature:
    """One variable: ``Feature('genpt:log')``."""

    def __init__(self, spec: str):
        self.spec = spec.strip()
        col, _, tr = self.spec.partition(":")
        self.column, self.transform = col, (tr or "id")
        if self.transform not in TRANSFORMS:
            raise ValueError("unknown transform %r in %r (known: %s)" % (tr, spec, ", ".join(TRANSFORMS)))

    def __call__(self, getter, available):
        return TRANSFORMS[self.transform][0](raw_column(self.column, getter, available))

    def inverse(self, v):
        inv = TRANSFORMS[self.transform][1]
        if inv is None:
            raise ValueError("transform %r of %r has no inverse" % (self.transform, self.spec))
        return inv(v)

    @property
    def label(self):
        return self.column if self.transform == "id" else "%s(%s)" % (self.transform, self.column)

    def __repr__(self):
        return "Feature(%r)" % self.spec


_CUT = re.compile(r"^\s*(abs\(\s*)?([A-Za-z_]\w*)\s*\)?\s*(<=|>=|==|!=|<|>)\s*([-+0-9.eE]+|inf)\s*$")
_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
        "==": np.equal, "!=": np.not_equal}


class Cut:
    """One cut: ``Cut('dR<0.4')``, ``Cut('abs(receta)<5')``."""

    def __init__(self, spec: str):
        m = _CUT.match(spec)
        if not m:
            raise ValueError("cannot parse cut %r (form: col<val, abs(col)>=val)" % spec)
        self.spec = spec.strip()
        self.absval, self.column, self.op, self.value = bool(m.group(1)), m.group(2), m.group(3), float(m.group(4))

    def mask(self, getter, available):
        v = raw_column(self.column, getter, available)
        return _OPS[self.op](np.abs(v) if self.absval else v, self.value)

    def __repr__(self):
        return "Cut(%r)" % self.spec
