#!/usr/bin/env python3
"""Write a synthetic sample in the export schema, one npz per slice.
    python3 scripts/make_toy.py rows/toy [--n 20000] [--tail 0.03] [--seed 0]"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jerfit import synthetic

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--tail", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for r in synthetic.make(n_per_slice=a.n, tail_fraction=a.tail, seed=a.seed):
    p = os.path.join(a.out, "toy_HT%s.npz" % r.meta["slice"]); r.save(p); print(p, len(r))
print("truth:", synthetic.TRUE)
