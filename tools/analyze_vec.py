#!/usr/bin/env python3
import numpy as np, os

T = os.path.expanduser("~/traces/")
g = np.array([1.0, 2.0, 3.0, 4.0])

def rank(a):
    r = np.empty(len(a)); r[np.argsort(a, kind="stable")] = np.arange(len(a)); return r

def report(label, h, e):
    used = h > 0
    rho = np.corrcoef(rank(h[used]), rank(e[used]))[0, 1]
    print(f"  {label}: objects={len(h)} touched={used.sum()} spearman={rho:.4f}")
    print("    budget  overlap  hot-captures  crit-captures  gap(pts)")
    for b in (0.01, 0.05, 0.10, 0.25):
        m = int(b * len(h))
        top_h = np.argsort(-h, kind="stable")[:m]
        top_e = np.argsort(-e, kind="stable")[:m]
        ov = len(np.intersect1d(top_h, top_e)) / m
        ch = e[top_h].sum() / e.sum()
        ce = e[top_e].sum() / e.sum()
        print(f"    {b:5.0%}   {ov:6.1%}   {ch:11.1%}   {ce:12.1%}   {100*(ce-ch):7.2f}")

for ef in (16, 64, 256):
    vh = np.fromfile(T + f"ef{ef}_vec_hist.bin", dtype=np.uint32).reshape(-1, 4).astype(np.float64)
    H = vh.sum(axis=1)
    E = (vh / g).sum(axis=1)
    print(f"\nefSearch={ef}")
    report("per vector", H, E)
    report("per 4KB page", H.reshape(-1, 8).sum(axis=1), E.reshape(-1, 8).sum(axis=1))
