#!/usr/bin/env python3
import numpy as np, os

T = os.path.expanduser("~/traces/")
g = np.array([1.0, 2.0, 3.0, 4.0])

for ef in (16, 64, 256):
    vh = np.fromfile(T + f"ef{ef}_vec_hist.bin", dtype=np.uint32).reshape(-1, 4).astype(np.float64)
    e0 = np.fromfile(T + f"ef{ef}_exp0.bin", dtype=np.uint32).astype(np.float64)
    n = len(e0)
    # vector pages: 8 vectors (512 B each) per 4 KB page
    vp_h = vh.sum(1).reshape(-1, 8).sum(1)
    vp_e = (vh / g).sum(1).reshape(-1, 8).sum(1)
    # level-0 neighbor-list pages: 32 nodes (128 B each) per page (approximate layout)
    pad = (-n) % 32
    np_h = np.pad(e0, (0, pad)).reshape(-1, 32).sum(1)
    np_e = np_h.copy()  # serial reads, weight 1.0
    H = np.concatenate([vp_h, np_h])
    E = np.concatenate([vp_e, np_e])
    is_nbr = np.concatenate([np.zeros(len(vp_h), bool), np.ones(len(np_h), bool)])
    print(f"\nefSearch={ef}  pages={len(H)} (vector {len(vp_h)}, neighbor {len(np_h)})")
    print("  budget  hot:nbr-pages  crit:nbr-pages  hot-captures  crit-captures  gap(pts)")
    for b in (0.01, 0.02, 0.05, 0.10, 0.25):
        m = int(b * len(H))
        th = np.argsort(-H, kind="stable")[:m]
        te = np.argsort(-E, kind="stable")[:m]
        ch = E[th].sum() / E.sum()
        ce = E[te].sum() / E.sum()
        print(f"  {b:5.0%}   {is_nbr[th].mean():10.1%}   {is_nbr[te].mean():12.1%}   {ch:10.1%}   {ce:11.1%}   {100*(ce-ch):7.2f}")
