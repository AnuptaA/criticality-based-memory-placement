#!/usr/bin/env python3
import sys, os, shutil

path = os.path.expanduser("~/faiss/faiss/impl/HNSW.cpp")
src = open(path).read()
if "g_tr_on" in src:
    print("already patched"); sys.exit(0)

def once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        print(f"ABORT: anchor '{what}' found {n} times (expected 1)"); sys.exit(1)
    return text.replace(old, new)

def func_slice(text, header):
    n = text.count(header)
    if n != 1:
        print(f"ABORT: header '{header}' found {n} times"); sys.exit(1)
    start = text.index(header)
    return start, text.index("\n}\n", start) + 3

inc = "#include <faiss/impl/VisitedTable.h>\n"
block = inc + '''
// ---- access tracing (added for the memory-placement study) ----
#include <cstdint>
bool g_tr_on = false;
uint32_t* g_tr_vec_hist = nullptr;  // [n*4]: vector reads by group size 1..4
uint32_t* g_tr_exp0 = nullptr;      // [n]: level-0 expansions (neighbor-list reads)
uint32_t* g_tr_expU = nullptr;      // [n]: upper-level expansions
uint64_t* g_tr_newsum = nullptr;    // [n]: new neighbors evaluated per level-0 expansion
#define TR_VEC(ids, n) do { if (g_tr_on) { for (int t_ = 0; t_ < (n); t_++) g_tr_vec_hist[(size_t)(ids)[t_] * 4 + ((n) - 1)]++; } } while (0)
// ---- end tracing ----
'''
src = once(src, inc, block, "include")

s, e = func_slice(src, "HNSWStats greedy_update_nearest_impl(")
f = src[s:e]
f = once(f, "        hnsw.neighbor_range(nearest, level, &begin, &end);\n",
         "        hnsw.neighbor_range(nearest, level, &begin, &end);\n        if (g_tr_on) g_tr_expU[nearest]++;\n", "upper expansion")
f = once(f, "                qdis.distances_batch_4(",
         "                TR_VEC(buffered_ids, 4);\n                qdis.distances_batch_4(", "upper batch")
f = once(f, "            float dis = qdis(buffered_ids[icnt]);",
         "            if (g_tr_on) g_tr_vec_hist[(size_t)buffered_ids[icnt] * 4]++;\n            float dis = qdis(buffered_ids[icnt]);", "upper leftover")
src = src[:s] + f + src[e:]

s, e = func_slice(src, "int search_from_candidates_fixVT(")
f = src[s:e]
f = once(f, "        int v0 = candidates.pop_min(&d0);\n",
         "        int v0 = candidates.pop_min(&d0);\n        if (g_tr_on) g_tr_exp0[v0]++;\n", "level0 pop")
old = "ndis += evaluator.evaluate("
if f.count(old) != 2:
    print(f"ABORT: expected 2 evaluate calls, found {f.count(old)}"); sys.exit(1)
f = f.replace(old, "TR_VEC(saved_j, counter); if (g_tr_on) g_tr_newsum[v0] += counter;\n            " + old)
src = src[:s] + f + src[e:]

shutil.copy(path, path + ".orig")
open(path, "w").write(src)
print("patched OK; backup at", path + ".orig")
