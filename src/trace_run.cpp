#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern bool g_tr_on;
extern uint32_t* g_tr_vec_hist;
extern uint32_t* g_tr_exp0;
extern uint32_t* g_tr_expU;
extern uint64_t* g_tr_newsum;

static std::vector<float> read_fvecs(const std::string& p, size_t& n, size_t& d) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { perror(p.c_str()); exit(1); }
    int dd; if (fread(&dd, 4, 1, f) != 1) exit(1);
    d = dd; fseek(f, 0, SEEK_END); n = ftell(f) / (4 * (d + 1)); fseek(f, 0, SEEK_SET);
    std::vector<float> out(n * d);
    for (size_t i = 0; i < n; i++) {
        int di; if (fread(&di, 4, 1, f) != 1) exit(1);
        if (fread(&out[i * d], 4, d, f) != d) exit(1);
    }
    fclose(f); return out;
}

static void share(const char* name, std::vector<uint64_t> v, uint64_t total) {
    std::sort(v.begin(), v.end(), std::greater<uint64_t>());
    printf("%s top-node shares:", name);
    double fr[] = {0.01, 0.05, 0.10, 0.25};
    for (double x : fr) {
        size_t m = (size_t)(x * v.size()); uint64_t s = 0;
        for (size_t i = 0; i < m; i++) s += v[i];
        printf("  top %2.0f%% -> %5.1f%%", x * 100, 100.0 * s / total);
    }
    printf("\n");
}

static void dump(const std::string& p, const void* data, size_t bytes) {
    FILE* f = fopen(p.c_str(), "wb"); fwrite(data, 1, bytes, f); fclose(f);
}

int main(int argc, char** argv) {
    int ef = argc > 1 ? atoi(argv[1]) : 64;
    std::string home = getenv("HOME");
    size_t nq, d;
    auto xq = read_fvecs(home + "/sift/sift_query.fvecs", nq, d);
    auto* index = dynamic_cast<faiss::IndexHNSW*>(
        faiss::read_index((home + "/indexes/sift_hnsw_M16.index").c_str()));
    size_t n = index->ntotal;
    g_tr_vec_hist = (uint32_t*)calloc(n * 4, 4);
    g_tr_exp0 = (uint32_t*)calloc(n, 4);
    g_tr_expU = (uint32_t*)calloc(n, 4);
    g_tr_newsum = (uint64_t*)calloc(n, 8);

    omp_set_num_threads(1);
    index->hnsw.efSearch = ef;
    const int k = 10;
    std::vector<faiss::idx_t> I(nq * k); std::vector<float> D(nq * k);
    g_tr_on = true;
    index->search(nq, xq.data(), k, D.data(), I.data());
    g_tr_on = false;

    uint64_t bs[4] = {0, 0, 0, 0}, tot_vec = 0, tot_e0 = 0, tot_eU = 0;
    std::vector<uint64_t> vreads(n), nreads(n);
    for (size_t i = 0; i < n; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 4; j++) { bs[j] += g_tr_vec_hist[i * 4 + j]; v += g_tr_vec_hist[i * 4 + j]; }
        vreads[i] = v; nreads[i] = (uint64_t)g_tr_exp0[i] + g_tr_expU[i];
        tot_vec += v; tot_e0 += g_tr_exp0[i]; tot_eU += g_tr_expU[i];
    }
    printf("efSearch=%d  queries=%zu\n", ef, nq);
    printf("per query: vector reads=%.0f  L0 expansions=%.0f  upper expansions=%.0f\n",
           (double)tot_vec / nq, (double)tot_e0 / nq, (double)tot_eU / nq);
    printf("vector reads by group size: 1:%.1f%%  2:%.1f%%  3:%.1f%%  4:%.1f%%\n",
           100.0 * bs[0] / tot_vec, 100.0 * bs[1] / tot_vec, 100.0 * bs[2] / tot_vec, 100.0 * bs[3] / tot_vec);
    share("vector reads  ", vreads, tot_vec);
    share("nbr-list reads", nreads, tot_e0 + tot_eU);

    std::string pre = home + "/traces/ef" + std::to_string(ef) + "_";
    dump(pre + "vec_hist.bin", g_tr_vec_hist, n * 4 * 4);
    dump(pre + "exp0.bin", g_tr_exp0, n * 4);
    dump(pre + "expU.bin", g_tr_expU, n * 4);
    dump(pre + "newsum.bin", g_tr_newsum, n * 8);
    return 0;
}
