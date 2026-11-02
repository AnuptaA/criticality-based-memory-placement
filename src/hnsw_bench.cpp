#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// .fvecs / .ivecs: each row is an int32 dimension, then that many values
template <typename T>
static std::vector<T> read_vecs(const std::string& path, size_t& n, size_t& d) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { perror(path.c_str()); exit(1); }
    int dd;
    if (fread(&dd, 4, 1, f) != 1) { fprintf(stderr, "empty file\n"); exit(1); }
    d = dd;
    fseek(f, 0, SEEK_END);
    size_t bytes = ftell(f);
    n = bytes / (4 * (d + 1));
    fseek(f, 0, SEEK_SET);
    std::vector<T> out(n * d);
    for (size_t i = 0; i < n; i++) {
        int di;
        if (fread(&di, 4, 1, f) != 1 || (size_t)di != d) { fprintf(stderr, "bad row %zu\n", i); exit(1); }
        if (fread(&out[i * d], 4, d, f) != d) { fprintf(stderr, "short read\n"); exit(1); }
    }
    fclose(f);
    return out;
}

int main() {
    std::string home = getenv("HOME");
    std::string sift = home + "/sift/";
    std::string idxpath = home + "/indexes/sift_hnsw_M16.index";

    size_t nb, nq, ng, d, d2, kg;
    auto xb = read_vecs<float>(sift + "sift_base.fvecs", nb, d);
    auto xq = read_vecs<float>(sift + "sift_query.fvecs", nq, d2);
    auto gt = read_vecs<int>(sift + "sift_groundtruth.ivecs", ng, kg);
    printf("base=%zu query=%zu dim=%zu gt_per_query=%zu\n", nb, nq, d, kg);

    faiss::IndexHNSW* index;
    std::ifstream probe(idxpath);
    if (probe.good()) {
        index = dynamic_cast<faiss::IndexHNSW*>(faiss::read_index(idxpath.c_str()));
        printf("loaded index from %s\n", idxpath.c_str());
    } else {
        auto* h = new faiss::IndexHNSWFlat(d, 16);  // M = 16
        h->hnsw.efConstruction = 200;
        auto t0 = clk::now();
        h->add(nb, xb.data());
        printf("built index in %.1f s\n", secs(t0, clk::now()));
        faiss::write_index(h, idxpath.c_str());
        index = h;
    }

    omp_set_num_threads(1);  // single thread, so latency is meaningful
    const int k = 10;
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);
    int efs[] = {16, 32, 64, 128, 256};
    for (int ef : efs) {
        index->hnsw.efSearch = ef;
        index->search(1000, xq.data(), k, D.data(), I.data());  // warm-up
        auto t0 = clk::now();
        index->search(nq, xq.data(), k, D.data(), I.data());
        double sec = secs(t0, clk::now());
        size_t hits = 0;
        for (size_t q = 0; q < nq; q++)
            for (int j = 0; j < k; j++)
                for (int l = 0; l < k; l++)
                    if (I[q * k + j] == (faiss::idx_t)gt[q * kg + l]) hits++;
        printf("efSearch=%3d  recall@10=%.4f  QPS=%.0f  latency=%.1f us\n",
               ef, (double)hits / (nq * k), nq / sec, sec / nq * 1e6);
    }
    return 0;
}
