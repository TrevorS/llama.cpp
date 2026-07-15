// fp4_oracle.cpp
// -----------------------------------------------------------------------------
// DSV4 lightning-indexer fp4 selection-fidelity oracle (CPU-only).
//
// Question it answers: if the indexer score kernel quantizes its inputs to
// e2m1 (MXFP4-style, block-32 power-of-2 scales) the way the model was
// QAT'd — see ds4.c dsv4_indexer_qat_row_inplace_cpu — how much does the
// per-token top-512 SELECTION SET change vs the current fp16/fp32 path?
//
// Two comparisons per config:
//   A (current llama.cpp): rotated q/k, no quantization
//   B (QAT / fp4 kernel) : rotated q/k, e2m1 round-trip on BOTH q and k
//   overlap(A,B) per token = |A ∩ B| / top_k
// ds4.c's comment says B *is* the official graph ("without it, the top-k
// compressed-row selection is not the model's graph"), so overlap(A,B) also
// measures how far our current fp16 path deviates from reference serving.
//
// Quantization functions are verbatim ports of ds4.c (lines ~2537-2604).
//
// Modes:
//   synthetic (default): gaussian q/k/w, sweep n_lid x seeds
//     ./fp4-oracle [--nt N] [--topk K] [--heads H] [--seeds S]
//   dump mode: raw float32 files, layout q[nt][heads][128] k[n_lid][128]
//     w[nt][heads] (pre-rotation, pre-quant — i.e., captured BEFORE the
//     graph's hadamard, or pass --pre-rotated if captured after it)
//     ./fp4-oracle --q q.bin --k k.bin --w w.bin --nt N --nlid M [--pre-rotated]
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

static constexpr int D_IDX = 128;

// ---- verbatim ports from ds4.c ----------------------------------------------

static float dsv4_e2m1fn_value_cpu(int i) {
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    return values[i & 7];
}

static float dsv4_e2m1fn_dequant_cpu(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_cpu(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - dsv4_e2m1fn_value_cpu(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_cpu(best);
}

static void dsv4_hadamard128_inplace_cpu(float * x) {
    for (uint32_t stride = 1; stride < 128; stride <<= 1) {
        for (uint32_t base = 0; base < 128; base += 2u * stride) {
            for (uint32_t i = 0; i < stride; i++) {
                const float a = x[base + i];
                const float b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const float scale = 0.08838834764831845f; // 1/sqrt(128)
    for (uint32_t i = 0; i < 128; i++) x[i] *= scale;
}

static void dsv4_fp4_act_quantize_row_inplace_cpu(float * x, uint32_t n) {
    for (uint32_t off = 0; off < n; off += 32) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 32; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }
        if (amax < 7.052966104933725e-38f) amax = 7.052966104933725e-38f;
        const float scale = ldexpf(1.0f, (int) ceilf(log2f(amax / 6.0f)));
        for (uint32_t i = 0; i < 32; i++) {
            float v = x[off + i] / scale;
            if (v > 6.0f) v = 6.0f;
            if (v < -6.0f) v = -6.0f;
            x[off + i] = dsv4_e2m1fn_dequant_cpu(v) * scale;
        }
    }
}

// ---- oracle ------------------------------------------------------------------

// score(t,j) = sum_h relu(q[t,h,:] . k[j,:]) * w[t,h]   (deep-prefill regime:
// every cached comp cell is visible, mask contributes a shared 0)
static void scores_row(const float * q_t, const float * w_t, const float * k,
                       int n_lid, int n_head, float * out) {
    for (int j = 0; j < n_lid; j++) {
        const float * kj = k + (size_t) j * D_IDX;
        float acc = 0.0f;
        for (int h = 0; h < n_head; h++) {
            const float * qh = q_t + (size_t) h * D_IDX;
            float dot = 0.0f;
            for (int d = 0; d < D_IDX; d++) dot += qh[d] * kj[d];
            acc += fmaxf(dot, 0.0f) * w_t[h];
        }
        out[j] = acc;
    }
}

// top-k indices, descending score, lower-index tie-break (matches ggml_top_k
// and the CUDA op's selection rule)
static void top_k_set(const float * s, int n, int k, std::vector<int> & out) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return s[a] > s[b] || (s[a] == s[b] && a < b);
    });
    out.assign(idx.begin(), idx.begin() + k);
    std::sort(out.begin(), out.end());
}

static double overlap(const std::vector<int> & a, const std::vector<int> & b) {
    size_t i = 0, j = 0, common = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { common++; i++; j++; }
        else if (a[i] < b[j]) i++;
        else j++;
    }
    return (double) common / (double) a.size();
}

static bool load_f32(const char * path, std::vector<float> & v, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    v.resize(n);
    const size_t rd = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (rd != n) { fprintf(stderr, "%s: expected %zu floats, got %zu\n", path, n, rd); return false; }
    return true;
}

int main(int argc, char ** argv) {
    int nt = 64, top_k = 512, n_head = 64, n_seeds = 3;
    const char *qf = nullptr, *kf = nullptr, *wf = nullptr;
    int nlid_arg = 0;
    bool pre_rotated = false;
    bool int8_disp   = false;
    bool f16_scores  = false; // model the f16 scores-buffer store in pass 1
    bool fp4mma_disp = false; // model the fp4-mma (e2m1xe2m1) pass-1 displacement

    for (int i = 1; i < argc; i++) {
        auto next = [&](int & i) { return atoi(argv[++i]); };
        if      (!strcmp(argv[i], "--nt"))     nt      = next(i);
        else if (!strcmp(argv[i], "--topk"))   top_k   = next(i);
        else if (!strcmp(argv[i], "--heads"))  n_head  = next(i);
        else if (!strcmp(argv[i], "--seeds"))  n_seeds = next(i);
        else if (!strcmp(argv[i], "--nlid"))   nlid_arg = next(i);
        else if (!strcmp(argv[i], "--q"))      qf = argv[++i];
        else if (!strcmp(argv[i], "--k"))      kf = argv[++i];
        else if (!strcmp(argv[i], "--w"))      wf = argv[++i];
        else if (!strcmp(argv[i], "--pre-rotated")) pre_rotated = true;
        else if (!strcmp(argv[i], "--int8-displacement")) int8_disp = true;
        else if (!strcmp(argv[i], "--f16-scores")) f16_scores = true;
        else if (!strcmp(argv[i], "--fp4-mma-displacement")) fp4mma_disp = true;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }

    // fp4-mma mode also emits the int8 pass-1 displacement for a direct A/B.
    // Both real kernels store scores as f16, so compare both under the f16 store.
    if (fp4mma_disp) { int8_disp = true; f16_scores = true; }

    const bool dump_mode = qf && kf && wf;
    std::vector<int> nlids = dump_mode ? std::vector<int>{nlid_arg}
                                       : std::vector<int>{2176, 8704, 33280};
    if (dump_mode && nlid_arg <= 0) { fprintf(stderr, "--nlid required in dump mode\n"); return 1; }

    printf("fp4 selection-fidelity oracle: nt=%d heads=%d top_k=%d %s\n",
           nt, n_head, top_k, dump_mode ? "(dump mode)" : "(synthetic)");
    printf("%-8s %-6s %-10s %-10s %-10s %-12s\n",
           "n_lid", "seed", "mean_ovl", "min_ovl", "p1_ovl", "tok<0.98");

    for (int n_lid : nlids) {
        for (int seed = 0; seed < (dump_mode ? 1 : n_seeds); seed++) {
            std::vector<float> q, k, w;
            if (dump_mode) {
                if (!load_f32(qf, q, (size_t) nt * n_head * D_IDX) ||
                    !load_f32(kf, k, (size_t) n_lid * D_IDX)       ||
                    !load_f32(wf, w, (size_t) nt * n_head)) return 1;
            } else {
                std::mt19937 rng(1234 + seed);
                std::normal_distribution<float> nd(0.0f, 1.0f);
                q.resize((size_t) nt * n_head * D_IDX);
                k.resize((size_t) n_lid * D_IDX);
                w.resize((size_t) nt * n_head);
                for (auto & v : q) v = nd(rng);
                for (auto & v : k) v = nd(rng);
                for (auto & v : w) v = nd(rng) / sqrtf((float)(D_IDX * n_head));
            }

            // path A: rotated, unquantized (current llama.cpp fp16/fp32 graph)
            std::vector<float> qa = q, ka = k;
            if (!pre_rotated) {
                for (size_t r = 0; r < qa.size() / D_IDX; r++) dsv4_hadamard128_inplace_cpu(qa.data() + r * D_IDX);
                for (size_t r = 0; r < ka.size() / D_IDX; r++) dsv4_hadamard128_inplace_cpu(ka.data() + r * D_IDX);
            }
            // path B: rotated + e2m1 round-trip on q AND k (QAT / fp4 kernel)
            std::vector<float> qb = qa, kb = ka;
            for (size_t r = 0; r < qb.size() / D_IDX; r++) dsv4_fp4_act_quantize_row_inplace_cpu(qb.data() + r * D_IDX, D_IDX);
            for (size_t r = 0; r < kb.size() / D_IDX; r++) dsv4_fp4_act_quantize_row_inplace_cpu(kb.data() + r * D_IDX, D_IDX);

            std::vector<double> ovl(nt);
            const int kk = std::min(top_k, n_lid);
            #pragma omp parallel for schedule(dynamic)
            for (int t = 0; t < nt; t++) {
                std::vector<float> sa(n_lid), sb(n_lid);
                std::vector<int> ta, tb;
                const float * w_t = w.data() + (size_t) t * n_head;
                scores_row(qa.data() + (size_t) t * n_head * D_IDX, w_t, ka.data(), n_lid, n_head, sa.data());
                scores_row(qb.data() + (size_t) t * n_head * D_IDX, w_t, kb.data(), n_lid, n_head, sb.data());
                top_k_set(sa.data(), n_lid, kk, ta);
                top_k_set(sb.data(), n_lid, kk, tb);
                ovl[t] = overlap(ta, tb);
            }
            std::sort(ovl.begin(), ovl.end());
            const double mean = std::accumulate(ovl.begin(), ovl.end(), 0.0) / nt;
            const int low = (int) std::count_if(ovl.begin(), ovl.end(), [](double o) { return o < 0.98; });
            printf("%-8d %-6d %-10.4f %-10.4f %-10.4f %d/%d\n",
                   n_lid, seed, mean, ovl.front(), ovl[(size_t) (0.01 * nt)], low, nt);

            // --int8-displacement: two-pass rescore design input. Pass 1 of the
            // exact pipeline ranks with the int8 kernel numerics OVER THE QAT
            // VALUES (path B); pass 2 rescores the top-(512+m) candidates
            // exactly. m must cover the worst rank displacement of any true
            // (path-B) top-512 member under the int8 ranking. Int8 scheme
            // matches dsv4_score_int8_kernel: per-token symmetric q scale
            // (amax/127 over all heads*dims), per-comp k scale, round-nearest,
            // relu applied to the dequantized per-head dot.
            if (int8_disp) {
                std::vector<int> mneed(nt);
                #pragma omp parallel for schedule(dynamic)
                for (int t = 0; t < nt; t++) {
                    const float * w_t = w.data() + (size_t) t * n_head;
                    const float * qb_t = qb.data() + (size_t) t * n_head * D_IDX;
                    // quantize q (per token) and k (per comp) to int8
                    float qamax = 0.0f;
                    for (int i = 0; i < n_head*D_IDX; i++) qamax = fmaxf(qamax, fabsf(qb_t[i]));
                    const float qinv = qamax > 0.0f ? 127.0f/qamax : 0.0f;
                    const float qsc  = qamax > 0.0f ? qamax/127.0f : 0.0f;
                    std::vector<int8_t> q8(n_head*D_IDX);
                    for (int i = 0; i < n_head*D_IDX; i++) q8[i] = (int8_t) lrintf(qb_t[i]*qinv);
                    std::vector<float> sc(n_lid);
                    for (int j = 0; j < n_lid; j++) {
                        const float * kj = kb.data() + (size_t) j * D_IDX;
                        float kamax = 0.0f;
                        for (int d = 0; d < D_IDX; d++) kamax = fmaxf(kamax, fabsf(kj[d]));
                        const float kinv = kamax > 0.0f ? 127.0f/kamax : 0.0f;
                        const float ksc  = kamax > 0.0f ? kamax/127.0f : 0.0f;
                        int8_t k8[D_IDX];
                        for (int d = 0; d < D_IDX; d++) k8[d] = (int8_t) lrintf(kj[d]*kinv);
                        float acc = 0.0f;
                        for (int h = 0; h < n_head; h++) {
                            int32_t dot = 0;
                            const int8_t * qh = q8.data() + (size_t) h * D_IDX;
                            for (int d = 0; d < D_IDX; d++) dot += (int32_t) qh[d] * k8[d];
                            acc += fmaxf((float) dot * qsc * ksc, 0.0f) * w_t[h];
                        }
                        // f16 scores-buffer store: pass-1 ranks over the
                        // rounded value (d128 paths since the lid-traffic PR)
                        sc[j] = f16_scores ? (float) (_Float16) acc : acc;
                    }
                    // full int8 ranking (desc, idx tie-break); find worst rank
                    // of any true (path-B) top-512 member
                    std::vector<float> sb(n_lid);
                    scores_row(qb_t, w_t, kb.data(), n_lid, n_head, sb.data());
                    std::vector<int> tb;
                    top_k_set(sb.data(), n_lid, kk, tb);
                    std::vector<int> rank(n_lid);
                    std::iota(rank.begin(), rank.end(), 0);
                    std::sort(rank.begin(), rank.end(), [&](int a, int b) {
                        return sc[a] > sc[b] || (sc[a] == sc[b] && a < b);
                    });
                    std::vector<int> pos(n_lid);
                    for (int r = 0; r < n_lid; r++) pos[rank[r]] = r;
                    int worst = 0;
                    for (int j : tb) worst = std::max(worst, pos[j]);
                    mneed[t] = std::max(0, worst + 1 - kk);
                }
                std::sort(mneed.begin(), mneed.end());
                printf("    int8-displacement: m p50=%d p99=%d p100=%d (over %d tokens)\n",
                       mneed[nt/2], mneed[(size_t)(0.99*nt)], mneed.back(), nt);
            }

            // --fp4-mma-displacement: pass-1 ranked by the fp4-mma lid score
            // kernel (step 3). BOTH operands live on the e2m1 grid: q = path-B
            // qb (the runtime kernel per-32 e2m1-quantizes the SAME rotated f32
            // q the QAT sim does, so qb already IS the runtime q-pack — do NOT
            // re-quantize, that would be an identity double-quant), k = path-B
            // kb (exactly the values the packed QAT container holds). Head dot
            // is accumulated in f32 in two 64-wide halves summed (h0+h1) to
            // approximate the m16n8k64 mma's per-k64-fragment accumulation;
            // relu, x weight, heads summed ascending; final score rounded
            // through f16 (the kernel stores half). Truth = path-B exact
            // ranking (scores_row on qb/kb, f32 serial — the same sb the int8
            // block ranks against), so m_need isolates accumulation-order +
            // f16-store effects only.
            if (fp4mma_disp) {
                std::vector<int> mneed(nt);
                #pragma omp parallel for schedule(dynamic)
                for (int t = 0; t < nt; t++) {
                    const float * w_t  = w.data()  + (size_t) t * n_head;
                    const float * qb_t = qb.data() + (size_t) t * n_head * D_IDX;
                    std::vector<float> sc(n_lid);
                    for (int j = 0; j < n_lid; j++) {
                        const float * kj = kb.data() + (size_t) j * D_IDX;
                        float acc = 0.0f;
                        for (int h = 0; h < n_head; h++) {
                            const float * qh = qb_t + (size_t) h * D_IDX;
                            float h0 = 0.0f, h1 = 0.0f; // two k64 mma fragments
                            for (int d = 0;  d < 64;  d++) h0 += qh[d] * kj[d];
                            for (int d = 64; d < 128; d++) h1 += qh[d] * kj[d];
                            acc += fmaxf(h0 + h1, 0.0f) * w_t[h];
                        }
                        sc[j] = (float) (_Float16) acc; // kernel stores f16
                    }
                    // truth = path-B exact serial ranking (same sb as int8 blk)
                    std::vector<float> sb(n_lid);
                    scores_row(qb_t, w_t, kb.data(), n_lid, n_head, sb.data());
                    std::vector<int> tb;
                    top_k_set(sb.data(), n_lid, kk, tb);
                    std::vector<int> rank(n_lid);
                    std::iota(rank.begin(), rank.end(), 0);
                    std::sort(rank.begin(), rank.end(), [&](int a, int b) {
                        return sc[a] > sc[b] || (sc[a] == sc[b] && a < b);
                    });
                    std::vector<int> pos(n_lid);
                    for (int r = 0; r < n_lid; r++) pos[rank[r]] = r;
                    int worst = 0;
                    for (int j : tb) worst = std::max(worst, pos[j]);
                    mneed[t] = std::max(0, worst + 1 - kk);
                }
                std::sort(mneed.begin(), mneed.end());
                printf("    fp4mma-displacement: m p50=%d p99=%d p100=%d (over %d tokens)\n",
                       mneed[nt/2], mneed[(size_t)(0.99*nt)], mneed.back(), nt);
            }

            // Q-tile union stats (gathered sparse FA design input): for a tile
            // of W consecutive queries, the gathered-KV length is the union of
            // their top-k selections. Reported as mean/max over tiles.
            {
                std::vector<std::vector<int>> sel(nt);
                #pragma omp parallel for schedule(dynamic)
                for (int t = 0; t < nt; t++) {
                    std::vector<float> s(n_lid);
                    scores_row(qa.data() + (size_t) t * n_head * D_IDX,
                               w.data() + (size_t) t * n_head, ka.data(), n_lid, n_head, s.data());
                    top_k_set(s.data(), n_lid, kk, sel[t]);
                }
                for (int W : {8, 16, 32, 64, 128, 256}) {
                    if (W > nt) continue;
                    double usum = 0; int umax = 0, ntiles = 0;
                    std::vector<char> seen(n_lid);
                    for (int t0 = 0; t0 + W <= nt; t0 += W, ntiles++) {
                        std::fill(seen.begin(), seen.end(), 0);
                        int u = 0;
                        for (int t = t0; t < t0 + W; t++)
                            for (int j : sel[t]) u += !seen[j] ? (seen[j] = 1) : 0;
                        usum += u; umax = std::max(umax, u);
                    }
                    printf("    union W=%-3d mean=%7.1f max=%d  (dense=%d, cut=%.1fx)\n",
                           W, usum / ntiles, umax, n_lid, n_lid / (usum / ntiles));
                }
            }
        }
    }
    return 0;
}
