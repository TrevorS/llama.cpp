#include "dsv4_hc_fused.cuh"

// out[e,t] = sum_ih x[e,ih,t] * w[ih,t]
// x [n_embd,hc,nt] contiguous, w [hc,nt] contiguous, out [n_embd,nt].
// Accumulate ih = 0..hc-1 left-to-right (matches the scalar graph).
static __global__ void dsv4_hc_weighted_sum_kernel(
        float       * __restrict__ out,
        const float * __restrict__ x,
        const float * __restrict__ w,
        const int n_embd, const int hc, const int nt) {
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_w[]; // hc
    for (int i = threadIdx.x; i < hc; i += blockDim.x) {
        s_w[i] = w[(size_t) t * hc + i];
    }
    __syncthreads();

    if (e >= n_embd) {
        return;
    }

    float acc = __fmul_rn(x[((size_t) t * hc + 0) * n_embd + e], s_w[0]);
    for (int ih = 1; ih < hc; ++ih) {
        acc = __fadd_rn(acc, __fmul_rn(x[((size_t) t * hc + ih) * n_embd + e], s_w[ih]));
    }
    out[(size_t) t * n_embd + e] = acc;
}

// out[e,dst,t] = x[e,t]*post[dst,t] + sum_src res[e,src,t]*comb[dst,src,t]
// x [n_embd,nt], res [n_embd,hc,nt], post [hc,nt], comb [dst,src,nt] (ne0=dst),
// out [n_embd,hc,nt]. Seed with x*post then add src = 0..hc-1 (matches scalar).
static __global__ void dsv4_hc_post_kernel(
        float       * __restrict__ out,
        const float * __restrict__ x,
        const float * __restrict__ res,
        const float * __restrict__ post,
        const float * __restrict__ comb,
        const int n_embd, const int hc, const int nt) {
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s[];
    float * s_post = s;        // hc
    float * s_comb = s + hc;   // hc*hc, laid out [src*hc + dst] (comb ne0=dst)
    for (int i = threadIdx.x; i < hc + hc * hc; i += blockDim.x) {
        if (i < hc) {
            s_post[i] = post[(size_t) t * hc + i];
        } else {
            s_comb[i - hc] = comb[(size_t) t * hc * hc + (i - hc)];
        }
    }
    __syncthreads();

    if (e >= n_embd) {
        return;
    }

    const float xv = x[(size_t) t * n_embd + e];
    float rv[GGML_DSV4_HC_MAX];
    for (int src = 0; src < hc; ++src) {
        rv[src] = res[((size_t) t * hc + src) * n_embd + e];
    }

    for (int dst = 0; dst < hc; ++dst) {
        float acc = __fmul_rn(xv, s_post[dst]);
        for (int src = 0; src < hc; ++src) {
            acc = __fadd_rn(acc, __fmul_rn(rv[src], s_comb[src * hc + dst]));
        }
        out[((size_t) t * hc + dst) * n_embd + e] = acc;
    }
}

// One block per token; the [hc, hc] matrix lives in shared memory. Lane s
// (resp. d) does SERIAL sums over the other axis so the accumulation order
// matches the CPU reference exactly (hc <= 8, negligible work — the win is
// replacing ~85 kernel launches with one).
static __global__ void dsv4_hc_sinkhorn_kernel(
        float * __restrict__ out, const float * __restrict__ comb,
        int hc, int nt, int iters, float eps) {
    const int t = blockIdx.x;
    if (t >= nt) return;
    __shared__ float m[GGML_DSV4_HC_MAX * GGML_DSV4_HC_MAX];
    const int n = hc * hc;
    const int tid = threadIdx.x;
    if (tid < n) m[tid] = comb[(int64_t) t * n + tid];
    __syncthreads();
    // softmax along d for each s, then +eps
    if (tid < hc) {
        const int s = tid;
        float mx = -INFINITY;
        for (int d = 0; d < hc; ++d) mx = fmaxf(mx, m[d + s*hc]);
        float sum = 0.0f;
        for (int d = 0; d < hc; ++d) { const float e = expf(m[d + s*hc] - mx); m[d + s*hc] = e; sum += e; }
        for (int d = 0; d < hc; ++d) m[d + s*hc] = m[d + s*hc]/sum + eps;
    }
    __syncthreads();
    for (int i = 0; i < iters; ++i) {
        if (i > 0) { // norm_rows (skipped before the first norm_cols)
            if (tid < hc) {
                const int s = tid;
                float rs = 0.0f;
                for (int d = 0; d < hc; ++d) rs += m[d + s*hc];
                rs += eps;
                for (int d = 0; d < hc; ++d) m[d + s*hc] /= rs;
            }
            __syncthreads();
        }
        if (tid < hc) { // norm_cols
            const int d = tid;
            float cs = 0.0f;
            for (int s = 0; s < hc; ++s) cs += m[d + s*hc];
            cs += eps;
            for (int s = 0; s < hc; ++s) m[d + s*hc] /= cs;
        }
        __syncthreads();
    }
    if (tid < n) out[(int64_t) t * n + tid] = m[tid];
}

void ggml_cuda_op_dsv4_hc_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = ggml_get_op_params_i32(dst, 0);
    cudaStream_t stream = ctx.stream();
    const int block = 256;

    if (mode == GGML_DSV4_HC_MODE_SINKHORN) {
        const ggml_tensor * comb = dst->src[0]; // [hc, hc, nt]
        GGML_ASSERT(comb->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(comb) && ggml_is_contiguous(dst));
        const int hc = comb->ne[0];
        const int nt = comb->ne[2];
        GGML_ASSERT(hc <= GGML_DSV4_HC_MAX);
        const int iters = ggml_get_op_params_i32(dst, 1);
        const float eps = ggml_get_op_params_f32(dst, 2);
        const int n_iters_eff = iters > 0 ? iters : 1; // iters==0 still does one norm_cols
        dsv4_hc_sinkhorn_kernel<<<nt, 64, 0, stream>>>(
                (float *) dst->data, (const float *) comb->data, hc, nt, n_iters_eff, eps);
        return;
    }

    if (mode == GGML_DSV4_HC_MODE_WEIGHTED_SUM) {
        const ggml_tensor * x = dst->src[0]; // [n_embd, hc, nt]
        const ggml_tensor * w = dst->src[1]; // [hc, nt]

        GGML_ASSERT(x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(w) && ggml_is_contiguous(dst));

        const int n_embd = x->ne[0];
        const int hc     = x->ne[1];
        const int nt     = x->ne[2];
        GGML_ASSERT(hc <= GGML_DSV4_HC_MAX);

        const dim3 grid((n_embd + block - 1) / block, nt, 1);
        dsv4_hc_weighted_sum_kernel<<<grid, block, hc * sizeof(float), stream>>>(
                (float *) dst->data, (const float *) x->data, (const float *) w->data,
                n_embd, hc, nt);
    } else {
        const ggml_tensor * x    = dst->src[0]; // [n_embd, nt]
        const ggml_tensor * res  = dst->src[1]; // [n_embd, hc, nt]
        const ggml_tensor * post = dst->src[2]; // [hc, nt]
        const ggml_tensor * comb = dst->src[3]; // [dst, src, nt]

        GGML_ASSERT(x->type == GGML_TYPE_F32 && res->type == GGML_TYPE_F32);
        GGML_ASSERT(post->type == GGML_TYPE_F32 && comb->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(res));
        GGML_ASSERT(ggml_is_contiguous(post) && ggml_is_contiguous(comb) && ggml_is_contiguous(dst));

        const int n_embd = x->ne[0];
        const int nt     = x->ne[1];
        const int hc     = res->ne[1];
        GGML_ASSERT(hc <= GGML_DSV4_HC_MAX);

        const dim3 grid((n_embd + block - 1) / block, nt, 1);
        const size_t shmem = (hc + hc * hc) * sizeof(float);
        dsv4_hc_post_kernel<<<grid, block, shmem, stream>>>(
                (float *) dst->data, (const float *) x->data, (const float *) res->data,
                (const float *) post->data, (const float *) comb->data,
                n_embd, hc, nt);
    }
}
