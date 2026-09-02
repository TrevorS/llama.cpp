#include "hc-mix.cuh"

// The hyper-connection mix in two launches instead of eight (norm+gamma, down mm, scale,
// silu, up mm, sigmoid, gate-mix, inject mm). Both kernels recompute the grouped RMSNorm
// from x with one warp in a fixed order, so the two see identical xn bits and nothing
// depends on the token count: every reduction has the same shape at width 1 and width 8.
//
//   down: one block per (column group, token); a column is a dot product over hc*n_embd,
//         read coalesced along e. Columns past lr are the injection weights (F32).
//   up:   one thread per (e, token); the four stream rows of w_up (lr each) are dotted
//         with lo, gated by a sigmoid, and collapsed into the mixed output.
//
// Weights are dequantized element-wise; Q8_0 and Q6_K follow the CPU dequantize_row_* so
// the reference and the kernel round the same way.

template <typename T> struct hc_w;

template <> struct hc_w<float> {
    static __device__ __forceinline__ float get(const void * w, int64_t i) { return ((const float *) w)[i]; }
};
template <> struct hc_w<half> {
    static __device__ __forceinline__ float get(const void * w, int64_t i) { return __half2float(((const half *) w)[i]); }
};
template <> struct hc_w<block_q8_0> {
    static __device__ __forceinline__ float get(const void * w, int64_t i) {
        const block_q8_0 * b = (const block_q8_0 *) w + i/QK8_0;
        return __half2float(b->d) * (float) b->qs[i % QK8_0];
    }
};
template <> struct hc_w<block_q6_K> {
    static __device__ __forceinline__ float get(const void * w, int64_t i) {
        const block_q6_K * b = (const block_q6_K *) w + i/QK_K;
        const int r  = i % QK_K;
        const int ip = r / 128;
        const int rr = r % 128;
        const int q  = rr / 32;
        const int il = rr % 32;
        const uint8_t * ql = b->ql + 64*ip;
        const uint8_t   qh = b->qh[32*ip + il];
        const int8_t  * sc = b->scales + 8*ip + il/16;
        const float d = __half2float(b->d);
        int8_t v;
        switch (q) {
            case 0:  v = (int8_t)((ql[il]      & 0xF) | (((qh >> 0) & 3) << 4)); break;
            case 1:  v = (int8_t)((ql[32 + il] & 0xF) | (((qh >> 2) & 3) << 4)); break;
            case 2:  v = (int8_t)((ql[il]      >>  4) | (((qh >> 4) & 3) << 4)); break;
            default: v = (int8_t)((ql[32 + il] >>  4) | (((qh >> 6) & 3) << 4)); break;
        }
        return d * sc[2*q] * (v - 32);
    }
};

// per-stream inverse rms of one token, computed by warp 0 in a block-size independent order
static __device__ __forceinline__ void hc_mix_rrms(const float * __restrict__ x, const int n_embd, const int hc, const float eps, float * rrms) {
    if (threadIdx.x < WARP_SIZE) {
        for (int c = 0; c < hc; ++c) {
            const float * xs = x + c*n_embd;
            float s = 0.0f;
            for (int e = threadIdx.x; e < n_embd; e += WARP_SIZE) {
                s += xs[e]*xs[e];
            }
            s = warp_reduce_sum(s);
            if (threadIdx.x == 0) {
                rrms[c] = rsqrtf(s / (float) n_embd + eps);
            }
        }
    }
    __syncthreads();
}

#define HC_MIX_DOWN_COLS  4
#define HC_MIX_DOWN_BLOCK 256

template <typename T>
static __global__ void k_hc_mix_down(
        const float * __restrict__ x,
        const float * __restrict__ gamma,
        const void  * __restrict__ w_down,
        const float * __restrict__ w_inject,
        float       * __restrict__ dst,
        const int hc_dim, const int n_embd, const int hc, const int lr, const int n_col,
        const float eps, const float scale) {

    const int t  = blockIdx.y;
    const int j0 = blockIdx.x*HC_MIX_DOWN_COLS;

    __shared__ float rrms[GGML_HC_MIX_MAX_HC];
    __shared__ float red[HC_MIX_DOWN_COLS][HC_MIX_DOWN_BLOCK/WARP_SIZE];

    const float * xt = x + (long long) t*hc_dim;

    hc_mix_rrms(xt, n_embd, hc, eps, rrms);

    float acc[HC_MIX_DOWN_COLS] = { 0.0f };

    for (int e = threadIdx.x; e < hc_dim; e += HC_MIX_DOWN_BLOCK) {
        const float xn = xt[e]*rrms[e / n_embd]*gamma[e];
#pragma unroll
        for (int k = 0; k < HC_MIX_DOWN_COLS; ++k) {
            const int j = j0 + k;
            if (j < n_col) {
                const float w = j < lr ? hc_w<T>::get(w_down, (int64_t) j*hc_dim + e) : w_inject[(int64_t) (j - lr)*hc_dim + e];
                acc[k] += xn * w;
            }
        }
    }

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

#pragma unroll
    for (int k = 0; k < HC_MIX_DOWN_COLS; ++k) {
        const float s = warp_reduce_sum(acc[k]);
        if (lane == 0) {
            red[k][warp] = s;
        }
    }
    __syncthreads();

    if (threadIdx.x < HC_MIX_DOWN_COLS) {
        const int k = threadIdx.x;
        const int j = j0 + k;
        if (j < n_col) {
            float s = 0.0f;
            for (int w = 0; w < HC_MIX_DOWN_BLOCK/WARP_SIZE; ++w) {
                s += red[k][w];
            }
            if (j < lr) {
                const float v = s*scale;
                s = v / (1.0f + expf(-v));
            }
            dst[(long long) t*n_col + j] = s;
        }
    }
}

#define HC_MIX_UP_BLOCK 64

template <typename T>
static __global__ void k_hc_mix_up(
        const float * __restrict__ x,
        const float * __restrict__ gamma,
        const void  * __restrict__ w_up,
        const float * __restrict__ lo,
        float       * __restrict__ dst,
        const int hc_dim, const int n_embd, const int hc, const int lr, const int s_lo,
        const float eps, const float scale) {

    const int t = blockIdx.y;
    const int e = blockIdx.x*HC_MIX_UP_BLOCK + threadIdx.x;

    extern __shared__ float lo_s[];
    __shared__ float rrms[GGML_HC_MIX_MAX_HC];

    const float * xt = x + (long long) t*hc_dim;

    for (int l = threadIdx.x; l < lr; l += HC_MIX_UP_BLOCK) {
        lo_s[l] = lo[(long long) t*s_lo + l];
    }

    hc_mix_rrms(xt, n_embd, hc, eps, rrms);

    if (e >= n_embd) {
        return;
    }

    float acc = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const int64_t i = (int64_t) c*n_embd + e;

        float g = 0.0f;
        for (int l = 0; l < lr; ++l) {
            g += lo_s[l] * hc_w<T>::get(w_up, i*lr + l);
        }
        const float gate = 1.0f / (1.0f + expf(-g));

        acc += xt[i]*rrms[c]*gamma[i] * gate;
    }
    dst[(long long) t*n_embd + e] = acc*scale;
}

void ggml_cuda_op_hc_mix_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * gamma    = dst->src[1];
    const ggml_tensor * w_down   = dst->src[2];
    const ggml_tensor * w_inject = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && gamma->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(gamma) && ggml_is_contiguous(w_down) && ggml_is_contiguous(dst));

    const int   hc     = ggml_get_op_params_i32(dst, 0);
    const float eps    = ggml_get_op_params_f32(dst, 1);
    const float scale  = ggml_get_op_params_f32(dst, 2);
    const int   hc_dim = x->ne[0];
    const int   nt     = x->ne[1];
    const int   lr     = w_down->ne[1];
    const int   n_col  = dst->ne[0];

    GGML_ASSERT(hc <= GGML_HC_MIX_MAX_HC);

    dim3 grid((n_col + HC_MIX_DOWN_COLS - 1)/HC_MIX_DOWN_COLS, nt, 1);

    const float * inj = w_inject ? (const float *) w_inject->data : nullptr;

#define HC_MIX_DOWN_LAUNCH(T) \
    k_hc_mix_down<T><<<grid, HC_MIX_DOWN_BLOCK, 0, ctx.stream()>>>( \
        (const float *) x->data, (const float *) gamma->data, w_down->data, inj, (float *) dst->data, \
        hc_dim, hc_dim/hc, hc, lr, n_col, eps, scale)

    switch (w_down->type) {
        case GGML_TYPE_F32:  HC_MIX_DOWN_LAUNCH(float);      break;
        case GGML_TYPE_F16:  HC_MIX_DOWN_LAUNCH(half);       break;
        case GGML_TYPE_Q8_0: HC_MIX_DOWN_LAUNCH(block_q8_0); break;
        case GGML_TYPE_Q6_K: HC_MIX_DOWN_LAUNCH(block_q6_K); break;
        default: GGML_ABORT("hc_mix_down: unsupported weight type %s", ggml_type_name(w_down->type));
    }
#undef HC_MIX_DOWN_LAUNCH
}

void ggml_cuda_op_hc_mix_up(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * gamma = dst->src[1];
    const ggml_tensor * w_up  = dst->src[2];
    const ggml_tensor * lo    = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && gamma->type == GGML_TYPE_F32 && lo->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(gamma) && ggml_is_contiguous(w_up) && ggml_is_contiguous(dst));
    GGML_ASSERT(lo->nb[0] == sizeof(float));

    const int   hc     = ggml_get_op_params_i32(dst, 0);
    const float eps    = ggml_get_op_params_f32(dst, 1);
    const float scale  = ggml_get_op_params_f32(dst, 2);
    const int   hc_dim = x->ne[0];
    const int   nt     = x->ne[1];
    const int   lr     = w_up->ne[0];
    const int   n_embd = hc_dim/hc;
    const int   s_lo   = lo->nb[1]/sizeof(float);

    GGML_ASSERT(hc <= GGML_HC_MIX_MAX_HC);

    dim3 grid((n_embd + HC_MIX_UP_BLOCK - 1)/HC_MIX_UP_BLOCK, nt, 1);
    const size_t smem = lr*sizeof(float);

#define HC_MIX_UP_LAUNCH(T) \
    k_hc_mix_up<T><<<grid, HC_MIX_UP_BLOCK, smem, ctx.stream()>>>( \
        (const float *) x->data, (const float *) gamma->data, w_up->data, (const float *) lo->data, (float *) dst->data, \
        hc_dim, n_embd, hc, lr, s_lo, eps, scale)

    switch (w_up->type) {
        case GGML_TYPE_F32:  HC_MIX_UP_LAUNCH(float);      break;
        case GGML_TYPE_F16:  HC_MIX_UP_LAUNCH(half);       break;
        case GGML_TYPE_Q8_0: HC_MIX_UP_LAUNCH(block_q8_0); break;
        case GGML_TYPE_Q6_K: HC_MIX_UP_LAUNCH(block_q6_K); break;
        default: GGML_ABORT("hc_mix_up: unsupported weight type %s", ggml_type_name(w_up->type));
    }
#undef HC_MIX_UP_LAUNCH
}
