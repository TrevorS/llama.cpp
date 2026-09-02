#include "common.cuh"

// the fused hyper-connection mix: see ggml_hc_mix_down / ggml_hc_mix_up in ggml.h
#define GGML_HC_MIX_MAX_HC 8

void ggml_cuda_op_hc_mix_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hc_mix_up  (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
