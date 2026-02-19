#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
import os, sys
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

import torch
from task import input_t, output_t

P = lambda *a: print(*a, file=sys.stderr)

HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <math.h>

// Naive: unpack FP4 nibbles, apply E8M0 scales, scalar MAC.
// One thread per output element. Correct but very slow.
#define BLOCK_DIM 16

__device__ float decode_fp4_e2m1(unsigned char nib) {
    // E2M1: sign(1) | exp(2) | mantissa(1)
    int sign = (nib >> 3) & 1;
    int exp  = (nib >> 1) & 3;
    int mant = nib & 1;
    if (exp == 0 && mant == 0) return 0.0f;
    float val = (1.0f + mant * 0.5f) * exp2f((float)(exp - 1));
    return sign ? -val : val;
}

__global__ void mxfp4_gemm_naive(
    const unsigned char* __restrict__ A_q,
    const unsigned char* __restrict__ B_q,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    __hip_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    int m = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int n = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (m >= M || n >= N) return;

    int K_packed = K >> 1;
    int K_sg = K >> 5;
    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        unsigned char byte_a = A_q[m * K_packed + (k >> 1)];
        unsigned char byte_b = B_q[n * K_packed + (k >> 1)];
        unsigned char nib_a = (k & 1) ? (byte_a >> 4) : (byte_a & 0xF);
        unsigned char nib_b = (k & 1) ? (byte_b >> 4) : (byte_b & 0xF);

        float sa = exp2f((float)A_scale[m * K_sg + (k >> 5)] - 127.0f);
        float sb = exp2f((float)B_scale[n * K_sg + (k >> 5)] - 127.0f);

        acc += decode_fp4_e2m1(nib_a) * sa * decode_fp4_e2m1(nib_b) * sb;
    }

    C[m * N + n] = __float2bfloat16(acc);
}

#include <torch/extension.h>

torch::Tensor mxfp4_gemm(
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int M, int N, int K
) {
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A_q.device()));
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((M + BLOCK_DIM-1)/BLOCK_DIM, (N + BLOCK_DIM-1)/BLOCK_DIM);
    mxfp4_gemm_naive<<<grid, block>>>(
        A_q.data_ptr<uint8_t>(), B_q.data_ptr<uint8_t>(),
        A_scale.data_ptr<uint8_t>(), B_scale.data_ptr<uint8_t>(),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()), M, N, K);
    return C;
}
"""

HIP_DECL = """
#include <torch/extension.h>
torch::Tensor mxfp4_gemm(
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int M, int N, int K);
"""

_hip_mod = None
try:
    from torch.utils.cpp_extension import load_inline
    _hip_mod = load_inline(
        name="mxfp4_gemm_naive",
        cpp_sources=HIP_DECL,
        cuda_sources=HIP_SOURCE,
        functions=["mxfp4_gemm"],
        extra_cuda_cflags=["-O2", "-std=c++17", "--offload-arch=gfx950"],
        verbose=False,
    )
    P("[naive] compiled ok")
except Exception as e:
    P(f"[naive] compilation failed: {e}")

_aiter_quant = None

def _get_quant():
    global _aiter_quant
    if _aiter_quant is None:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        _aiter_quant = dynamic_mxfp4_quant
    return _aiter_quant

def _e8m0_unshuffle(s, m, n):
    sm, sn = s.shape
    t = s.view(torch.uint8).view(sm//32, sn//8, 4, 16, 2, 2)
    return t.permute(0,5,3,1,4,2).contiguous().view(sm,sn)[:m,:n].contiguous()


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape
    n = B_q.shape[0]

    b_scale_raw = _e8m0_unshuffle(B_scale_sh, n, k // 32)
    quant = _get_quant()
    fp4, sc = quant(A)

    if _hip_mod is not None:
        try:
            return _hip_mod.mxfp4_gemm(
                fp4.contiguous(), B_q.view(torch.uint8).contiguous(),
                sc.contiguous(), b_scale_raw.view(torch.uint8).contiguous(),
                m, n, k)
        except Exception as e:
            P(f"[naive] error: {e}")

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    return gemm_afp4wfp4(fp4, B_q.view(torch.uint8), sc, b_scale_raw, dtype=torch.bfloat16)
