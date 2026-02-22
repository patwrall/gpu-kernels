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

using fp4x2_t = unsigned char;
using fp4x64_reg_t = fp4x2_t __attribute__((ext_vector_type(32)));
using fp32x16_t = float __attribute__((ext_vector_type(16)));

#define WARP_SIZE 64
#define TILE_M 32
#define TILE_N 32
#define TILE_K 64

// First attempt at using mfma_scale_f32_32x32x64_f8f6f4.
// Register layout is probably wrong — output is incorrect but want to
// see if the instruction compiles and runs before fixing the mapping.
__global__ __launch_bounds__(WARP_SIZE)
void mxfp4_gemm_mfma_v1(
    const unsigned char* __restrict__ A_q,
    const unsigned char* __restrict__ B_q,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    __hip_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    int lane_id = threadIdx.x;
    int m_base = blockIdx.x * TILE_M;
    int n_base = blockIdx.y * TILE_N;

    fp32x16_t acc = {};
    int K_packed = K >> 1;
    int K_sg = K >> 5;

    for (int k = 0; k < K; k += TILE_K) {
        fp4x64_reg_t a_reg = {};
        fp4x64_reg_t b_reg = {};

        // Naive packing: lane i loads row i of the tile
        // This doesn't match the expected mfma lane layout but lets us test the instruction
        int a_row = m_base + (lane_id % TILE_M);
        if (a_row < M) {
            int off = a_row * K_packed + (k >> 1);
            for (int i = 0; i < 32; i++)
                a_reg[i] = (off + i < (a_row+1) * K_packed) ? A_q[off + i] : 0;
        }

        int b_row = n_base + (lane_id % TILE_N);
        if (b_row < N) {
            int off = b_row * K_packed + (k >> 1);
            for (int i = 0; i < 32; i++)
                b_reg[i] = (off + i < (b_row+1) * K_packed) ? B_q[off + i] : 0;
        }

        unsigned char sa = (a_row < M) ? A_scale[a_row * K_sg + k/32] : 127;
        unsigned char sb = (b_row < N) ? B_scale[b_row * K_sg + k/32] : 127;

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }

    // Output mapping is also wrong here — just dumping for debugging
    int out_row = m_base + (lane_id / 2);
    int out_col = n_base + (lane_id % 2) * 16;
    if (out_row < M) {
        for (int i = 0; i < 16 && out_col + i < N; i++)
            C[out_row * N + out_col + i] = __float2bfloat16(acc[i]);
    }
}

#include <torch/extension.h>

torch::Tensor mxfp4_gemm(
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int M, int N, int K
) {
    auto C = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A_q.device()));
    dim3 grid((M + TILE_M-1)/TILE_M, (N + TILE_N-1)/TILE_N);
    mxfp4_gemm_mfma_v1<<<grid, WARP_SIZE>>>(
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
        name="mxfp4_gemm_mfma_v1",
        cpp_sources=HIP_DECL,
        cuda_sources=HIP_SOURCE,
        functions=["mxfp4_gemm"],
        extra_cuda_cflags=["-O2", "-std=c++17", "--offload-arch=gfx950"],
        verbose=False,
    )
    P("[mfma_v1] compiled ok")
except Exception as e:
    P(f"[mfma_v1] compilation failed: {e}")

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
            P(f"[mfma_v1] error: {e}")

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    return gemm_afp4wfp4(fp4, B_q.view(torch.uint8), sc, b_scale_raw, dtype=torch.bfloat16)
