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

#define TILE_M 32
#define TILE_N 32
#define TILE_K 64
#define TILE_K_BYTES 32
#define WARP_SIZE 64
#define SCALE_GROUP 32
#define BLOCK_M 64
#define BLOCK_N 64
#define NUM_WARPS 4
#define BLOCK_THREADS (NUM_WARPS * WARP_SIZE)

// Added LDS tiling for A and B. Single buffer, __syncthreads between
// load and compute. Each warp computes a 32x32 output tile.
__global__ __launch_bounds__(BLOCK_THREADS)
void mxfp4_gemm_lds(
    const unsigned char* __restrict__ A_q,
    const unsigned char* __restrict__ B_q,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    __hip_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    int m_start = blockIdx.x * BLOCK_M;
    int n_start = blockIdx.y * BLOCK_N;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_m = warp_id >> 1;
    int warp_n = warp_id & 1;
    int tile_m = m_start + warp_m * TILE_M;
    int tile_n = n_start + warp_n * TILE_N;

    fp32x16_t acc = {};
    int K_packed = K >> 1;
    int K_sg = K >> 5;

    __shared__ unsigned char smem_a[BLOCK_M * TILE_K_BYTES];
    __shared__ unsigned char smem_b[BLOCK_N * TILE_K_BYTES];

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        int k_byte = k_tile >> 1;

        // Cooperatively fill LDS
        int elems = BLOCK_M * TILE_K_BYTES;
        for (int off = tid; off < elems; off += BLOCK_THREADS) {
            int row = off / TILE_K_BYTES;
            int col = off % TILE_K_BYTES;
            int grow = m_start + row;
            smem_a[off] = (grow < M && k_byte + col < K_packed)
                ? A_q[grow * K_packed + k_byte + col] : 0;
        }
        for (int off = tid; off < elems; off += BLOCK_THREADS) {
            int row = off / TILE_K_BYTES;
            int col = off % TILE_K_BYTES;
            int grow = n_start + row;
            smem_b[off] = (grow < N && k_byte + col < K_packed)
                ? B_q[grow * K_packed + k_byte + col] : 0;
        }
        __syncthreads();

        fp4x64_reg_t a_reg = {};
        fp4x64_reg_t b_reg = {};

        int a_base = warp_m * TILE_M * TILE_K_BYTES;
        int b_base = warp_n * TILE_N * TILE_K_BYTES;

        for (int i = 0; i < TILE_K_BYTES; i++) {
            a_reg[i] = smem_a[a_base + (lane_id % TILE_M) * TILE_K_BYTES + i];
            b_reg[i] = smem_b[b_base + (lane_id % TILE_N) * TILE_K_BYTES + i];
        }

        int sk = k_tile / SCALE_GROUP;
        int a_row = tile_m + (lane_id % TILE_M);
        int b_row = tile_n + (lane_id % TILE_N);
        unsigned char sa = (a_row < M && sk < K_sg) ? A_scale[a_row * K_sg + sk] : 127;
        unsigned char sb = (b_row < N && sk < K_sg) ? B_scale[b_row * K_sg + sk] : 127;

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);

        __syncthreads();
    }

    int out_col = tile_n + (lane_id % 32);
    if (out_col < N) {
        for (int i = 0; i < 16; i++) {
            int out_row = tile_m + (lane_id / 32) * 4 + (i % 4) + (i / 4) * 8;
            if (out_row < M)
                C[out_row * N + out_col] = __float2bfloat16(acc[i]);
        }
    }
}

#include <torch/extension.h>

torch::Tensor mxfp4_gemm(
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int M, int N, int K
) {
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A_q.device()));
    dim3 grid((M + BLOCK_M-1)/BLOCK_M, (N + BLOCK_N-1)/BLOCK_N);
    mxfp4_gemm_lds<<<grid, BLOCK_THREADS>>>(
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
        name="mxfp4_gemm_lds",
        cpp_sources=HIP_DECL,
        cuda_sources=HIP_SOURCE,
        functions=["mxfp4_gemm"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--offload-arch=gfx950"],
        verbose=False,
    )
    P("[lds] compiled ok")
except Exception as e:
    P(f"[lds] compilation failed: {e}")

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
            P(f"[lds] error: {e}")

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    return gemm_afp4wfp4(fp4, B_q.view(torch.uint8), sc, b_scale_raw, dtype=torch.bfloat16)
