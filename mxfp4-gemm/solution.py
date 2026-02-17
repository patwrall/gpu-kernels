import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
TORCH_LIBRARY(my_module, m) {
  m.def("gemm(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C) -> Tensor");
  m.impl("gemm", &gemm);
}
"""

ext = load_inline(
    "gemm_hip",
    cpp_sources="",
    cuda_sources=HIP_SRC,
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "--offload-arch=gfx1201",
        "-ffast-math",
        "-mwavefrontsize64",
    ],
    extra_ldflags=[],
)

gemm = torch.ops.my_module.gemm


def custom_kernel(data: input_t) -> output_t:
    a, b, _, _, sfa, sfb, c = data
    return gemm(a, b, sfa, sfb, c)
