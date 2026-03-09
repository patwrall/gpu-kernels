# gpu-kernels

## mxfp4-gemm — AMD MI355X

MXFP4 GEMM using native MFMA on gfx950.

Uses `mfma_scale_f32_32x32x64_f8f6f4` to consume FP4 directly into FP32 accumulators. Double-buffered LDS with 128-bit vectorized loads. Falls back to Triton `gemm_afp4wfp4` if the HIP kernel fails to compile.

Benchmark shape: M=16, N=7168, K=7168.
