# Benchmark of `TriangleAttention` over N: eager Julia on the GPU (with and without the fused kernel) against AlphaFold3's PyTorch
# layer on the same GPU, and Reactant on the CPU and on the GPU with the kernel. Every output is checked against PyTorch on the
# CPU (`relerr`), including for masks that leave whole rows dead. Results: test/results.layers.csv.
#
#   julia --project=test test/triangle_attention.bench.jl
#
# The driver and its environment variables (N, methods, GPU, ...) are described in test/bench_utils.jl.
include("bench_utils.jl")
bench_driver("TriangleAttention")
