# Benchmark of `TriangleMultiplication` over N: eager Julia on the GPU against AlphaFold3's PyTorch layer on the same GPU, and
# Reactant on the CPU and on the GPU. Every output is checked against PyTorch on the CPU (`relerr`). Results:
# test/results.layers.csv.
#
#   julia --project=test test/triangle_multiplication.bench.jl
#
# The driver and its environment variables (N, methods, GPU, ...) are described in test/bench_utils.jl.
include("bench_utils.jl")
bench_driver("TriangleMultiplication")
