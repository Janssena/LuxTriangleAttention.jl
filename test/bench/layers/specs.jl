# What each layer benchmark sweeps: plain data, shared by the driver (`test/bench_utils.jl`, which loads no package) and the
# workers (`test/bench/layers/<layer>.jl`).
#
#   variants   the layer configurations (name = how the CSV and the table label them)
#   cases      the inputs: mask or no mask, and for attention a mask that leaves whole rows dead
#   methods    what is timed by default, `extras` what `LTA_BENCH_EXTRAS=1` adds
#
# `memory` is the one (variant, case) whose memory is measured (`LTA_BENCH_MEMORY=1`): a cell costs a process of its own.
#
# A method is `<framework>_<device>[_kernel]`: `julia_*` is the eager layer, `xla_*` the same layer compiled by Reactant,
# `torch_*` the hand-written AlphaFold3 PyTorch class (test/python/alphafold3.jl). `_kernel` is `use_kernel=static(true)`.

const BENCH_REF_METHOD = "torch_cpu"      # the ground truth every method's output is compared with, never timed unless asked

const LAYER_SPECS = Dict(
    "Attention" => (
        variants=["AlphaFold3 (pair bias)"],
        cases=["no mask", "mask"],
        memory=("AlphaFold3 (pair bias)", "mask"),
        methods=["julia_gpu", "julia_gpu_kernel", "torch_gpu", "xla_cpu", "xla_gpu_kernel"],
        extras=["julia_cpu", "torch_cpu", "xla_gpu"],
    ),
    "TriangleAttention" => (
        variants=["AlphaFold3 (Starting)", "AlphaFold3 (Ending)"],
        cases=["no mask", "mask", "dead rows"],
        memory=("AlphaFold3 (Starting)", "mask"),
        methods=["julia_gpu", "julia_gpu_kernel", "torch_gpu", "xla_cpu", "xla_gpu_kernel"],
        extras=["julia_cpu", "torch_cpu", "xla_gpu"],
    ),
    "TriangleMultiplication" => (
        variants=["AlphaFold3 (Outgoing)", "AlphaFold3 (Incoming)"],
        cases=["no mask", "mask"],
        memory=("AlphaFold3 (Outgoing)", "mask"),
        methods=["julia_gpu", "julia_gpu_kernel", "torch_gpu", "xla_cpu", "xla_gpu"],
        extras=["julia_cpu", "torch_cpu"],
    ),
)
