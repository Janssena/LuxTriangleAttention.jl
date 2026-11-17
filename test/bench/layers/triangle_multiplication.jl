# Worker for `test/triangle_multiplication.bench.jl`: `TriangleMultiplication` against AlphaFold3's PyTorch `TriangleMultiplication`,
# at AlphaFold3's sizes (pair width 128, hidden 128), in the layer's default parameter layout (`fused`, `fused_glu`).
# Run by the driver, one method per process: see `common.jl`. `julia_gpu_kernel` is `use_kernel=true` (the `TriMul` op's
# `:default` kernel); it runs eagerly only, so there is no `xla_*_kernel` method.

include(joinpath(@__DIR__, "common.jl"))

const CHN_IN = parse(Int, get(ENV, "LTA_BENCH_C", "128"))
const CHN_HIDDEN = parse(Int, get(ENV, "LTA_BENCH_HIDDEN", "128"))
const BATCH = 1

function make_problem(variant)
    is_outgoing = occursin("Outgoing", variant)
    layer(kernel) = TriangleMultiplication(CHN_IN, CHN_HIDDEN; is_outgoing, use_bias=true, use_kernel=static(kernel))
    ps, st = Lux.setup(Random.Xoshiro(42), layer(false))
    py_layer() = begin
        cls = is_outgoing ? :AF3TriangleMultiplicationOutgoing : :AF3TriangleMultiplicationIncoming
        py = getproperty(pyimport("__main__"), cls)(CHN_IN, CHN_HIDDEN)
        sync_trimul!(py, ps; norm_in=:layer_norm_in, norm_out=:layer_norm_out,
            a=(linear=:linear_a_p, gate=:linear_a_g), b=(linear=:linear_b_p, gate=:linear_b_g),
            out=(linear=:linear_z, gate=:linear_g))
        py
    end
    # the module takes a mask always, so no mask is a mask of ones
    py_mask(mask, N) = mask === nothing ? to_py(ones(T, N, N, BATCH); swap_batch_dim=true) :
                       to_py(permutedims(mask, (3, 1, 2))).to(py_dtype(T))
    py_inputs((x, mask), N) = ((to_py(x; swap_batch_dim=true), py_mask(mask, N)), (;))
    return (; layer, ps, st, py_layer, py_inputs)
end

function make_inputs(case, N)
    rng = Random.Xoshiro(1000 + N)
    x = randn(rng, T, CHN_IN, N, N, BATCH)
    case == "no mask" && return (x, nothing)
    return (x, rand(rng, N, N, BATCH) .< 0.8)
end

worker_main(; variants=LAYER_SPECS["TriangleMultiplication"].variants, cases=LAYER_SPECS["TriangleMultiplication"].cases,
    make_problem, make_inputs, est_bytes=N -> 0, kernel_dim=0)
