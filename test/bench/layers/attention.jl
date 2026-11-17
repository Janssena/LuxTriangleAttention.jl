# Worker for `test/attention.bench.jl`: `Attention` with a pair bias (`bias_layout=:kq`, self-attention on `[C, N, B]`) against
# AlphaFold3's PyTorch `Attention`. The logits here are `[H, N, N]` not `[H, N, N, N]`, so every method fits at every N.
# Run by the driver, one method per process: see `common.jl`.

include(joinpath(@__DIR__, "common.jl"))

const CHN_IN = parse(Int, get(ENV, "LTA_BENCH_C", "128"))
const HEAD_DIM = parse(Int, get(ENV, "LTA_BENCH_D", "32"))
const NUM_HEADS = parse(Int, get(ENV, "LTA_BENCH_H", "4"))
const BATCH = 1

function make_problem(variant)
    layer(kernel) = Attention(CHN_IN, HEAD_DIM, NUM_HEADS; use_bias=false, bias_layout=:kq, use_kernel=static(kernel))
    ps, st = Lux.setup(Random.Xoshiro(42), layer(false))
    py_layer() = begin
        py = getproperty(pyimport("__main__"), :AF3Attention)(CHN_IN, CHN_IN, CHN_IN, HEAD_DIM, NUM_HEADS)
        sync_attention!(py, ps)
        py
    end
    # PyTorch adds its biases to the scores `[B, H, Nq, Nk]`: the pair bias `bias[h, q, k, b]` and, for a mask on the keys, 1e9 * (mask - 1)
    function py_inputs((x, bias, mask), N)
        x_py = to_py(x; swap_batch_dim=true)
        biases = Any[to_py(permutedims(bias, (4, 1, 2, 3)))]
        if mask !== nothing
            additive = reshape(T(1e9) .* (T.(permutedims(mask, (2, 1))) .- one(T)), BATCH, 1, 1, N)
            pushfirst!(biases, to_py(additive))
        end
        return ((x_py, x_py), (; biases))
    end
    return (; layer, ps, st, py_layer, py_inputs)
end

function make_inputs(case, N)
    rng = Random.Xoshiro(1000 + N)
    x = randn(rng, T, CHN_IN, N, BATCH)
    bias = randn(rng, T, NUM_HEADS, N, N, BATCH)
    mask = case == "no mask" ? nothing : rand(rng, N, BATCH) .< 0.8
    return (x, bias, mask)
end

worker_main(; variants=LAYER_SPECS["Attention"].variants, cases=LAYER_SPECS["Attention"].cases,
    make_problem, make_inputs, est_bytes=N -> 0, kernel_dim=HEAD_DIM)
