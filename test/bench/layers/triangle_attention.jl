# Worker for `test/triangle_attention.bench.jl`: `TriangleAttention` against AlphaFold3's PyTorch `TriangleAttention`, at AlphaFold3's
# sizes (pair width 128, 4 heads of 32). Run by the driver, one method per process: see `common.jl`.

include(joinpath(@__DIR__, "common.jl"))

const CHN_IN = parse(Int, get(ENV, "LTA_BENCH_C", "128"))
const HEAD_DIM = parse(Int, get(ENV, "LTA_BENCH_D", "32"))
const NUM_HEADS = parse(Int, get(ENV, "LTA_BENCH_H", "4"))
const BATCH = 1
const USE_BIAS = (false, (layer_norm=true,))        # AlphaFold3's: a LayerNorm and a bias projection, nothing else

function make_problem(variant)
    is_starting = occursin("Starting", variant)
    layer(kernel) = TriangleAttention(CHN_IN, HEAD_DIM, NUM_HEADS; is_starting, use_bias=USE_BIAS, use_kernel=static(kernel))
    ps, st = Lux.setup(Random.Xoshiro(42), layer(false))
    py_layer() = begin
        py = getproperty(pyimport("__main__"), :AF3TriangleAttention)(CHN_IN, HEAD_DIM, NUM_HEADS; starting=is_starting, inf=1.0f9)
        sync_af3_triangle_attention!(py, ps)
        py
    end
    py_mask(mask) = mask === nothing ? nothing : to_py(permutedims(mask, (3, 1, 2))).to(py_dtype(T))   # PyTorch wants floats
    py_inputs((x, mask), N) = ((to_py(x; swap_batch_dim=true), py_mask(mask)), (;))
    return (; layer, ps, st, py_layer, py_inputs)
end

# "dead rows": a tenth of the rows have no key to attend to at all. They must come out uniform (finite), not NaN.
function make_inputs(case, N)
    rng = Random.Xoshiro(1000 + N)
    x = randn(rng, T, CHN_IN, N, N, BATCH)
    case == "no mask" && return (x, nothing)
    mask = rand(rng, N, N, BATCH) .< 0.8
    if case == "dead rows"
        mask[1:10:N, :, :] .= false
    end
    return (x, mask)
end

# the scores of the unfused path: [Nk, Nq, H, Nj, B] in Float32, a few times over (logits, softmax, a copy)
est_bytes(N) = 3 * 4 * NUM_HEADS * N^3 * BATCH

worker_main(; variants=LAYER_SPECS["TriangleAttention"].variants, cases=LAYER_SPECS["TriangleAttention"].cases,
    make_problem, make_inputs, est_bytes, kernel_dim=HEAD_DIM)
