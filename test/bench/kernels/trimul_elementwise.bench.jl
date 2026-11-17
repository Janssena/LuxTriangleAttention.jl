# Device time of `TriMul`'s four stage kernels on one block of `nb` tokens, from the device's own timestamps
# (`KernelDispatch.time_binary`: back-to-back dispatches, no host latency, so it is not disturbed by a busy CPU), with the
# effective memory bandwidth each reaches. For the elementwise-kernel work in docs/FEATURE_trimul_kernel.md §7.5.
#
#     julia --project=test test/bench/kernels/trimul_elementwise.bench.jl [C=128] [H=128] [n=16384]

using LuxTriangleAttention, KernelDispatch, Random, Printf, Statistics
import LuxTriangleAttention: Lux
using Metal
const LTA = LuxTriangleAttention
const KD = KernelDispatch

const C = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
const H = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
const n = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16384

function main()
    m = TriangleMultiplication(C, H; use_kernel=true, use_bias=true)
    ps, st = Lux.setup(Random.Xoshiro(1), m)
    dev(t) = Lux.fmap(a -> a isa Array ? MtlArray(a) : a, t)
    ps, st = dev(ps), dev(st)
    N = 40                                   # any size: compiles the variants
    x = MtlArray(randn(Float32, C, N, N, 1))
    mask = MtlArray(rand(N, N, 1) .< 0.8)
    m(x, mask, ps, st)
    op, k = LTA.TriMul(), LTA.TriMulStaged()
    be = LTA.backend_of(x)
    p = LTA._trimul_point(be, C, H)
    w = LTA._trimul_params(m, ps, st)
    key = LTA._trimul_key(Float32, Float32, C, H, p, w.hb, true)
    f(dims...) = MtlArray(randn(Float32, dims...))
    T = n
    xn, x2, y, X = f(C, n), f(C, T), f(H, n), f(T, H)
    P, a, b, V, G, out = f(n, 4H), f(T, H), f(T, H), f(C, n), f(C, n), f(C, T)
    mk = MtlArray(rand(T) .< 0.8)
    s = (; t0=Int32(0), n=Int32(n), T=Int32(T), C=Int32(C), mmod=Int32(T), εi=w.εi, εo=w.εo)
    bufs = (; xn, x=x2, w.γi, w.βi, y, X, w.γo, w.βo, a, b, P, bias=something(w.bab, f(1)), mask=mk, out, V, G,
        bo=something(w.bo, f(1)), bg=something(w.bg, f(1)))
    mb(nelem) = nelem * 4 / 2^20
    rows = (
        (:ln_in, n, n * C * 2),
        (:glu, n * H, n * 4H + 2 * n * H + n / 4),
        (:ln_out, n, n * H * 2),
        (:gate, n * C, 3 * n * C),
    )
    println("TriMul stage kernels, C=$C H=$H, one block of n=$n tokens, device time")
    for (stage, extent, floats) in rows
        args = LTA.trimul_args(Val(stage), bufs, s)
        t = minimum(KD.time_binary(op, k, stage, args...; key, extent, reps=20) for _ in 1:5)
        @printf("  %-8s %8.3f ms   %6.1f GB/s  (%.0f MB moved)\n", stage, t * 1e3, floats * 4 / t / 1e9, floats * 4 / 2^20)
    end
end

main()
