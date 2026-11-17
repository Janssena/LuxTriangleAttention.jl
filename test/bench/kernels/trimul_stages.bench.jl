# Stage breakdown of `TriMul`'s `:default` forward on the machine's GPU: the forward of `src/kernels/trimul/op.jl`, with a
# device synchronisation around every stage (kernel launches and GEMMs), medians over repetitions. The sync makes each stage
# visible but adds round-trip time, so the sum is above an unsynchronised forward; it is for comparing stages, not totals.
#
#     julia --project=test test/bench/kernels/trimul_stages.bench.jl [N=512] [C=128] [H=128]
#
# Keep this body in step with `KernelDispatch.forward(::TriMul, ::TriMulStaged, ...)`.

using LuxTriangleAttention, KernelDispatch, Random, Printf, Statistics, LinearAlgebra
import LuxTriangleAttention: Lux
using Metal
const LTA = LuxTriangleAttention
const KD = KernelDispatch

const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
const C = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
const H = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 128
const B = 1

sync() = Metal.synchronize()

function stages_once!(times, m, x, mask, ps, st; is_outgoing=true)
    op, k = LTA.TriMul(), LTA.TriMulStaged()
    core = m.core
    T = N * N * B
    be = LTA.backend_of(x)
    p = LTA._trimul_point(be, C, H)
    w = LTA._trimul_params(m, ps, st)
    key = LTA._trimul_key(Float32, Float32, C, H, p, w.hb, mask !== nothing)
    nb = min(p.nb, T)
    ws(dims...) = similar(x, Float32, dims...)
    ph = ws(1)
    x2 = reshape(x, C, T)
    out = ws(C, N, N, B)
    xn, P = ws(C, nb), ws(nb * 4H)
    a, b = ws(T, H), ws(T, H)
    io = KD.InOutArray
    tm(label, f) = (sync(); t = time_ns(); r = f(); sync(); times[label] = get(times, label, 0.0) + (time_ns() - t) / 1e6; r)
    launch(stage, buf, s; extent) = KD.call_binary(op, k, stage, LTA.trimul_args(Val(stage), buf, s)...; key, extent)
    s0 = (; T=Int32(T), C=Int32(C), mmod=Int32(mask === nothing ? 1 : length(mask)), εi=w.εi, εo=w.εo)
    abuf = (; xn=io(xn), x=x2, w.γi, w.βi, a=io(a), b=io(b), bias=something(w.bab, ph),
        mask=mask === nothing ? similar(x, Bool, 1) : reshape(mask, :))
    for t0 in 0:nb:(T - 1)
        n = min(nb, T - t0)
        s = merge(s0, (; t0=Int32(t0), n=Int32(n)))
        Pb = LTA._block(P, n, 4H)
        tm("K1 ln_in", () -> launch(:ln_in, abuf, s; extent=n))
        tm("K1 GEMM C→4H", () -> LinearAlgebra.mul!(Pb, transpose(view(xn, :, 1:n)), transpose(w.Wab)))
        tm("K1 glu", () -> launch(:glu, merge(abuf, (; P=Pb)), s; extent=n * H))
    end
    LTA.KernelDispatch.release!(be, P)
    pa, pb = reshape(a, N, N, B * H), reshape(b, N, N, B * H)
    X = tm("K2 contraction", () -> LTA._contract(core.is_outgoing, pa, pb))
    foreach(z -> LTA.KernelDispatch.release!(be, z), (pa, pb, a, b))
    Xt = reshape(X, T, H)
    y, V, G = ws(H, nb), ws(C, nb), ws(C, nb)
    obuf = (; xn=io(xn), x=x2, w.γi, w.βi, y=io(y), X=Xt, w.γo, w.βo, out=io(reshape(out, C, T)), V, G,
        bo=something(w.bo, ph), bg=something(w.bg, ph))
    for t0 in 0:nb:(T - 1)
        n = min(nb, T - t0)
        s = merge(s0, (; t0=Int32(t0), n=Int32(n)))
        tm("K3 ln_out", () -> launch(:ln_out, obuf, s; extent=n))
        tm("K3 GEMM H→C", () -> LinearAlgebra.mul!(view(V, :, 1:n), w.Wo, view(y, :, 1:n)))
        tm("K3 ln_in (gate)", () -> launch(:ln_in, obuf, s; extent=n))
        tm("K3 GEMM gate", () -> LinearAlgebra.mul!(view(G, :, 1:n), w.Wg, view(xn, :, 1:n)))
        tm("K3 gate", () -> launch(:gate, obuf, s; extent=n * C))
    end
    foreach(z -> LTA.KernelDispatch.release!(be, z), (xn, y, V, G, Xt, X))
    return out
end

function main()
    m = TriangleMultiplication(C, H; use_kernel=true)
    ps, st = Lux.setup(Random.Xoshiro(1), m)
    ps = Lux.fmap(a -> a isa Array{Float32} ? a .+ 0.1f0 .* randn(Random.Xoshiro(length(a)), Float32, size(a)) : a, ps)
    dev(t) = Lux.fmap(a -> a isa Array ? MtlArray(a) : a, t)
    rng = Random.Xoshiro(2)
    x = MtlArray(randn(rng, Float32, C, N, N, B))
    mask = MtlArray(rand(rng, N, N, B) .< 0.8)
    ps, st = dev(ps), dev(st)
    y, _ = m(x, mask, ps, st)                      # compiles the variants
    ref = Array(first(m(x, mask, ps, st)))
    all = Dict{String,Vector{Float64}}()
    order = String[]
    reps = 15
    for r in 0:reps
        t = Dict{String,Float64}()
        out = stages_once!(t, m, x, mask, ps, st)
        r == 0 && @assert maximum(abs.(Array(out) .- ref)) <= 1e-5 * maximum(abs.(ref)) "staged copy disagrees with the forward"
        r == 0 && continue
        for (k, v) in t
            haskey(all, k) || (all[k] = Float64[]; push!(order, k))
            push!(all[k], v)
        end
    end
    println("TriMul :default stages, N=$N C=$C H=$H B=$B, ms (median of $reps, synced per stage)")
    tot = 0.0
    for k in sort(order)
        v = median(all[k]); tot += v
        @printf("  %-18s %7.2f\n", k, v)
    end
    @printf("  %-18s %7.2f\n", "total (synced)", tot)
    # Unsynchronised whole forward for reference.
    f() = (m(x, mask, ps, st); sync())
    f()
    ts = [(t = time_ns(); f(); (time_ns() - t) / 1e6) for _ in 1:reps]
    @printf("  %-18s %7.2f\n", "forward (median)", median(ts))
end

main()
