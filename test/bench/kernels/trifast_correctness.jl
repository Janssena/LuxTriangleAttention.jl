# Correctness of both registry kernels — our flash kernel (`:default`) and trifast's metallibs
# (`:trifast`) — on IDENTICAL operands, through the same `SDPA()` / `sdpa_bwd` path
# the layers use, against a from-scratch reference. Forward (`o`, `lse`) and backward (`dq`, `dk`,
# `dv`, `dbias`).
#
# The problem deliberately has B = 2, H = 2 and a random key mask: batch and head are not adjacent
# in this package's `[hd, H, Nq, Nj, B]` layout, so a kernel addressing `b·H + h` with one stride
# passes at B = 1 and fails here; and a mask that is all-keep cannot catch a mis-addressed mask.
#
#     julia --project=test test/bench/kernels/trifast_correctness.jl

include(joinpath(@__DIR__, "registry_setup.jl"))   # BEFORE `using LuxTriangleAttention`
using LuxTriangleAttention, Metal, Random, Printf
include(joinpath(@__DIR__, "..", "..", "support", "sdpa_bwd.jl"))   # sdpa_bwd

const LTA = LuxTriangleAttention

"""
    reference(q, k, v, keep, bias, dO) -> (o, lse, dq, dk, dv, dbias)

Plain-loop attention and its analytic gradients on `[hd, H, n, n, B]` operands, per (head, outer
row, batch) slice: `L = s·KᵀQ + bias` (`[key, query]`), keys with `keep = false` dropped,
`P = softmax_keys(L)`, `O = V·P`; then `dV = dO·Pᵀ`, `dP = Vᵀ·dO`, `dL = P ⊙ (dP − Σ_keys P⊙dP)`,
`dQ = s·K·dL`, `dK = s·Q·dLᵀ`, and `dbias` sums `dL` over the outer rows it is shared across.
"""
function reference(q, k, v, keep, bias, dO)
    hd, H, n, nj, B = size(q)
    s = Float32(1 / sqrt(hd))
    o, dq, dk, dv = zero(q), zero(q), zero(k), zero(v)
    lse = zeros(Float32, 1, n, H, nj, B)
    db = zeros(Float32, size(bias))
    for b in 1:B, i in 1:nj, h in 1:H
        Q, K, V, G = (@view(x[:, h, :, i, b]) for x in (q, k, v, dO))
        L = s .* (K' * Q) .+ bias[:, :, h, b]
        L[.!keep[:, i, b], :] .= -Inf32
        m = maximum(L; dims=1)
        E = exp.(L .- m)
        P = E ./ sum(E; dims=1)
        o[:, h, :, i, b] .= V * P
        lse[1, :, h, i, b] .= vec(m .+ log.(sum(E; dims=1)))
        dP = V' * G
        dL = P .* (dP .- sum(P .* dP; dims=1))
        dv[:, h, :, i, b] .= G * P'
        dq[:, h, :, i, b] .= s .* (K * dL)
        dk[:, h, :, i, b] .= s .* (Q * dL')
        db[:, :, h, b] .+= dL
    end
    return o, lse, dq, dk, dv, db
end

relerr(a, b) = maximum(abs.(Array(a) .- b)) / maximum(abs.(b))

function check(name, q, k, v, keep, bias, dO, ref)
    kern = kernel_named(name)
    o, lse = KernelDispatch.forward(LTA.SDPA(), kern, q, k, v, keep, bias)
    dq, dk, dv, db = sdpa_bwd(q, k, v, keep, bias, o, lse, dO; kernel=kern)
    ok = true
    for (lbl, got, want) in zip(("o", "lse", "dq", "dk", "dv", "dbias"),
        (o, lse, dq, dk, dv, db), ref)
        e = relerr(got, want)
        ok &= e < 1.0f-3
        @printf "  %-9s %-6s rel err %.2e\n" name lbl e
    end
    return ok
end

function main()
    Metal.functional() || error("no functional Metal device")
    hd, H, n, B = 32, 2, 64, 2
    rng = Random.Xoshiro(0)
    q, k, v, dO = (randn(rng, Float32, hd, H, n, n, B) for _ in 1:4)
    bias = randn(rng, Float32, n, n, H, B)
    keep = rand(rng, n, n, B) .> 0.3f0              # KEEP mask, [key, outer, batch]
    keep[1, :, :] .= true                           # every row keeps at least one key
    ref = reference(q, k, v, keep, bias, dO)

    tune_default_fixed!(; dim=hd)
    register_trifast!(; dim=hd)
    dev = map(MtlArray, (q, k, v, keep, bias, dO))

    ok = true
    for name in (:default, :trifast)
        ok &= check(name, dev[1], dev[2], dev[3], dev[4], dev[5], dev[6], ref)
    end
    println(ok ? "PASS: both kernels match the reference on identical operands." :
            "FAIL: a kernel does NOT match the reference.")
    return ok
end

main() || exit(1)
