# Correctness of the device backward (`flash_bwd_dq!`/`flash_bwd_dk!`/`flash_bwd_dv!` in
# `src/kernels/flash/backward.jl`) against an independent ground truth: Enzyme's reverse-mode
# gradient of a plain triple-nested-loop attention, written directly against the KERNEL's own
# indexing convention (mask `[key, i]`, bias `[key, query, bh]`, no `i` axis on bias — see the layout
# note on `flash_fwd!` in `src/kernels/flash/forward.jl`) rather than `LuxTriangleAttention`'s
# layer-level `[hd, H, Nq, Nj, B]` layout. This is deliberately NOT the same code the kernel or
# `reference_sdpa` is built from, so a shared bug in the tiling logic cannot cancel out in both
# places.
#
# Metal-only: there is no portable way to launch a raw-pointer `@kernel` through ordinary
# KernelAbstractions (CPU's own backend does not support `@index(Group/Local, ...)` the way
# `unsafe_indices=true` needs). Every existing use of `flash_fwd!` is likewise Metal-AOT-only or via
# Reactant's `custom_call`; this test uses the same Metal-AOT route `run_fwd!`/`run_bwd!` do. It
# lives under `test/kernels/` rather than `test/ext/metal/` because what is under test is the
# KERNEL's math (base-package code) — Metal is only the vehicle that can launch it; contrast
# `test/ext/metal/launch.test.jl` and `test/ext/metal/flash.test.jl`, which test Metal's OWN AOT
# build/launch machinery.

using Test, Random, Metal, Enzyme

# The extension module is not `using`-able by name; every other Metal-native call in this codebase
# goes through the seams `LuxTriangleAttention` itself exposes (`_DEVICE_BINARY` etc.) rather than
# reaching into the extension, but `run_fwd!`/`run_bwd!`/`build_bwd` have no such seam yet (they are
# not part of the `_sdpa_kernel` registry contract) — so this test fetches the extension module
# directly, the standard way to reach one from outside `Base.require`'s normal `using` path.
const LTAMetalExt = Base.get_extension(LuxTriangleAttention, :LuxTriangleAttentionMetalExt)

"""
    loop_reference!(o, q, k, v, mask, bias)

Reference forward, `[dim, n, nouter, bh]` layout, matching the kernel's exact addressing:
`mask[kk, i]` (shared across every query and batch/head), `bias[kk, js, bh]` (shared across every
`i`). `nouter = n` is triangle attention; anything else is the rectangular case (MSA rows/columns). Written as
literal loops — no `batched_matmul`, no tiling — so it shares no arithmetic path with either
`flash_fwd!` or `reference_sdpa`.
"""
function loop_reference!(o::AbstractArray{T,4}, q, k, v, mask, bias) where {T}
    dim, n, nouter, bh = size(q)
    scale = T(inv(sqrt(dim)))
    for b in 1:bh, i in 1:nouter, j in 1:n
        s = fill(T(-Inf), n)
        for kk in 1:n
            mask[kk, i] && continue
            acc = zero(T)
            for d in 1:dim
                acc += q[d, j, i, b] * k[d, kk, i, b]
            end
            s[kk] = acc * scale + bias[kk, j, b]
        end
        m = maximum(s)
        p = exp.(s .- m)
        l = sum(p)
        for d in 1:dim
            acc = zero(T)
            for kk in 1:n
                acc += p[kk] * v[d, kk, i, b]
            end
            o[d, j, i, b] = acc / l
        end
    end
    return nothing
end

"""
    loop_reference_grads(q, k, v, mask, bias, dO) -> (dq, dk, dv, dbias)

Enzyme's reverse-mode gradient of [`loop_reference!`](@ref) for a given upstream `dO`, at `Float64`
— the ground truth every device-kernel gradient below is checked against.
"""
function loop_reference_grads(q, k, v, mask, bias, dO)
    o = zeros(Float64, size(q))
    dq, dk, dv, dbias = zero(q), zero(k), zero(v), zero(bias)
    Enzyme.autodiff(Enzyme.Reverse, loop_reference!,
        Enzyme.Duplicated(o, copy(dO)), Enzyme.Duplicated(q, dq), Enzyme.Duplicated(k, dk),
        Enzyme.Duplicated(v, dv), Enzyme.Const(mask), Enzyme.Duplicated(bias, dbias))
    return dq, dk, dv, dbias
end

if Metal.functional()
    @testset "device backward vs Enzyme (loop reference)" begin
        rng = Random.Xoshiro(42)
        dim, n, bh = 4, 6, 2
        tg, kt = 4, 8   # kt >= n: a single key/query tile covers the whole problem

        q64 = randn(rng, Float64, dim, n, n, bh)
        k64 = randn(rng, Float64, dim, n, n, bh)
        v64 = randn(rng, Float64, dim, n, n, bh)
        bias64 = randn(rng, Float64, n, n, bh)
        dO64 = randn(rng, Float64, dim, n, n, bh)
        mask = falses(n, n)
        mask[2, 3] = true   # a sparse hole only — a fully-masked row softmaxes to NaN

        dq_ref, dk_ref, dv_ref, dbias_ref = loop_reference_grads(q64, k64, v64, mask, bias64, dO64)

        T = Float32
        q, k, v, bias, dO = T.(q64), T.(k64), T.(v64), T.(bias64), T.(dO64)
        qd, kd, vd, biasd, maskd, dOd = MtlArray.((q, k, v, bias, mask, dO))

        o = MtlArray(zeros(T, dim, n, n, bh))
        lse = MtlArray(zeros(Float32, n, n, bh))
        LTAMetalExt.run_fwd!(o, lse, qd, kd, vd, biasd, maskd; tg, kt, n, heads=bh, dim)

        dq = MtlArray(zeros(T, dim, n, n, bh))
        dk = MtlArray(zeros(T, dim, n, n, bh))
        dv = MtlArray(zeros(T, dim, n, n, bh))
        dbias = MtlArray(zeros(Float32, n, n, bh))
        LTAMetalExt.run_bwd!(dq, dk, dv, dbias, qd, kd, vd, biasd, maskd, o, dOd, lse;
            tg, kt, n, heads=bh, dim)

        @test Array(o) ≈ T.(let o64 = zeros(Float64, dim, n, n, bh)
            loop_reference!(o64, q64, k64, v64, mask, bias64)
            o64
        end) rtol = 1.0f-4

        @test Array(dq) ≈ T.(dq_ref) rtol = 1.0f-3 atol = 1.0f-5
        @test Array(dk) ≈ T.(dk_ref) rtol = 1.0f-3 atol = 1.0f-5
        @test Array(dv) ≈ T.(dv_ref) rtol = 1.0f-3 atol = 1.0f-5
        @test Array(dbias) ≈ T.(dbias_ref) rtol = 1.0f-3 atol = 1.0f-5
    end

    # Rectangular problems (`nouter ≠ n`): MSA row attention attends over residues for each of
    # `N_seq` rows, column attention the reverse. Both orderings of the two extents, with ragged
    # tiles (`kt < n`, `n` not a multiple of `tg`), so every index the new `nouter` argument enters
    # — the query row, the mask base and the `lse` row — is exercised away from the square case
    # where a `nouter`/`n` mix-up would be invisible.
    @testset "rectangular n=$n nouter=$nouter" for (n, nouter) in ((6, 9), (9, 4))
        rng = Random.Xoshiro(7)
        dim, bh = 4, 2
        tg, kt = 4, 4

        q64 = randn(rng, Float64, dim, n, nouter, bh)
        k64 = randn(rng, Float64, dim, n, nouter, bh)
        v64 = randn(rng, Float64, dim, n, nouter, bh)
        bias64 = randn(rng, Float64, n, n, bh)
        dO64 = randn(rng, Float64, dim, n, nouter, bh)
        mask = falses(n, nouter)
        mask[2, 3] = true
        mask[n, nouter] = true

        dq_ref, dk_ref, dv_ref, dbias_ref = loop_reference_grads(q64, k64, v64, mask, bias64, dO64)
        o64 = zeros(Float64, dim, n, nouter, bh)
        loop_reference!(o64, q64, k64, v64, mask, bias64)

        T = Float32
        qd, kd, vd, biasd, maskd, dOd = MtlArray.((T.(q64), T.(k64), T.(v64), T.(bias64), mask,
            T.(dO64)))
        o = MtlArray(zeros(T, dim, n, nouter, bh))
        lse = MtlArray(zeros(Float32, n, nouter, bh))
        LTAMetalExt.run_fwd!(o, lse, qd, kd, vd, biasd, maskd; tg, kt, n, nouter, heads=bh, dim)
        @test Array(o) ≈ T.(o64) rtol = 1.0f-4

        dq, dk, dv = (MtlArray(zeros(T, dim, n, nouter, bh)) for _ in 1:3)
        dbias = MtlArray(zeros(Float32, n, n, bh))
        LTAMetalExt.run_bwd!(dq, dk, dv, dbias, qd, kd, vd, biasd, maskd, o, dOd, lse;
            tg, kt, n, nouter, heads=bh, dim)
        @test Array(dq) ≈ T.(dq_ref) rtol = 1.0f-3 atol = 1.0f-5
        @test Array(dk) ≈ T.(dk_ref) rtol = 1.0f-3 atol = 1.0f-5
        @test Array(dv) ≈ T.(dv_ref) rtol = 1.0f-3 atol = 1.0f-5
        # `dbias` sums over every outer index: `nouter` atomic adds per address.
        @test Array(dbias) ≈ T.(dbias_ref) rtol = 1.0f-3 atol = 1.0f-5
    end
else
    @warn "Metal not functional; skipping device backward correctness test" maxlog = 1
end
