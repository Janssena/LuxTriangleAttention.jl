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
# Reactant's `custom_call`; this test runs each stage on packed device buffers through
# `LuxTriangleAttention.run_stage!`, the launch the tuner verifies candidates with. It
# lives under `test/kernels/` rather than in `test/kernels/launch.gpu.test.jl` because what is under test is the
# KERNEL's math (base-package code) — Metal is only the vehicle that can launch it.

using Test, Random, Enzyme

const LTAb = LuxTriangleAttention

"""
    run_flash(q, k, v, bias, mask, dO; tg, kt, n, nouter=n, bh) -> (o, lse, dq, dk, dv, dbias)

The `:default` forward and its three backward stages at one configuration (`sb = min(4, kt)`), on
packed `[dim, n, nouter, bh]` device buffers (`nh = 1`, one mask shared by all `bh` slices), each
binary built with `build_stage` and launched with `run_stage!`. `dbias` comes back in the plain
keys-fastest layout (the kernel writes it transposed).
"""
function run_flash(q, k, v, bias, mask, dO; tg, kt, n, nouter=n, bh)
    T, dim = eltype(q), size(q, 1)
    P = LTAb.packed_problem(q, k, v, bias, mask, dO, zeros(T, size(q)),
        zeros(Float32, n, nouter, bh), zeros(Float32, n, nouter, bh))
    geo = (; n, nouter, heads=bh, mheads=bh)
    bin(s, p) = LTAb.build_stage(LTAb.FlashScalar(), s, T, dim, p)
    LTAb.run_stage!(LTAb.FlashScalar(), :fwd, bin(:fwd, (; tg, kt, sb=min(4, kt))), P; geo...)
    copyto!(P.delta, LTAb.flash_bwd_delta(P.o, P.dO))
    for s in (:dq, :dk, :dv)
        LTAb.run_stage!(LTAb.FlashScalar(), s, bin(s, (; tg, kt)), P; geo...)
    end
    return (Array(P.o), Array(P.lse), Array(P.dq), Array(P.dk), Array(P.dv),
        permutedims(Array(P.dbias_t), (2, 1, 3)))
end

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
        if m == -Inf        # no live key: uniform attention, o = mean(v) (its gradient reaches v only)
            for d in 1:dim
                acc = zero(T)
                for kk in 1:n
                    acc += v[d, kk, i, b]
                end
                o[d, j, i, b] = acc / n
            end
            continue
        end
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

if GPU !== nothing
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
        mask[2, 3] = true   # a sparse hole
        mask[:, 5] .= true  # and one fully masked slice: uniform attention, o = mean(v)

        dq_ref, dk_ref, dv_ref, dbias_ref = loop_reference_grads(q64, k64, v64, mask, bias64, dO64)

        T = Float32
        q, k, v, bias, dO = T.(q64), T.(k64), T.(v64), T.(bias64), T.(dO64)
        o, lse, dq, dk, dv, dbias = run_flash(q, k, v, bias, mask, dO; tg, kt, n, bh)

        @test all(==(-Inf), Array(lse)[:, 5, :])           # the dead slice's marker
        @test all(isfinite, Array(lse)[:, [1, 2, 3, 4, 6], :])
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
        mask[:, 2] .= true      # a fully masked slice: o = mean(v), dv = Σ dO / n, nothing else

        dq_ref, dk_ref, dv_ref, dbias_ref = loop_reference_grads(q64, k64, v64, mask, bias64, dO64)
        o64 = zeros(Float64, dim, n, nouter, bh)
        loop_reference!(o64, q64, k64, v64, mask, bias64)

        T = Float32
        o, lse, dq, dk, dv, dbias = run_flash(T.(q64), T.(k64), T.(v64), T.(bias64), mask, T.(dO64);
            tg, kt, n, nouter, bh)
        @test all(==(-Inf), Array(lse)[:, 2, :])
        @test Array(o) ≈ T.(o64) rtol = 1.0f-4

        @test Array(dq) ≈ T.(dq_ref) rtol = 1.0f-3 atol = 1.0f-5
        @test Array(dk) ≈ T.(dk_ref) rtol = 1.0f-3 atol = 1.0f-5
        @test Array(dv) ≈ T.(dv_ref) rtol = 1.0f-3 atol = 1.0f-5
        # `dbias` sums over every outer index: `nouter` atomic adds per address.
        @test Array(dbias) ≈ T.(dbias_ref) rtol = 1.0f-3 atol = 1.0f-5
    end

    # Mixed precision: 16-bit storage of the operands, `dO` and the gradients, Float32 accumulation. The reference
    # is the Float64 gradient on the same rounded values, so only the kernel's arithmetic and the rounding of its
    # outputs are measured. BFloat16 has Float32's range but an 8-bit mantissa, hence the looser bounds; a
    # BFloat16 ACCUMULATOR (rather than storage) would lose far more, over `n` keys, and is not what the kernel does.
    relL2(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))
    @testset "$(nameof(T)) storage, Float32 accumulation" for T in (Float16, LTAb.BFloat16)
        rng = Random.Xoshiro(11)
        dim, n, nouter, bh = 8, 12, 5, 2
        tg, kt = 8, 8
        r(dims...) = T.(randn(rng, Float32, dims...))          # `randn` has no BFloat16 method
        q, k, v, bias, dO = r(dim, n, nouter, bh), r(dim, n, nouter, bh), r(dim, n, nouter, bh), r(n, n, bh),
            r(dim, n, nouter, bh)
        mask = falses(n, nouter)
        mask[2, 3] = true
        mask[:, 2] .= true      # fully masked slice

        dq_ref, dk_ref, dv_ref, dbias_ref = loop_reference_grads(Float64.(q), Float64.(k), Float64.(v), mask,
            Float64.(bias), Float64.(dO))
        o64 = zeros(Float64, dim, n, nouter, bh)
        loop_reference!(o64, Float64.(q), Float64.(k), Float64.(v), mask, Float64.(bias))

        o, lse, dq, dk, dv, dbias = run_flash(q, k, v, bias, mask, dO; tg, kt, n, nouter, bh)
        tol_o, tol_g = T === Float16 ? (2.0e-3, 1.0e-2) : (1.0e-2, 3.0e-2)
        @test eltype(o) === T && eltype(dq) === T
        @test all(isfinite, o) && all(isfinite, dq) && all(isfinite, dk) && all(isfinite, dv)
        @test relL2(o, o64) < tol_o
        @test relL2(dq, dq_ref) < tol_g
        @test relL2(dk, dk_ref) < tol_g
        @test relL2(dv, dv_ref) < tol_g
        @test relL2(dbias, dbias_ref) < tol_g
    end
else
    @warn "Metal not functional; skipping device backward correctness test" maxlog = 1
end
