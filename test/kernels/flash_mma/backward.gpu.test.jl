# The backward of `src/kernels/flash_mma/backward.jl` (`:dkv` and `:dq`), run through its KernelDispatch
# method directly (shared setup in `flash_mma/runtests.jl`) on the `o` and `lse` its own forward
# wrote, against Enzyme's reverse-mode gradient of LuxLib's `scaled_dot_product_attention`
# (`mma_reference_grads`). Float16 is compared against the Float64 gradient of the same (upcast)
# values, so only the kernel's own rounding is measured.

@testset "$T, $(shape), bias=$hb, mask=$hm" for (T, shape, hb, hm) in MMA_CASES
    c = mma_case(T, shape, hb, hm)
    o, lse = mma_forward(c, T, shape)
    g = mma_backward(c, o, lse)
    ref = mma_reference_grads(c)

    @test all(x -> x === nothing || eltype(x) === T, g)   # dbias is `nothing` without a bias
    for (name, a, r) in zip((:dq, :dk, :dv, :dbias), g, ref)
        r === nothing && (@test a === nothing; continue)
        @test all(isfinite, Array(a))
        @test relL2_mma(Array(a), r) < tol_bwd(T)
    end
end

# Gradients of a storage type the device computes at another type (CUDA: Float32 at Float16) come back in the
# caller's types, `dbias` in the bias's, accurate to the narrower type.
@testset "$T storage computed at $(LTA_.mma_compute_type(LTA_.mma_backend(), T))" for T in CAST_TYPES
    Tc = LTA_.mma_compute_type(LTA_.mma_backend(), T)
    shape = (32, 2, 37, 5, 2)
    c = mma_case(T, shape, true, true)
    o, lse = mma_forward(c, T, shape)
    g = mma_backward(c, o, lse)
    ref = mma_reference_grads(c)
    @test all(x -> eltype(x) === T, g)
    for (name, a, r) in zip((:dq, :dk, :dv, :dbias), g, ref)
        @test all(isfinite, Array(a))
        @test relL2_mma(Array(a), r) < tol_bwd(Tc) * 4
    end
end

# Repeatability: the same inputs through the same backward, `REPEATS` times, must give bit-identical gradients. A key-tile (`dK`,
# `dV`) or query-row (`dQ`) thread owns its output, so those are deterministic by construction. `dbias` sums over the outer axis
# with float atomics, whose order is the scheduler's: it is required to repeat only once `DBIAS_DETERMINISTIC` says it does, and
# until then the number of distinct results is reported. A large outer extent, so that many atomics hit each address.
@testset "repeatability, $T, bias=$hb" for T in MMA_TYPES, hb in (true, false)
    shape = (32, 2, 96, 96, 1)
    REPEATS = 5
    c = mma_case(T, shape, hb, true)
    o, lse = mma_forward(c, T, shape)
    runs = [Tuple(x === nothing ? nothing : Array(x) for x in mma_backward(c, o, lse)) for _ in 1:REPEATS]
    for (i, name) in enumerate((:dq, :dk, :dv))
        @test all(r -> isequal(r[i], runs[1][i]), runs)
    end
    if hb
        distinct = length(unique(r[4] for r in runs))
        DBIAS_DETERMINISTIC ? (@test distinct == 1) :
            @info "dbias repeatability ($T): $distinct distinct result(s) in $REPEATS runs"
    end
end

# The output residual (`output_residual` preference): the forward returns `(o, lse, e)` for a type the device computes in (not Float32
# storage on Metal), `e = T(O32 − o)` is what the rounding of `o` dropped, so it is bounded by half an ulp of `o`, and a backward that
# uses `o + e` for δ is no less accurate than one that does not. With the preference off, the forward returns the pair.
@testset "output residual, $T" for T in MMA_ALL_TYPES
    shape = (32, 2, 96, 96, 1)
    Tc = LTA_.mma_compute_type(LTA_.mma_backend(), T)
    c = mma_case(T, shape, true, true)
    out = mma_forward(c, T, shape)
    @test length(out) == (LTA_._mma_resid(Tc) ? 3 : 2)
    if LTA_._mma_resid(Tc)
        o, lse, e = out
        @test size(e) == size(o)
        p = Tc === Float16 ? 11 : 8                            # significand bits
        ao, ae = abs.(Float64.(Array(o))), abs.(Float64.(Array(e)))
        @test all(ae .<= ao .* 2.0^-(p - 1) .+ Float64(floatmin(Tc)))     # |e| ≤ half an ulp of o (with slack)
        @test count(>(0), ae) > length(ae) ÷ 4                 # and it is not all zero
        ref = mma_reference_grads(c)
        err(g) = maximum(relL2_mma(Array(a), r) for (a, r) in zip(g[1:3], ref[1:3]))
        with = mma_backward(c, o, lse, e)
        without = mma_backward(c, o, lse)
        @test err(with) <= 1.1 * err(without)
    end
end
