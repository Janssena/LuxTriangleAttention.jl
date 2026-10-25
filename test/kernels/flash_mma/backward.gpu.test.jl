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
