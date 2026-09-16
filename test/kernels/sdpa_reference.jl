# An independent (untiled, one-shot) ground truth for attention, computed without going through
# `Lux.scaled_dot_product_attention` or any tiled kernel — so a bug shared between the fast paths
# and their own reference cannot cancel out. Shared by every test that needs `lse`, which
# `Lux.scaled_dot_product_attention` does not expose: `test/kernels/backend.test.jl`,
# `test/ext/metal/launch.test.jl`, `test/ext/reactant/kernel_launch.test.jl`.
#
# Not explicitly `include`d by its consumers — like `convert_types` in `test/python/setup.jl`, it
# relies on `test/runtests.jl` including `test/kernels/runtests.jl` (which includes this) before
# anything in `test/ext/` that needs it. Kept out of `test/kernels/backend.test.jl` itself so a
# standalone re-include of that one file does not silently double-define it.

"""
    reference_sdpa(q, k, v, mask, bias) -> (output, lse)

One-shot (untiled) attention: `output` matches `Lux.scaled_dot_product_attention` exactly, and
`lse` is the log-sum-exp of the masked/biased logits along the key axis — the same quantity every
kernel in the `_sdpa_kernel` family caches for its reverse rule, computed independently of any of
them.
"""
function reference_sdpa(q, k, v, mask, bias)
    Impl = Lux.LuxLib.Impl
    Nd = ndims(q)
    bd = (2, 4:Nd...)
    scale = inv(sqrt(Float32(size(q, 1))))

    logits = Impl.batched_matmul(k, q .* scale;
        lhs_contracting_dim=1, rhs_contracting_dim=1, lhs_batching_dims=bd, rhs_batching_dims=bd)
    logits = Impl.apply_attention_mask_bias(logits, mask, bias)

    m = maximum(logits; dims=1)
    l = sum(exp.(logits .- m); dims=1)
    lse = m .+ log.(l)
    probs = exp.(logits .- lse)

    output = permutedims(
        Impl.batched_matmul(v, probs; lhs_contracting_dim=3, rhs_contracting_dim=1,
            lhs_batching_dims=bd, rhs_batching_dims=ntuple(Base.Fix2(+, 2), Val(Nd - 2))),
        (1, 3, 2, 4:Nd...))
    return output, lse
end
