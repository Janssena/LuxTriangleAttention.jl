# Shared inputs for the 16-bit attention tests: `test/layers/attention.test.jl` (eager, CPU) and
# `test/layers/attention.test.jl` (compiled). Like `reference_sdpa`, not `include`d by its
# consumers: `test/runtests.jl` includes it first.
#
# The case is built so that the attention LOGITS leave Float16's range BEFORE the 1/√d scale and stay
# inside it after: that is exactly the window in which "scale, then multiply" is fine and "multiply,
# then scale" overflows, which is what a compiler's reassociation does to `Lux.scaled_dot_product_attention`.

"""
    f16_overflow_attention(rng; chn=16, hd=16, H=2, N=8, B=1, w=4f0) -> NamedTuple

A fused-QKV `Attention` on a sequence `[chn, N, B]` (or, with `pair=true`, a pair input `[chn, N, N, B]`, the
shape `chunk_size` actually slices) whose unscaled logits are ≈ 1.5e5 (> 65504) and whose
scaled logits are ≈ 3.7e4 (< 65504). Returns the layer, its parameters and input in Float16 (`ps16`,
`st16`, `x16`) and, rounded through Float16 first, in Float64 (`ps64`, `st64`, `x64`) for a reference
that sees the same numbers, plus `maxlogit` / `maxscaled` (Float64, from the parameters themselves)
for the tests to assert the window.

Key `j` gets a larger projection than key `j-1`, so the softmax is close to one-hot on the last
unmasked key with a gap far above Float16's spacing at these magnitudes (32): the output is then
robust to logit rounding and a wrong path fails on value, not just on `Inf`/`NaN`.
"""
function f16_overflow_attention(rng; chn=16, hd=16, H=2, N=8, B=1, w=4.0f0, pair=false)
    layer = Attention(chn, hd, H)
    ps, st = Lux.setup(rng, layer)
    HD = H * hd
    W = copy(ps.qkv.weight)
    W[1:(2HD), :] .= w                        # q and k rows: every head dim sees w·Σₙ x
    ps = merge(ps, (; qkv=merge(ps.qkv, (; weight=W))))

    s = 1.0f0 .+ 0.5f0 .* (1:N) ./ N          # grows along the attended axis
    x = repeat(reshape(s, 1, N, 1), chn, 1, B) .+ 0.01f0 .* randn(rng, Float32, chn, N, B)

    # A pair input `[chn, Ni, Nj, B]` attends over `Ni` for each of `Nj` columns (the shape `chunk_size` slices:
    # dim 4 of the prepared q/k/v is `Nj`); every column carries the same window, so the logit range holds.
    pair && (x = repeat(reshape(x, chn, N, 1, B), 1, 1, N, 1) .+ 0.001f0 .* randn(rng, Float32, chn, N, N, B))
    ps16, st16, x16 = Lux.f16(ps), Lux.f16(st), Float16.(x)
    ps64, st64, x64 = Lux.f64(ps16), Lux.f64(st16), Float64.(x16)

    xs = pair ? x64[:, :, 1, :] : x64                      # the attended window, `[chn, N, B]`
    qkv = ps64.qkv.weight * reshape(xs, chn, :)           # [3HD, N·B]
    q = reshape(qkv[1:HD, :], hd, H, :)
    k = reshape(qkv[(HD+1):(2HD), :], hd, H, :)
    maxlogit = maximum(h -> maximum(abs, q[:, h, :]' * k[:, h, :]), 1:H)
    return (; layer, ps16, st16, x16, ps64, st64, x64, maxlogit, maxscaled=maxlogit / sqrt(hd))
end

relerr(y, ref) = sqrt(sum(abs2, Float64.(y) .- Float64.(ref)) / sum(abs2, Float64.(ref)))

"""
    f16_range_qkv(hd, H, n, nouter, B) -> (q, k, v)

Float16 `[hd, H, n, nouter, B]` operands for the KERNEL tests (no layer, no parameters). With
`a = √(8e4/hd)`, `q ≡ a` and `k_j = a·s_j`, `s_j = 1 + j/2n ∈ (1, 1.5]`: the logits are `8e4·s_j` (> 65504)
before the 1/√d scale and at most `1.2e5/√hd` (< 65504 for every `hd ≥ 4`) after. The softmax is close to
one-hot on the last key with a gap far above Float16's spacing, so rounding cannot change the answer.
"""
function f16_range_qkv(hd, H, n, nouter, B; rng=Random.Xoshiro(5))
    a = sqrt(8.0f4 / hd)
    s = reshape(1.0f0 .+ 0.5f0 .* (1:n) ./ n, 1, 1, n, 1, 1)
    q = fill(Float16(a), hd, H, n, nouter, B)
    k = Float16.(a .* s .* ones(Float32, hd, H, 1, nouter, B))
    v = Float16.(randn(rng, Float32, hd, H, n, nouter, B))
    return q, k, v
end
