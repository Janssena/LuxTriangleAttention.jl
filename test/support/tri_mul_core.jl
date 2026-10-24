# Shared by `test/layers/tri_mul_core.test.jl` (eager) and `test/ext/reactant/tri_mul_core.test.jl`
# (compiled): a per-element Float64 reference for `TriMulCore`, and the parameter helpers its tests use.
# Like `reference_sdpa`, not `include`d by its consumers: `test/runtests.jl` includes it first.
#
# Every configuration is `(fused, fused_glu, use_bias)`: one projection for `a` and `b` or one each, each a
# fused GLU (one Dense, values then gates) or a GLU with its own gate Dense.

using LuxTriangleAttention: fold_trimul

tmc_cast(::Type{T}, x::AbstractArray) where {T} = T.(x)
tmc_cast(::Type{T}, x::NamedTuple) where {T} = map(v -> tmc_cast(T, v), x)

# Random, non-trivial parameters (Lux zero-initialises biases and sets the LayerNorm to the identity).
tmc_perturb(rng, p::AbstractArray) = p .+ 0.3f0 .* randn(rng, Float32, size(p))
tmc_perturb(rng, p::NamedTuple) = map(q -> tmc_perturb(rng, q), p)

# `fold_trimul` takes a `TriangleMultiplication`'s parameters; a bare core's go under `core`.
function tmc_fold(ps, st, k)
    ps_f, st_f = fold_trimul((; core=ps), (; core=st), k)
    return ps_f.core, st_f.core
end

# Add `v` to the VALUE biases of one GLU's parameters (the first rows of its Dense; all of them when the
# gate has its own Dense, the first half when one Dense holds values then gates).
function tmc_add_values!(g, v)
    n = isempty(g.gate) ? length(g.linear.bias) ÷ 2 : length(g.linear.bias)
    n == length(v) || error("expected $n value rows, got $(length(v))")
    view(g.linear.bias, 1:n) .+= v
    return g
end

# Large, per-channel-different value biases for `a` and `b`, so that Σₖ a·b exceeds Float16's range while
# the channels still differ enough for the output LayerNorm to be well conditioned.
function tmc_bump_values(ps, v)
    ps = deepcopy(ps)
    if !isempty(ps.glu_ab)
        tmc_add_values!(ps.glu_ab, vcat(v, v))
    else
        tmc_add_values!(ps.glu_a, v)
        tmc_add_values!(ps.glu_b, v)
    end
    return ps
end

# The core written out per element in Float64: projections, GLU, mask, `Σₖ` with explicit loops (no
# planes, no batched matmul), LayerNorm over the hidden channels, output gate. Also returns max |X|.
function tmc_reference(x, mask, ps; outgoing, eps)
    C, N, _, B = size(x)
    H = length(ps.layer_norm_out.scale)
    sig(v) = 1 ./ (1 .+ exp.(-v))
    lin(p, v) = haskey(p, :bias) ? p.weight * v .+ p.bias : p.weight * v
    function glu_ab(xt)
        g = ps.glu_ab
        if isempty(g.gate)                     # one Dense: values (a, b), then gates (a, b)
            P = lin(g.linear, xt)
            v, gt = P[1:2H, :], P[(2H + 1):(4H), :]
        else
            v, gt = lin(g.linear, xt), lin(g.gate, xt)
        end
        r = v .* sig(gt)
        return r[1:H, :], r[(H + 1):(2H), :]
    end
    function glu_one(g, xt)
        if isempty(g.gate)                     # one Dense: values, then gates
            P = lin(g.linear, xt)
            return P[1:H, :] .* sig(P[(H + 1):(2H), :])
        end
        return lin(g.linear, xt) .* sig(lin(g.gate, xt))
    end

    y, Xmax = zeros(Float64, C, N, N, B), 0.0
    for b in 1:B
        xt = reshape(x[:, :, :, b], C, N * N)  # token (i, j) at column i + N (j - 1)
        a, bb = isempty(ps.glu_ab) ? (glu_one(ps.glu_a, xt), glu_one(ps.glu_b, xt)) : glu_ab(xt)
        if mask !== nothing
            m = reshape(Float64.(ndims(mask) == 3 ? mask[:, :, b] : mask), 1, :)
            a, bb = a .* m, bb .* m
        end
        A, Bm = reshape(a, H, N, N), reshape(bb, H, N, N)
        X = zeros(H, N, N)
        for h in 1:H, i in 1:N, j in 1:N
            X[h, i, j] = outgoing ? sum(A[h, i, k] * Bm[h, j, k] for k in 1:N) :
                                    sum(A[h, k, i] * Bm[h, k, j] for k in 1:N)
        end
        Xmax = max(Xmax, maximum(abs, X))
        Xt = reshape(X, H, N * N)
        μ = sum(Xt; dims=1) ./ H
        σ² = sum(abs2, Xt .- μ; dims=1) ./ H
        Yn = (Xt .- μ) ./ sqrt.(σ² .+ eps) .* vec(ps.layer_norm_out.scale) .+ vec(ps.layer_norm_out.bias)
        o = lin(ps.glu_out.linear, Yn) .* sig(lin(ps.glu_out.gate, xt))
        y[:, :, :, b] = reshape(o, C, N, N)
    end
    return (; y, Xmax)
end

const TMC_CONFIGS = (
    (true, true, false), (true, true, true), (true, false, false), (true, false, true),
    (false, true, false), (false, true, true), (false, false, false), (false, false, true),
)
tmc_name((fused, fused_glu, use_bias), out) =
    "fused=$fused fused_glu=$fused_glu bias=$use_bias $(out ? "out" : "in")"
