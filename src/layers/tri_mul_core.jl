"""
    TriMulCore(chn_in, chn_hidden; is_outgoing=true, fused=true, kwargs...)

Core logic for Triangle Multiplication. Handles projections, masking, and 
batched matrix multiplication for edge updates.

## Arguments
- `chn_in`: Input channels.
- `chn_hidden`: Hidden channels.

## Keyword Arguments
- `is_outgoing`: If `true`, performs "Outgoing" multiplication.
- `fused`: If `true`, fuses the A & B projections into a single GLU operation.
- `fused_glu`: If `true`, uses fused GLU layers.
- `use_bias`: Boolean for internal layer biases.
- `rank`: The rank (ndims) of the input.

## Inputs
- `x`: Input tensor with shape `[C, N, N, B]`.
- `mask`: Optional mask with shape `[N, N, B]`.

## Returns
- `y`: Output tensor with shape `[chn_in, N, N, B]`.
- `st`: Updated state.
"""
struct TriMulCore{F<:StaticBool, DIR<:StaticBool, GAB, GA, GB, LNO, GOUT} <: Lux.AbstractLuxContainerLayer{(:glu_ab, :glu_a, :glu_b, :layer_norm_out, :glu_out)}
    fused::F
    is_outgoing::DIR
    glu_ab::GAB
    glu_a::GA
    glu_b::GB
    layer_norm_out::LNO
    glu_out::GOUT
    chn_hidden::Int
end

function TriMulCore(
    chn_in::Int, chn_hidden::Int;
    is_outgoing::Union{Bool, StaticBool}=static(true),
    fused::Union{<:StaticBool,Bool}=static(true), 
    fused_glu::Union{<:StaticBool,Bool}=static(true),
    layernorm_eps=1f-5, use_bias=false, rank=4, kwargs...
)   
    fused_static = static(fused)
    
    if known(fused_static)
        # Fuses p_a, p_b, g_a, g_b into a single operation (Dense to 2H/4H depending on chn_hidden)
        glu_ab = ChannelsLast(GatedLinearUnit(chn_in => 2 * chn_hidden; fused=fused_glu, use_bias))
        glu_a = Lux.NoOpLayer()
        glu_b = Lux.NoOpLayer()
    else
        glu_ab = Lux.NoOpLayer()
        glu_a = ChannelsLast(GatedLinearUnit(chn_in => chn_hidden; fused=fused_glu, use_bias))
        glu_b = ChannelsLast(GatedLinearUnit(chn_in => chn_hidden; fused=fused_glu, use_bias))
    end

    layer_norm_out = ChannelsLast(StableLayerNorm((chn_hidden, ntuple(one, rank-2)...); dims=1, epsilon=layernorm_eps))
    
    # Output Gate: Dual input (chn_hidden from Y, chn_in from X)
    glu_out = ChannelsLast(GatedLinearUnit((chn_hidden, chn_in) => chn_in; fused=false, use_bias))

    return TriMulCore(
        fused_static,
        static(is_outgoing),
        glu_ab, glu_a, glu_b,
        layer_norm_out,
        glu_out,
        chn_hidden
    )
end

# The contraction wants one `[N, N]` matrix per hidden channel and batch element ("planes"), while
# the rest of the layer is channel-first (`[C, N, N, B]`). The projections are therefore computed
# TOKEN-major, `xᵀ·Wᵀ` → `[N²·B, 2H]`, which IS the planes layout: column `h` holds the `B`
# contiguous `[N, N]` matrices of channel `h`. The contraction then runs on views of that array with
# a lazy `batched_transpose` and no `permutedims`. The output side stays token-major too: the output
# LayerNorm reduces over dim 2 of `[N²·B, H]` and `glu_out` runs as `Y·Wᵀ`, so the only transpose is of
# the final `[N²·B, C]` result back to channel-first. 
# The sublayers are therefore wrapped in `ChannelsLast`: the same parameters as the plain layers, 
# applied to `[T, C]` data.
function (m::TriMulCore)(x_norm::AbstractArray{T,4}, mask, ps, st) where {T}
    C, N, _, B = size(x_norm)
    H = m.chn_hidden
    x_tok = transpose(reshape(x_norm, C, N * N * B))         # [T, C] lazy, tokens (i, j, b)

    a, b, st_ab = _ab_planes(m, x_tok, mask, ps, st, N, B)    # [N, N, B·H] each, slot = b + B·(h-1)
    # 16-bit data takes one of two paths (`ContractionMode`, src/precision.jl). The contraction stays
    # in THIS body for every mode — moved into a helper, `Lux.batched_matmul` on these views no
    # longer infers — and only its operands' width and the output LayerNorm depend on the mode.
    mode = contraction_mode(x_norm, st)
    X = _contract(m.is_outgoing, _widen(mode, a), _widen(mode, b))   # [N, N, B·H]
    y_tok = reshape(X, N * N * B, H)                          # token-major: slot order = token order

    y_norm, layer_norm_out = _normalize_out(mode, m, y_tok, ps, st, realtype(eltype(a)))
    out_tok, glu_out = m.glu_out((y_norm, x_tok), ps.glu_out, st.glu_out)    # [N²B, C]

    # Each sublayer's state is merged into its old one, so a key it does not return (`fold_k`, or a
    # `gate` the GLU leaves out) stays for the next call and the state keeps its type.
    st = merge(st, st_ab, (; layer_norm_out=merge(st.layer_norm_out, layer_norm_out),
                             glu_out=merge(st.glu_out, glu_out)))
    return reshape(permutedims(out_tok, (2, 1)), C, N, N, B), st
end

# 16-bit data with unscaled parameters (`HalfSafe`): the contraction overflows Float16 (AlphaFold2's
# multimer template stack reaches |X| ≈ 4.8e5 at N = 494), so it runs in Float32. X is only [N²·B, H].
_widen(::HalfSafe, a) = Float32.(a)
_widen(_, a) = a

_normalize_out(::FullPrecision, m, y_tok, ps, st, T) =
    m.layer_norm_out(y_tok, ps.layer_norm_out, st.layer_norm_out)

# `Folded(k)`: the value halves of a and b carry 2⁻ᵏ (`fold_trimul`), so X = Σ a·b / 4ᵏ stays in range.
# The output LayerNorm is scale-invariant and removes the 4⁻ᵏ by itself — exactly, once ε is scaled
# with X's variance (16⁻ᵏ). That ε is far below Float16's range, hence Float32 statistics, and X is
# normalised in Float32 too: 1/√(σ² + ε·16⁻ᵏ) reaches 4ᵏ/√ε ≈ 1e6 for a token whose X is all zero (a
# fully masked pair), which is `Inf` in Float16 and turns that token's 0·rstd into NaN.
_normalize_out(mode::Folded, m, y_tok, ps, st, T) =
    _layer_norm_out_f32(m, Float32.(y_tok), ps, Float32(m.layer_norm_out.layer.layer.epsilon) * 16.0f0^(-mode.k), T),
    st.layer_norm_out

# `HalfSafe`: X is Float32; normalise in Float32 and narrow to the data's type.
_normalize_out(::HalfSafe, m, y_tok, ps, st, T) =
    _layer_norm_out_f32(m, y_tok, ps, Float32(m.layer_norm_out.layer.layer.epsilon), T),
    st.layer_norm_out

function _layer_norm_out_f32(m, y_tok, ps, ε, ::Type{T}) where {T}
    pl = ps.layer_norm_out
    γ = haskey(pl, :scale) ? reshape(pl.scale, 1, :) : nothing
    β = haskey(pl, :bias) ? reshape(pl.bias, 1, :) : nothing
    return layernorm_f32stats(y_tok, γ, β, m.layer_norm_out.layer.layer.activation, 2, ε, T)
end

# Outgoing (AlphaFold2 Alg 11): X[i, j] = Σ_k a[i, k] b[j, k]  →  A · Bᵀ per plane
_contract(::True, a, b) = Lux.batched_matmul(a, Lux.batched_transpose(b))
# Incoming (AlphaFold2 Alg 12): X[i, j] = Σ_k a[k, i] b[k, j]  →  Aᵀ · B per plane
_contract(::False, a, b) = Lux.batched_matmul(Lux.batched_transpose(a), b)

"""
    _ab_planes(m::TriMulCore, x_tok, mask, ps, st, N, B) -> (a, b, st_ab)

Project the normalised input into the gated, masked `a` and `b` representations, laid out as
`[N, N, B·H]` planes (batch slot `b + B·(h-1)`), ready for the batched contraction. `st_ab` holds the
updated states of the GLU(s) that ran, keyed by their field names.

With `fused = true` one GLU produces both (`[N²·B, 2H]`, `a` in the first `H` columns); the planes
are contiguous views of it. With `fused = false`, `glu_a` and `glu_b` produce one each.
"""
function _ab_planes(m::TriMulCore{True}, x_tok, mask, ps, st, N, B)
    H = m.chn_hidden
    ab, glu_ab = m.glu_ab(x_tok, ps.glu_ab, st.glu_ab)
    ab = _apply_tri_mask(ab, mask)                                             # [N²B, 2H]
    planes = reshape(ab, N, N, 2 * B * H)
    a = view(planes, :, :, 1:(B * H))
    b = view(planes, :, :, (B * H + 1):(2 * B * H))
    return a, b, (; glu_ab=merge(st.glu_ab, glu_ab))
end

function _ab_planes(m::TriMulCore{False}, x_tok, mask, ps, st, N, B)
    H = m.chn_hidden
    a, glu_a = m.glu_a(x_tok, ps.glu_a, st.glu_a)
    b, glu_b = m.glu_b(x_tok, ps.glu_b, st.glu_b)
    a, b = _apply_tri_mask(a, mask), _apply_tri_mask(b, mask)
    return reshape(a, N, N, B * H), reshape(b, N, N, B * H),
           (; glu_a=merge(st.glu_a, glu_a), glu_b=merge(st.glu_b, glu_b))
end

"""
    _apply_tri_mask(ab, mask)

Zero the rows of token-major `ab::[N²·B, ·]` whose pair is masked out (`true` = keep). `mask` is
`[N, N, B]`, or `[N, N]` shared by every batch element. Out of place, so it is differentiable; under
Reactant it fuses into the GLU broadcast.
"""
_apply_tri_mask(ab, ::Nothing) = ab
function _apply_tri_mask(ab::AbstractMatrix{T}, mask::AbstractArray) where {T}
    nn = size(mask, 1) * size(mask, 2)
    ab3 = reshape(ab, nn, :, size(ab, 2))                     # [N², B, cols]
    return reshape(ifelse.(reshape(mask, nn, :, 1), ab3, zero(T)), size(ab))
end

"""
    fold_trimul(ps, st, k) -> (ps, st)

Prepare a [`TriangleMultiplication`](@ref)'s parameters for the [`Folded`](@ref) path: scale the
VALUE projections of `a` and `b` (never the gates) by 2⁻ᵏ and record `fold_k = k` in the core's state.
Apply it to the Float32 parameters, before casting to 16 bits (the scaling is a power of two, so it
is exact either way, but subnormal weights would round). Use [`fold_k`](@ref)`(L)` for the longest
sequence the parameters must serve.

The contraction then sums `Σ a·b / 4ᵏ`, and the output LayerNorm (scale-invariant, with ε scaled by
16⁻ᵏ in the layer) makes the layer's result exactly the unscaled one.
"""
function fold_trimul(ps::NamedTuple, st::NamedTuple, k::Integer)
    s, c = 2.0f0^(-k), ps.core
    core = if !isempty(c.glu_ab)
        merge(c, (; glu_ab=_scale_glu_values(c.glu_ab, s)))
    else                                                     # separate glu_a / glu_b
        merge(c, (; glu_a=_scale_glu_values(c.glu_a, s), glu_b=_scale_glu_values(c.glu_b, s)))
    end
    return merge(ps, (; core)), merge(st, (; core=merge(st.core, (; fold_k=Int(k)))))
end

# Scale the VALUE rows of one GLU's parameters by `s`, never its gates. A GLU either has its own gate
# Dense (all of `linear` is values) or is fused: one Dense whose first half of rows are the values and
# whose second half are the gates. This holds for the fused `a`/`b` GLU and for each of the separate ones
# (`fused=false` with the default `fused_glu=true` gives fused GLUs of H values and H gates).
function _scale_glu_values(g::NamedTuple, s)
    if haskey(g, :gate) && !isempty(g.gate)
        return merge(g, (; linear=_scale_rows(g.linear, :, s)))
    end
    n = size(g.linear.weight, 1) ÷ 2
    return merge(g, (; linear=_scale_rows(g.linear, 1:n, s)))
end

# Scale rows `r` (along the output dimension) of a Dense's weight and bias by `s`, out of place.
_scale_rows(lin::NamedTuple, r, s) = map(a -> (a = copy(a); selectdim(a, 1, r) .*= s; a), lin)
