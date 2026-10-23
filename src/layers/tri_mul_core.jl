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

    a, b = _ab_planes(m, x_tok, mask, ps, st, N, B)           # [N, N, B·H] each, slot = b + B·(h-1)
    X = _contract(m.is_outgoing, a, b)                        # [N, N, B·H]
    y_tok = reshape(X, N * N * B, H)                          # token-major: slot order = token order

    y_norm, _ = m.layer_norm_out(y_tok, ps.layer_norm_out, st.layer_norm_out)
    out_tok, _ = m.glu_out((y_norm, x_tok), ps.glu_out, st.glu_out)          # [N²B, C]

    return reshape(permutedims(out_tok, (2, 1)), C, N, N, B), st
end

# Outgoing (Alg 11): X[i, j] = Σ_k a[i, k] b[j, k]  →  A · Bᵀ per plane
_contract(::True, a, b) = Lux.batched_matmul(a, Lux.batched_transpose(b))
# Incoming (Alg 12): X[i, j] = Σ_k a[k, i] b[k, j]  →  Aᵀ · B per plane
_contract(::False, a, b) = Lux.batched_matmul(Lux.batched_transpose(a), b)

"""
    _ab_planes(m::TriMulCore, x_tok, mask, ps, st, N, B) -> (a, b)

Project the normalised input into the gated, masked `a` and `b` representations, laid out as
`[N, N, B·H]` planes (batch slot `b + B·(h-1)`), ready for the batched contraction.

With `fused = true` one GLU produces both (`[N²·B, 2H]`, `a` in the first `H` columns); the planes
are contiguous views of it. With `fused = false`, `glu_a` and `glu_b` produce one each.
"""
function _ab_planes(m::TriMulCore{True}, x_tok, mask, ps, st, N, B)
    H = m.chn_hidden
    ab = _apply_tri_mask(first(m.glu_ab(x_tok, ps.glu_ab, st.glu_ab)), mask)   # [N²B, 2H]
    planes = reshape(ab, N, N, 2 * B * H)
    return view(planes, :, :, 1:(B * H)), view(planes, :, :, (B * H + 1):(2 * B * H))
end

function _ab_planes(m::TriMulCore{False}, x_tok, mask, ps, st, N, B)
    H = m.chn_hidden
    a = _apply_tri_mask(first(m.glu_a(x_tok, ps.glu_a, st.glu_a)), mask)
    b = _apply_tri_mask(first(m.glu_b(x_tok, ps.glu_b, st.glu_b)), mask)
    return reshape(a, N, N, B * H), reshape(b, N, N, B * H)
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
