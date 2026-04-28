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

## Inputs
- `x`: Input tensor with shape `[C, N, N, B]`.
- `mask`: Optional mask with shape `[N, N, B]`.

## Returns
- `y`: Output tensor with shape `[chn_hidden, N, N, B]`.
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
    layernorm_eps=1f-5, use_bias=false, kwargs...
)   
    fused_static = static(fused)
    
    if known(fused_static)
        # Fuses p_a, p_b, g_a, g_b into a single operation (Dense to 2H/4H depending on chn_hidden)
        glu_ab = GatedLinearUnit(chn_in => 2 * chn_hidden; fused=fused_glu, use_bias)
        glu_a = Lux.NoOpLayer()
        glu_b = Lux.NoOpLayer()
    else
        glu_ab = Lux.NoOpLayer()
        glu_a = GatedLinearUnit(chn_in => chn_hidden; fused=fused_glu, use_bias)
        glu_b = GatedLinearUnit(chn_in => chn_hidden; fused=fused_glu, use_bias)
    end

    layer_norm_out = Lux.LayerNorm((chn_hidden, 1, 1); dims=1, epsilon=layernorm_eps)
    
    # Output Gate: Dual input (chn_hidden from Y, chn_in from X)
    glu_out = GatedLinearUnit((chn_hidden, chn_in) => chn_in; fused=false, use_bias)

    return TriMulCore(
        fused_static,
        static(is_outgoing),
        glu_ab, glu_a, glu_b,
        layer_norm_out,
        glu_out,
        chn_hidden
    )
end

function (m::TriMulCore)(x_norm, mask, ps, st)
    # [C, N, N, B] -> [H, N, N, B]
    a, b, st_proj = _prep_ab(m, x_norm, ps, st)
    
    # Apply Mask [N, N, B]
    _apply_tri_mask!(a, b, mask)

    # 3. Batched Matmul [H, N, N, B]
    # Outgoing (Alg 11): Σ_k A[i, k] * B[j, k] -> Contract Dim 3 (j/k)
    # Incoming (Alg 12): Σ_k A[k, i] * B[k, j] -> Contract Dim 2 (i/k)
    cdim = known(m.is_outgoing) ? 3 : 2
    y_raw = Lux.batched_matmul(
        a, b;
        lhs_contracting_dim=cdim,
        rhs_contracting_dim=cdim,
        lhs_batching_dims=(1, 4),
        rhs_batching_dims=(1, 4)
    ) # Result is [N, N, H, B]
    
    y = permutedims(y_raw, (3, 1, 2, 4))

    y_norm, layer_norm_out = m.layer_norm_out(y, ps.layer_norm_out, st.layer_norm_out)
    out, glu_out = m.glu_out((y_norm, x_norm), ps.glu_out, st.glu_out)

    return out, merge(st_proj, (; layer_norm_out, glu_out))
end

"""
    _prep_ab(m::TriMulCore, x, ps, st)

Helper to project the input into the `A` and `B` representations for multiplication.
If `fused=true`, uses a single projection followed by split.

## Arguments
- `m`: `TriMulCore` layer.
- `x`: Input tensor.

## Returns
- `(a, b)`: Projected representations.
- `st`: Updated state.
"""
function _prep_ab(m::TriMulCore{True}, x::AbstractArray{T,N}, ps, st) where {T,N}
    ab, st_glu = m.glu_ab(x, ps.glu_ab, st.glu_ab)
    
    H = m.chn_hidden
    a = view(ab, 1:H, ntuple(_ -> Colon(), N-1)...)
    b = view(ab, (H+1):(2*H), ntuple(_ -> Colon(), N-1)...)

    return a, b, (glu_ab=st_glu, glu_a=st.glu_a, glu_b=st.glu_b)
end

# Split projections: a and b calculated separately
function _prep_ab(m::TriMulCore{False}, x, ps, st)
    a, st_a = m.glu_a(x, ps.glu_a, st.glu_a)
    b, st_b = m.glu_b(x, ps.glu_b, st.glu_b)

    return a, b, (glu_ab=st.glu_ab, glu_a=st_a, glu_b=st_b)
end


"""
    _apply_tri_mask!(a, b, mask)

Applies the triangle mask to the projected `A` and `B` representations.

## Arguments
- `a`, `b`: Projected representations.
- `mask`: Input mask.
"""
_apply_tri_mask!(a, b, mask::Nothing) = nothing
function _apply_tri_mask!(a::AbstractArray{T}, b, mask::AbstractArray{Bool}) where T
    mask_reshaped = reshape(mask, 1, size(mask)...)
    @. a = ifelse(mask_reshaped, a, zero(T))
    @. b = ifelse(mask_reshaped, b, zero(T))
    return nothing
end

function _apply_tri_mask!(a::AbstractArray{T}, b, mask::AbstractArray{<:Real}) where T
    mask_reshaped = reshape(mask, 1, size(mask)...)
    @. a = a * mask_reshaped
    @. b = b * mask_reshaped
    return nothing
end