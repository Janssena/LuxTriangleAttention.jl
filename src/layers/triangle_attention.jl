"""
    TriangleAttention(chn_in, head_dim, num_heads; is_starting=true, kwargs...)

Implementation of Triangle Attention (Algorithm 13 & 14 from AlphaFold2).
Updates the pair representation by performing axial attention with a triangle bias.

## Arguments
- `chn_in`: Input channels.
- `head_dim`: Dimension of each attention head.
- `num_heads`: Number of attention heads.

## Keyword Arguments
- `is_starting`: If `true`, performs "Starting Node" attention (Algorithm 13). 
  If `false`, performs "Ending Node" attention (Algorithm 14).
- `use_bias`: Boolean or NamedTuple for internal layer biases.
- `layernorm_eps`: Epsilon for LayerNorm.

## Inputs
- `x`: Input tensor with shape `[C, Ni, Nj, B]`.
- `mask`: Optional mask with shape `[Ni, Nj, B]`.

## Returns
- `y`: Output tensor with shape `[C, Ni, Nj, B]`.
- `st`: Updated state.

## Example
```julia
using Lux, LuxTriangleAttention, Random

model = TriangleAttention(64, 32, 4; is_starting=true)
ps, st = Lux.setup(Random.default_rng(), model)

x = randn(Float32, 64, 32, 32, 1)
mask = rand(Bool, 32, 32, 1) # Pairwise boolean mask
raw_bias = randn(Float32, 4, 32, 32, 1) # Raw triangle bias

# Prepare bias for internal mechanism
bias = prep_triangle_bias(raw_bias)

# Pass via NamedTuple
y, st = model((; x, mask, bias), ps, st)
```
"""
struct TriangleAttention{P<:StaticBool,LN,B,MHA} <: Lux.AbstractLuxContainerLayer{(:layer_norm,:linear,:mha)}
    permute::P # if True, attends over Nj else Ni
    layer_norm::LN
    linear::B
    mha::MHA
end

function TriangleAttention(
    chn_in::Int, head_dim::Int, num_heads::Int;
    is_starting::Union{Bool, StaticBool}=static(true), use_bias=true, layernorm_eps=1f-5, kwargs...
) 
    use_bias = resolve_defaults(use_bias, (:layer_norm, :linear, :mha))
    shape = (chn_in, 1, 1)
    layer_norm = if use_bias.layer_norm
        Lux.LayerNorm(shape; dims=1, epsilon=layernorm_eps)
    else
        LayerNormNoBias(shape; dims=1, epsilon=layernorm_eps)
    end
    linear = Lux.Dense(chn_in, num_heads; use_bias=use_bias.linear)
    mha = Attention(chn_in, head_dim, num_heads; use_bias=use_bias.mha, kwargs...)

    return TriangleAttention(
        static(is_starting), # starting: attend over Nj, ending: attend over Ni
        layer_norm,
        linear,
        mha
    )
end

(m::TriangleAttention)(inputs::NamedTuple, ps, st) = _triangle_attention_forward(
    m,
    inputs.x, 
    get(inputs, :mask, nothing), 
    ps, st
)
(m::TriangleAttention)(x::AbstractArray, ps, st) = m(x, nothing, ps, st)

function (m::TriangleAttention)(x::AbstractArray, mask, ps, st)
    x, mask = permute_ij_maybe(m.permute, x, mask)
    x_norm, layer_norm = m.layer_norm(x, ps.layer_norm, st.layer_norm) # [C, Ni, Nj, B]
    
    bias, linear = m.linear(x_norm, ps.linear, st.linear) # [H, Ni, Nj, B]
    bias = prep_triangle_bias(bias) # [1, Ni, H, Nj, B] 

    y, mha = m.mha(x_norm, bias, mask, ps.mha, st.mha)

    y = permute_ij_maybe(m.permute, y)

    return y, merge(st, (; layer_norm, linear, mha))
end

"""
    prep_triangle_bias(bias)

Reshapes a bias tensor into the format expected by `Attention` for triangle attention.

## Arguments
- `bias`: Tensor of shape `[H, Ni, Nj, B]`.

## Returns
- Tensor of shape `[Ni, Nj, H, 1, B]`.
"""
prep_triangle_bias(::Nothing) = nothing
function prep_triangle_bias(bias::AbstractArray{T,4}) where T
    H, Ni, Nj, B = size(bias)
    return reshape(permutedims(bias, (2, 3, 1, 4)), Ni, Nj, H, 1, B)
end

"""
    permute_ij_maybe(is_starting, x, [mask])

Swaps the `i` and `j` dimensions of the input tensor(s) if `is_starting` is `false`.

## Arguments
- `is_starting`: `StaticBool`. If `false`, triggers permutation.
- `x`: Input tensor.
- `mask`: Optional input mask.

## Returns
- Permuted (or original) tensor(s).
"""
permute_ij_maybe(::False, x) = x

# starting = true -> permute = false -> no permutation
permute_ij_maybe(::False, x, mask) = x, mask

# starting = false -> permute = true ->  swap the i and j dimensions
permute_ij_maybe(::True, x) = ending_permute(x)
permute_ij_maybe(::True, args::Vararg) = map(ending_permute, args)

"""
    ending_permute(x)

Utility to swap the `i` and `j` dimensions (2nd and 3rd spatial dimensions) 
of the input tensor.

## Arguments
- `x`: Input tensor.

## Returns
- Tensor with swapped `i` and `j` dimensions.
"""
ending_permute(::Nothing) = nothing
ending_permute(x::AbstractArray{T,5}) where T = permutedims(x, (1, 2, 4, 3, 5)) # swap i and j in [D, H, N, N, B]
ending_permute(x::AbstractArray{T,4}) where T = permutedims(x, (1, 3, 2, 4)) # swap i and j in [C, N, N, B]
ending_permute(mask::AbstractArray{T,3}) where T = permutedims(mask, (2, 1, 3)) # swap i and j in [N, N, B]