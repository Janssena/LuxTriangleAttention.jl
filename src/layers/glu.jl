struct GatedLinearUnit{F<:StaticBool,Act,L,G} <: Lux.AbstractLuxContainerLayer{(:linear, :gate)}
    fused::F
    activation::Act
    linear::L
    gate::G
    out_chn::Int
end

"""
    GatedLinearUnit(in_chn => out_chn; fused=static(true), activation=Lux.sigmoid, use_bias=false)
    GatedLinearUnit((in_val, in_gate) => out_chn; ...)

Implements a Gated Linear Unit: `y = linear(x_linear) * activation(gate(x_gate))`.

## Arguments
- `in_out`: A `Pair` mapping input channels to output channels. Can also be `(in_val, in_gate) => out_chn`.

## Keyword Arguments
- `fused`: If `true`, `in_val` must equal `in_gate`, and a single `Dense` call is used for both projections.
- `activation`: Activation function to apply to the gate branch.
- `use_bias`: Whether to use bias in the linear projections.

## Inputs
- `x`: Input tensor with shape `[C, N, ...]` or a `Tuple` of tensors for dual inputs.

## Returns
- `y`: Output tensor with shape `[out_chn, N, ...]`.
- `st`: Updated state.
"""
function GatedLinearUnit(
    in_out::Pair; 
    fused::Union{Bool, StaticBool}=static(true), activation=Lux.sigmoid, use_bias=false
)
    in_chn, out_chn = in_out
    in_val, in_gate = in_chn isa Tuple ? in_chn : (in_chn, in_chn)
    fused_static = static(fused)

    if known(fused_static)
        @assert in_val == in_gate "Fused GatedLinearUnit requires val and gate to have same input dimension."
        linear = Lux.Dense(in_val => 2 * out_chn; use_bias)
        gate = Lux.NoOpLayer()
    else
        linear = Lux.Dense(in_val => out_chn; use_bias)
        gate = Lux.Dense(in_gate => out_chn; use_bias)
    end

    return GatedLinearUnit(fused_static, activation, linear, gate, out_chn)
end

GatedLinearUnit(in::Union{Tuple{Int,Int}, Int}, out::Int, args...; kwargs...) = 
    GatedLinearUnit(in => out, args...; kwargs...)

"""
    SwiGLU(in_chn => out_chn; kwargs...)

A variant of `GatedLinearUnit` using the `swish` activation function. 
Commonly used in Transformer architectures (e.g., LLaMA, Boltz-1).

## Arguments
- `in_out`: Input to Output mapping.

## Keyword Arguments
- `kwargs...`: Passed to `GatedLinearUnit`.
"""
SwiGLU(args...; kwargs...) = GatedLinearUnit(args...; activation=Lux.swish, kwargs...)

function (m::GatedLinearUnit)(x::AbstractArray, ps, st)
    return __glu_forward(m, x, ps, st)
end

function (m::GatedLinearUnit)(x::Tuple{<:AbstractArray, <:AbstractArray}, ps, st)
    return __glu_forward_dual(m, x[1], x[2], ps, st)
end

function __glu_forward(m::GatedLinearUnit{True}, x::AbstractArray{T,N}, ps, st) where {T,N}
    v_and_g, linear = m.linear(x, ps.linear, st.linear)
    v = view(v_and_g, 1:m.out_chn, ntuple(_ -> Colon(), N-1)...)
    g = view(v_and_g, (m.out_chn+1):(2*m.out_chn), ntuple(_ -> Colon(), N-1)...)
    y = @. v * m.activation(g)
    return y, (; linear, )
end

function __glu_forward(m::GatedLinearUnit{False}, x, ps, st)
    v, linear = m.linear(x, ps.linear, st.linear)
    g, gate = m.gate(x, ps.gate, st.gate)
    y = @. v * m.activation(g)
    return y, (; linear, gate)
end

function __glu_forward_dual(::GatedLinearUnit{True}, args...)
    error("Fused GatedLinearUnit does not support dual inputs.")
end

function __glu_forward_dual(m::GatedLinearUnit{False}, x_linear, x_gate, ps, st)
    v, linear = m.linear(x_linear, ps.linear, st.linear)
    g, gate = m.gate(x_gate, ps.gate, st.gate)
    y = @. v * m.activation(g)
    return y, (; linear, gate)
end