struct TriAttnCore{Q,K,V,G,O,T} <: Lux.AbstractLuxContainerLayer{(:linear_q,:linear_k,:linear_v,:gate,:linear_out)}
    linear_q::Q
    linear_k::K
    linear_v::V 
    gate::G
    linear_out::O
    chn_hidden::Int # D, Feature dimension per head
    num_heads::Int # H, Number of attention heads
    inf::T
end

TriAttnCore(chn_in::Int, chn_hidden::Int, num_heads::Int; kwargs...) = 
    TriAttnCore(chn_in, chn_in, chn_in, chn_hidden, num_heads; kwargs...)

function TriAttnCore(
    chn_q::Int, chn_k::Int, chn_v::Int, chn_hidden::Int, num_heads::Int;
    inf::Real=_safe_inf(Float32), gate::Bool=true, qkv_use_bias::Bool=true, 
    gate_use_bias::Bool=true, out_use_bias::Bool=true
) 
    return TriAttnCore(
        Lux.Dense(chn_q => chn_hidden * num_heads; use_bias=qkv_use_bias),
        Lux.Dense(chn_k => chn_hidden * num_heads; use_bias=qkv_use_bias),
        Lux.Dense(chn_v => chn_hidden * num_heads; use_bias=qkv_use_bias),
        gate ? Lux.Dense(chn_q => chn_hidden * num_heads, Lux.sigmoid; use_bias=gate_use_bias) : Lux.NoOpLayer(),
        Lux.Dense(chn_hidden * num_heads => chn_q; use_bias=out_use_bias),
        chn_hidden,
        num_heads,
        inf
    )
end

Lux.initialstates(rng::Random.AbstractRNG, attn::TriAttnCore) = (
    linear_q = Lux.initialstates(rng, attn.linear_q),
    linear_k = Lux.initialstates(rng, attn.linear_k),
    linear_v = Lux.initialstates(rng, attn.linear_v),
    gate = Lux.initialstates(rng, attn.gate),
    linear_out = Lux.initialstates(rng, attn.linear_out),
    inf = attn.inf
)

(m::TriAttnCore)(inputs::NamedTuple, ps, st) = _tri_attn_core_forward(
    m,
    inputs.x, 
    inputs.bias,
    get(inputs, :mask, nothing), 
    ps, st
)

(m::TriAttnCore)(x, bias, ps, st) = m(x, bias, nothing, ps, st)
(m::TriAttnCore)(x, bias, mask, ps, st) = _tri_attn_core_forward(m, x, bias, mask, ps, st)

function _tri_attn_core_forward(m::TriAttnCore, x, bias, mask, ps, st)
    _, N, _, B = size(x) # [C, N, N, B]
    qkv_dims = (m.chn_hidden, m.num_heads, N, N, B)

    _q, linear_q = m.linear_q(x, ps.linear_q, st.linear_q) # [DxH, N, N, B]
    _k, linear_k = m.linear_k(x, ps.linear_k, st.linear_k) # [DxH, N, N, B]
    _v, linear_v = m.linear_v(x, ps.linear_v, st.linear_v) # [DxH, N, N, B]
    q, k, v = map(Base.Fix2(reshape, qkv_dims), (_q, _k, _v)) # [D, H, N, N, B]

    out = triangle_attention(q, k, v, bias, mask; neg_inf = -st.inf) # [D, H, N, N, B]
    out = reshape(out, (m.chn_hidden * m.num_heads, N, N, B)) # [DxH, N, N, B]
    out, gate = __tri_attn_gate_maybe(m.gate, out, x, ps.gate, st.gate) # [DxH, N, N, B]

    y, linear_out = m.linear_out(out, ps.linear_out, st.linear_out) # [C, N, N, B]

    return y, merge(st, (; linear_q, linear_k, linear_v, gate, linear_out))
end

__tri_attn_gate_maybe(::Lux.NoOpLayer, out, x, ps, st) = out, st
function __tri_attn_gate_maybe(l, out, x, ps, st) 
    g, st_gate = l(x, ps, st) # [DxH, N, N, B]
    return (@. out * g), st_gate
end
