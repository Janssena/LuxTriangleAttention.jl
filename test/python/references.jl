# The Python side of every parity unit, `PYTHON_REFERENCES[(component, model)]`: what the sweeps in test/support/parity.jl
# run to record the references (support/fixtures.jl). Loaded only when generating them; needs PyCall and torch (setup.jl).
#
# attention:                make_py(chn, head_dim, num_heads) builds the layer and call_py(py_layer, x_py, mask,
#                           mask_additive_py, bias_py, T) calls it, as the references take the mask and the pair bias
#                           in different ways.
# triangle_attention:       make_py(chn_in, chn_hidden, num_heads, is_starting, inf) builds the layer and
#                           sync_py(py_layer, ps) copies the parameters into it.
# triangle_multiplication:  make_py(is_outgoing, chn_in, chn_hidden), sync_py(py_layer, ps) and mask_py(mask, T, N, B),
#                           the mask in the form the reference takes.

include("alphafold2.jl")
include("alphafold3.jl")
include("boltz2.jl")

# The a and b projections of a `TriMulCore`'s parameters as two unfused GLUs each, whichever way the layer holds them
# (`fused`: one GLU for both, a in the first `H` rows; `fused_glu`: values and gates in one Dense).
function trimul_ab(core::NamedTuple)
    if isempty(core.glu_ab)
        return _unfuse(core.glu_a), _unfuse(core.glu_b)
    end
    u = _unfuse(core.glu_ab)
    H = size(u.linear.weight, 1) ÷ 2
    rows(r) = (linear=map(p -> selectdim(p, 1, r), u.linear), gate=map(p -> selectdim(p, 1, r), u.gate))
    return rows(1:H), rows(H+1:2H)
end

"""
    sync_trimul!(py, ps; norm_in, norm_out, out, a=nothing, b=nothing, ab=nothing)

Copy a `TriangleMultiplication`'s parameters, in whichever of its four layouts, into a Python layer. The names are
the Python layer's attributes: for the layer norms, `out` for the output GLU as `(linear, gate)`, and either `a` and `b`
or, when Python holds both projections in one Dense with `a` first, `ab`.
"""
function sync_trimul!(py, ps::NamedTuple; norm_in, norm_out, out, a=nothing, b=nothing, ab=nothing)
    sync_layernorm!(py[norm_in], ps.layer_norm)
    ua, ub = trimul_ab(ps.core)
    if ab === nothing
        sync_glu!(py, ua; ref=a)
        sync_glu!(py, ub; ref=b)
    else
        both(x, y) = map((p, q) -> cat(p, q; dims=1), x, y)
        sync_glu!(py, (linear=both(ua.linear, ub.linear), gate=both(ua.gate, ub.gate)); ref=ab)
    end
    sync_layernorm!(py[norm_out], ps.core.layer_norm_out)
    sync_glu!(py, ps.core.glu_out; ref=out)
    return nothing
end

const PYTHON_REFERENCES = Dict{Tuple{String,String},NamedTuple}()

# ── Attention ─────────────────────────────────────────────────────────────────────────────────────

# AlphaFold2 and AlphaFold3 take the mask and the pair bias as two additive biases.
PYTHON_REFERENCES["attention", "af2"] = (
    make_py=(c, hd, H) -> py"AF2Attention"(c, c, c, hd, H),
    call_py=(py_layer, x_py, mask, mask_add_py, bias_py, T) -> py_layer(x_py, x_py; biases=[mask_add_py, bias_py]))

PYTHON_REFERENCES["attention", "af3"] = (
    make_py=(c, hd, H) -> py"AF3Attention"(c, c, c, hd, H),
    call_py=(py_layer, x_py, mask, mask_add_py, bias_py, T) -> py_layer(x_py, x_py; biases=[mask_add_py, bias_py]))

# Boltz-2 takes the pair bias and a 0/1 mask as separate arguments.
PYTHON_REFERENCES["attention", "boltz2"] = (
    make_py=(c, hd, H) -> py"Boltz2Attention"(c, c, c, hd, H),
    call_py=(py_layer, x_py, mask, mask_add_py, bias_py, T) -> py_layer(x_py, x_py, bias_py,
        mask_add_py,
        to_py(ndims(mask) == 2 ? mask : permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T))))

# ── TriangleAttention ─────────────────────────────────────────────────────────────────────────────

PYTHON_REFERENCES["triangle_attention", "af2"] = (
    make_py=(c, hd, H, starting, inf) -> py"AF2TriangleAttention"(c, hd, H, starting=starting, inf=inf),
    sync_py=sync_triangle_attention!)

PYTHON_REFERENCES["triangle_attention", "af3"] = (
    make_py=(c, hd, H, starting, inf) -> py"AF3TriangleAttention"(c, hd, H, starting=starting, inf=inf),
    sync_py=sync_af3_triangle_attention!)

PYTHON_REFERENCES["triangle_attention", "boltz2"] = (
    make_py=(c, hd, H, starting, inf) -> py"Boltz2TriangleAttention"(c, hd, H, starting=starting, inf=inf),
    sync_py=sync_triangle_attention!)

# ── TriangleMultiplication ────────────────────────────────────────────────────────────────────────

# AlphaFold2 projects `a` and `b` separately, with a bias on every layer; its module takes no mask when there is none.
PYTHON_REFERENCES["triangle_multiplication", "af2"] = (
    make_py=(out, c, h) -> py"AF2TriangleMultiplicativeUpdate"(c, h, _outgoing=out),
    sync_py=(py, ps) -> sync_trimul!(py, ps; norm_in=:layer_norm_in, norm_out=:layer_norm_out,
        a=(linear=:linear_a_p, gate=:linear_a_g), b=(linear=:linear_b_p, gate=:linear_b_g),
        out=(linear=:linear_z, gate=:linear_g)),
    mask_py=(mask, T, N, B) -> isnothing(mask) ? nothing :
        to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T)))

# AlphaFold3 projects `a` and `b` separately, with a bias on every layer, and its module takes a mask always, so no mask
# is a mask of ones.
PYTHON_REFERENCES["triangle_multiplication", "af3"] = (
    make_py=(out, c, h) -> (out ? py"AF3TriangleMultiplicationOutgoing" : py"AF3TriangleMultiplicationIncoming")(c, h),
    sync_py=(py, ps) -> sync_trimul!(py, ps; norm_in=:layer_norm_in, norm_out=:layer_norm_out,
        a=(linear=:linear_a_p, gate=:linear_a_g), b=(linear=:linear_b_p, gate=:linear_b_g),
        out=(linear=:linear_z, gate=:linear_g)),
    mask_py=(mask, T, N, B) -> isnothing(mask) ? to_py(ones(T, N, N, B); swap_batch_dim=true) :
        to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T)))

# Boltz-2 projects `a` and `b` in one Dense (`a` first), without biases, at the full width of the input, and its module
# takes a mask always.
PYTHON_REFERENCES["triangle_multiplication", "boltz2"] = (
    make_py=(out, c, h) -> (out ? py"Boltz2TriangleMultiplicationOutgoing" : py"Boltz2TriangleMultiplicationIncoming")(c),
    sync_py=(py, ps) -> sync_trimul!(py, ps; norm_in=:norm_in, norm_out=:norm_out,
        ab=(linear=:p_in, gate=:g_in), out=(linear=:p_out, gate=:g_out)),
    mask_py=(mask, T, N, B) -> isnothing(mask) ? to_py(ones(T, N, N, B); swap_batch_dim=true) :
        to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T)))
