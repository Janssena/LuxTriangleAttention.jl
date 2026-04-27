import Pkg

python_path = abspath(joinpath(@__DIR__, "..", "..", "python", "tri-attn", "bin", "python"))
ENV["PYTHON"] = python_path
Pkg.build("PyCall")

using PyCall

const torch = pyimport("torch")
convert_types(::Type{Float64}) = Lux.f64

py_dtype(::Type{Float64}) = torch.float64
py_dtype(::Type{Float32}) = torch.float32
py_dtype(::Type{Float16}) = torch.float16
py_dtype(::Type{Int32}) = torch.int32
py_dtype(::Type{Int64}) = torch.int64
py_dtype(::Type{Bool}) = torch.bool

_swap_batch_dim(x::AbstractVector) = x
_swap_batch_dim(x::AbstractArray{T,N}) where {T,N} = permutedims(x, (N, 2:N-1..., 1))

function to_py(x::AbstractArray{T}; swap_batch_dim=false, device="cpu") where T
    x_py = swap_batch_dim ? _swap_batch_dim(x) : x

    return torch.from_numpy(collect(x_py)).to(py_dtype(T)).to(device)
end

function to_jl(x::PyObject; device="cpu", swap_batch_dim=false)
    x_jl = device == "cpu" ? x.detach().cpu() : x.detach().gpu()
    x_jl = x_jl.numpy()
    return swap_batch_dim ? _swap_batch_dim(x_jl) : x_jl
end

function copy_jl_ps_to_py!(py::PyObject, jl::AbstractArray{T}; swap_batch_dim=false) where T 
    @assert py"type"(py) == torch.nn.Parameter "Passed PyObject is not a torch.nn.Parameter"
    @assert py.shape == size(jl) "Shape of py $(py.shape) and jl $(size(jl)) do not match."
    py.data = to_py(jl; swap_batch_dim)
    
    return nothing
end

function sync_dense!(py::PyObject, jl::NamedTuple)
    @assert py"hasattr"(py, "weight") "PyObject does not have weight attribute."
    @assert (py"hasattr"(py, "bias") && !isnothing(py.bias)) == (:bias ∈ keys(jl)) "PyObject and NamedTuple have non-matching bias attributes."

    copy_jl_ps_to_py!(py.weight, jl.weight)
    if :bias ∈ keys(jl) && (py"hasattr"(py, "bias") && !isnothing(py.bias))
        copy_jl_ps_to_py!(py.bias, jl.bias)
    end

    return nothing
end

function sync_layernorm!(py::PyObject, jl::NamedTuple)
    @assert py"hasattr"(py, "weight") "PyObject does not have weight attribute."
    @assert (py"hasattr"(py, "bias") && !isnothing(py.bias)) == (:bias ∈ keys(jl)) "PyObject and NamedTuple have non-matching bias attributes."

    copy_jl_ps_to_py!(py.weight, vec(jl.scale))
    if :bias ∈ keys(jl) && (py"hasattr"(py, "bias") && !isnothing(py.bias))
        copy_jl_ps_to_py!(py.bias, vec(jl.bias))
    end

    return nothing
end


function sync_glu!(py::PyObject, jl::NamedTuple; ref=(linear = :linear_z, gate = :linear_g))
    @assert py"hasattr"(py, ref.linear) "PyObject does not have the referenced linear attribute ($(ref.linear))."
    @assert py"hasattr"(py, ref.gate) "PyObject does not have the referenced gate attribute ($(ref.gate))." 
    @assert (py"hasattr"(py[ref.linear], "bias") && !isnothing(py[ref.linear].bias)) == (:bias ∈ keys(jl.linear)) "PyObject linear and NamedTuple have non-matching bias attributes."
    gate_should_have_bias_keys = isempty(jl.gate) ? (:bias ∈ keys(jl.linear)) : (:bias ∈ keys(jl.gate))
    @assert (py"hasattr"(py[ref.gate], "bias") && !isnothing(py[ref.gate].bias)) == gate_should_have_bias_keys "PyObject gate and NamedTuple have non-matching bias attributes."

    jl_unfused = _unfuse(jl)
    sync_dense!(py[ref.linear], jl_unfused.linear)
    sync_dense!(py[ref.gate], jl_unfused.gate)

    return nothing
end

function _unfuse(jl::NamedTuple{(:linear, :gate)})
    if !isempty(jl.gate)
        return jl
    end

    w = jl.linear.weight
    chn = size(w, 1) ÷ 2

    ps = (
        linear = (weight = view(w, 1:chn, :), ),
        gate = (weight = view(w, chn+1:2*chn, :), ),
    )

    if :bias ∈ keys(jl.linear)
        b = jl.linear.bias
        ps = (
            linear = merge(ps.linear, (bias = view(b, 1:chn), )),
            gate = merge(ps.gate, (bias = view(b, chn+1:2*chn), )),
        )
    end

    return ps
end

function sync_qkv!(py::PyObject, ps::NamedTuple)
    if :weight ∈ keys(ps)
        # Julia qkv is fully fused: [3*C_hidden*H, C_in]
        # Python has linear_q, linear_k, linear_v
        # Split Julia weight
        W = ps.weight
        C_h = size(W, 1) ÷ 3

        copy_jl_ps_to_py!(py.linear_q.weight, view(W, 1:C_h, :))
        copy_jl_ps_to_py!(py.linear_k.weight, view(W, C_h+1:2*C_h, :))
        copy_jl_ps_to_py!(py.linear_v.weight, view(W, 2*C_h+1:3*C_h, :))

        if :bias ∈ keys(ps)
            B = ps.bias
            copy_jl_ps_to_py!(py.linear_q.bias, view(B, 1:C_h))
            copy_jl_ps_to_py!(py.linear_k.bias, view(B, C_h+1:2*C_h))
            copy_jl_ps_to_py!(py.linear_v.bias, view(B, 2*C_h+1:3*C_h))
        end
    elseif :q in keys(ps) && :kv in keys(ps)
        # Version where only kv is fused.
        sync_dense!(py.linear_q, ps.q)
        W_kv = ps.kv.weight
        C_h = size(W_kv, 1) ÷ 3
        copy_jl_ps_to_py!(py.linear_k.weight, view(W_kv, 1:C_h, :))
        copy_jl_ps_to_py!(py.linear_v.weight, view(W_kv, C_h+1:2*C_h, :))
        if :bias ∈ keys(ps.kv)
            B = ps.kv.bias
            copy_jl_ps_to_py!(py.linear_k.bias, view(B, 1:C_h))
            copy_jl_ps_to_py!(py.linear_v.bias, view(B, C_h+1:2*C_h))
        end
    else # unfused
        sync_dense!(py.linear_q, ps.q)
        sync_dense!(py.linear_k, ps.k)
        sync_dense!(py.linear_v, ps.v)
    end

    return nothing
end

function sync_attention!(py::PyObject, ps::NamedTuple)
    sync_qkv!(py, ps.qkv)
    sync_dense!(py.linear_o, ps.out)

    if !isempty(ps.gate)
        sync_dense!(py.linear_g, ps.gate)
    end

    return nothing
end

sync_af3_triangle_attention!(
    args...;
    ref=(layer_norm=:layer_norm, linear=:linear_z, mha=:mha)
) = sync_triangle_attention!(args...; ref)

function sync_triangle_attention!(
    py::PyObject,
    ps::NamedTuple;
    ref=(layer_norm=:layer_norm, linear=:linear, mha=:mha)
)
    sync_layernorm!(py[ref.layer_norm], ps.layer_norm)
    sync_dense!(py[ref.linear], ps.linear)
    sync_attention!(py[ref.mha], ps.mha)

    return nothing
end