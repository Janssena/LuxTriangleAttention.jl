import Pkg

python_path = abspath(joinpath(@__DIR__, "..", "..", "python", "tri-attn", "bin", "python"))
ENV["PYTHON"] = python_path
Pkg.build("PyCall")

using PyCall

const torch = pyimport("torch")

_swap_batch_dim(x::AbstractVector) = x
_swap_batch_dim(x::AbstractArray{T, N}) where {T,N} = permutedims(x, (N, 2:N-1..., 1))

function to_py(x::AbstractArray{T}; swap_batch_dim=false, device="cpu") where T
    if T <: Integer
        py_dtype = T == Int32 ? torch.int32 : torch.int64
    else
        py_dtype = T == Float64 ? torch.float64 : (T == Float16 ? torch.float16 : torch.float32)
    end
    x_py = swap_batch_dim ? _swap_batch_dim(x) : x

    return torch.from_numpy(collect(x_py)).to(py_dtype).to(device).contiguous()
end

function to_jl(x::PyObject; device="cpu", swap_batch_dim=false)
    x_jl = device == "cpu" ? x.detach().cpu() : x.detach().gpu()
    x_jl = x_jl.numpy()
    return swap_batch_dim ? _swap_batch_dim(x_jl) : x_jl
end

copy_jl_ps_to_py!(py::PyObject, jl::AbstractArray; swap_batch_dim=false) = 
    py.detach().copy_(to_py(jl; swap_batch_dim))

function sync_dense!(py::PyObject, jl::NamedTuple)
    @assert py"hasattr"(py, "weight") "PyObject does not have weight attribute."
    @assert (py"hasattr"(py, "bias") && !isnothing(py.bias)) == (:bias in keys(jl)) "PyObject and NamedTuple have non-matching bias attributes."

    copy_jl_ps_to_py!(py.weight, jl.weight)
    if :bias in keys(jl) && (py"hasattr"(py, "bias") && !isnothing(py.bias))
        copy_jl_ps_to_py!(py.bias, jl.bias)
    end
end

function sync_layernorm!(py::PyObject, jl::NamedTuple)
    @assert py"hasattr"(py, "weight") "PyObject does not have weight attribute."
    @assert (py"hasattr"(py, "bias") && !isnothing(py.bias)) == (:bias in keys(jl)) "PyObject and NamedTuple have non-matching bias attributes."

    copy_jl_ps_to_py!(py.weight, vec(jl.scale))
    if :bias in keys(jl) && (py"hasattr"(py, "bias") && !isnothing(py.bias))
        copy_jl_ps_to_py!(py.bias, vec(jl.bias))
    end
end