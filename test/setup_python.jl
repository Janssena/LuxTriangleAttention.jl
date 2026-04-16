import Pkg

python_path = abspath(joinpath(@__DIR__, "..", "python", "tri-attn", "bin", "python"))
ENV["PYTHON"] = python_path
Pkg.build("PyCall")

using PyCall

const torch = pyimport("torch")

py"""
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import math
from einops import rearrange, einsum

def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:

    sm_scale = q.shape[-1] ** -0.5

    q = rearrange(q, "b h i j d -> b i h j d")
    k = rearrange(k, "b h i j d -> b i h d j")
    v = rearrange(v, "b h i j d -> b i h j d")

    a = torch.matmul(q, k) * sm_scale  

    bias = rearrange(bias, "b h i j -> b () h i j")
    a.add_(bias) 

    if mask is not None:
        mask = rearrange(mask, "b i j -> b i () () j")
        # Assuming mask is 1.0 for valid, 0.0 for invalid
        a.masked_fill_(mask == 0, torch.finfo(q.dtype).min)

    a = torch.softmax(a, dim=-1)
    o = torch.matmul(a, v)

    return rearrange(o, "b i h j d -> b h i j d")
"""

_swap_batch_dim(x::AbstractVector) = x
_swap_batch_dim(x::AbstractArray{T, N}) where {T,N} = permutedims(x, (N, 2:N-1..., 1))

function to_py(x::AbstractArray{T}; swap_batch_dim=true, device="cpu") where T
    if T <: Integer
        py_dtype = T == Int32 ? torch.int32 : torch.int64
    else
        py_dtype = T == Float64 ? torch.float64 : (T == Float16 ? torch.float16 : torch.float32)
    end
    x_py = swap_batch_dim ? _swap_batch_dim(x) : x

    return torch.from_numpy(collect(x_py)).to(py_dtype).to(device).contiguous()
end

function to_jl(x::PyObject; device="cpu", swap_batch_dim=true)
    x_jl = device == "cpu" ? x.detach().cpu() : x.detach().gpu()
    x_jl = x_jl.numpy()
    return swap_batch_dim ? _swap_batch_dim(x_jl) : x_jl
end