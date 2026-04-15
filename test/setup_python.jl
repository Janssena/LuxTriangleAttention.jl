import Pkg

python_path = abspath(joinpath(@__DIR__, "..", "python", "tri-attn", "bin", "python"))
ENV["PYTHON"] = python_path
Pkg.build("PyCall")

using PyCall

const torch = pyimport("torch")

py"""
import numpy as np
import torch
import math
from einops import rearrange, einsum

def neg_inf(dtype) -> float:
    return torch.finfo(dtype).min

def triangle_attention_simple(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    q = q * (q.shape[-1] ** -0.5)

    qk_dot = einsum(q, k, "... h i j d, ... h i k d -> ... h i j k")

    #                                                                                 i  j k
    qk_dot_bias = qk_dot + rearrange(bias, "... h n m -> ... h () n m")
    # This modifies qk_dot_bias in place.
    qk_dot_bias.masked_fill_(
        #                                              h  i j  k
        rearrange(mask, "... n m -> ... () n () m"),
        neg_inf(q.dtype),
    )
    a_ijk = torch.softmax(qk_dot_bias, dim=-1)

    o_ij = einsum(a_ijk, v, "... h i j k, ... h i k d -> ... h i j d")

    return o_ij

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
    a += bias

    # CORRECTED POLARITY AND NONE-CHECK
    if mask is not None:
        # 1.0 means valid (0 bias), 0.0 means invalid (-inf bias)
        mask_bias = neg_inf(q.dtype) * (1.0 - mask.to(q.dtype))
        mask_bias = rearrange(mask_bias, "b i j -> b i () () j")
        a += mask_bias

    a = torch.softmax(a, dim=-1)
    a_v = torch.matmul(a, v)

    o = rearrange(a_v, "b i h j d -> b h i j d")

    return o
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