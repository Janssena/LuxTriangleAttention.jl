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