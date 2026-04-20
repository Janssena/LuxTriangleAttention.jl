py"""
import math
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm
from typing import List, Optional, Tuple

def af3_permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    num_first_dims = len(tensor.shape)-len(inds)
    first_inds = list(range(num_first_dims))
    return tensor.permute(first_inds + [num_first_dims + i for i in inds])

def af3_flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


class AF3Attention(nn.Module):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        super().__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, bias=False
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, bias=False
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = af3_flatten_final_dims(o, 2)
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=True)

        attn_dtype = torch.float32 if use_high_precision else q.dtype
        with torch.amp.autocast("cuda", dtype=attn_dtype, enabled=False):
            scores = torch.einsum("...qc, ...kc->...qk", q, k)
            for b in biases:
                scores += b
            scores = nn.functional.softmax(scores, dim=-1)

        attention = torch.einsum("...qk, ...kc->...qc", scores.to(dtype=v.dtype), v)

        o = attention.transpose(-2, -3)
        o = self._wrap_up(o, q_x)

        return o


class AF3TriangleAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ) -> None:
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)
        self.linear_z = Linear(c_in, self.no_heads, bias=False)

        self.mha = AF3Attention(
            c_q=self.c_in,
            c_k=self.c_in,
            c_v=self.c_in,
            c_hidden=self.c_hidden,
            no_heads=self.no_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)

        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        triangle_bias = af3_permute_final_dims(self.linear_z(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        x = self.mha(
            q_x=x,
            kv_x=x,
            biases=biases,
        )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


AF3TriangleAttentionStartingNode = AF3TriangleAttention


class AF3TriangleAttentionEndingNode(AF3TriangleAttention):
    def __init__(self, *args, **kwargs):
        kwargs["starting"] = False
        super().__init__(*args, **kwargs)
"""
