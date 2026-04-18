py"""
import math
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm
from typing import List, Optional, Tuple
from functools import partialmethod

def af2_permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    num_first_dims = len(tensor.shape)-len(inds)
    first_inds = list(range(num_first_dims))
    return tensor.permute(first_inds + [num_first_dims + i for i in inds])

def af2_flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))

def _af2_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    print('q', query.shape)
    print('mask', biases[0].shape)
    print('bias', biases[1].shape)
    key = af2_permute_final_dims(key, (1, 0))
    a = torch.matmul(query, key)
    print('a', a.shape)
    for b in biases:
        a += b
        print('here')
    
    a = nn.functional.softmax(a, dim=-1)
    print('maybe')
    a = torch.matmul(a, value)
    print('there')
    return a

class AF2Attention(nn.Module):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        inf:float = 1e9,
    ):
        super(AF2Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        self.inf = inf

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
            self.c_hidden * self.no_heads, self.c_q
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
        q_x: torch.Tensor, 
        kv_x: torch.Tensor,
        apply_scale: bool = True
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

    def _wrap_up(self,
        o: torch.Tensor, 
        q_x: torch.Tensor
    ) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = af2_flatten_final_dims(o, 2)
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale = True)
        
        o = _af2_attention(q, k, v, biases)
        o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


class AF2TriangleAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, starting=True, inf=1e9
    ):
        super(AF2TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)
        self.linear = Linear(c_in, self.no_heads, bias=False)

        self.mha = AF2Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    def forward(self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if(not self.starting):
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)

        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = af2_permute_final_dims(self.linear(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        x = self.mha(q_x=x, kv_x=x, biases=biases)

        if(not self.starting):
            x = x.transpose(-2, -3)

        return x


AF2TriangleAttentionStartingNode = AF2TriangleAttention


class AF2TriangleAttentionEndingNode(AF2TriangleAttention):
    __init__ = partialmethod(AF2TriangleAttention.__init__, starting=False)
"""
