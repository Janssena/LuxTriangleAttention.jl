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
    key = af2_permute_final_dims(key, (1, 0))
    a = torch.matmul(query, key)
    for b in biases:
        a += b
    
    a = nn.functional.softmax(a, dim=-1)
    a = torch.matmul(a, value)
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


class AF2BaseTriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z, c_hidden, _outgoing):
        super(AF2BaseTriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = Linear(self.c_z, self.c_z)
        self.linear_z = Linear(self.c_hidden, self.c_z)

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if(self._outgoing):
            a = af2_permute_final_dims(a, (2, 0, 1))
            b = af2_permute_final_dims(b, (2, 1, 0))
        else:
            a = af2_permute_final_dims(a, (2, 1, 0))
            b = af2_permute_final_dims(b,  (2, 0, 1))

        if(_inplace_chunk_size is not None):
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i: i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i: i + _inplace_chunk_size, :, :]
                a[..., i: i + _inplace_chunk_size, :, :] = (
                    torch.matmul(
                        a_chunk,
                        b_chunk,
                    )
                )
            p = a
        else:
            p = torch.matmul(a, b)

        return af2_permute_final_dims(p, (1, 2, 0))

    def forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False
    ) -> torch.Tensor:
        pass


class AF2TriangleMultiplicativeUpdate(AF2BaseTriangleMultiplicativeUpdate):
    def __init__(self, c_z, c_hidden, _outgoing=True):
        super(AF2TriangleMultiplicativeUpdate, self).__init__(c_z=c_z,
                                                           c_hidden=c_hidden,
                                                           _outgoing=_outgoing)

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden)
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden)

    def _inference_forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_chunk_size: Optional[int] = None,
        with_add: bool = True,
    ):
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)
       
        def compute_projection_helper(pair, mask, a=True):
            if(a):
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p
            
            pair = self.layer_norm_in(pair)
            p = self.sigmoid(linear_g(pair))
            p = p * linear_p(pair)
            p = p * mask
            p = af2_permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(pair, mask, a=True, chunked=True): 
            need_transpose = self._outgoing ^ a
            if(not chunked):
                p = compute_projection_helper(pair, mask, a)
                if(need_transpose):
                    p = p.transpose(-1, -2)
            else:
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.bias.shape[-1]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = compute_projection_helper(
                        pair[..., i: i + inplace_chunk_size, :, :],
                        mask[..., i: i + inplace_chunk_size, :, :], 
                        a,
                    )
                    if(need_transpose):
                        pair_chunk = pair_chunk.transpose(-1, -2)
                        p[..., i: i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i: i + inplace_chunk_size, : ] = pair_chunk
                    
                    del pair_chunk

            return p

        a = compute_projection(z, mask, True, chunked=True)

        if(inplace_chunk_size is not None):
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim
            
            def empty_slicer(t):
                return [slice(None) for _ in t.shape]
            
            def slice_tensor(t, start, end, dim):
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[s]

            def flip_z_cache_(z_cache, z):
                quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
                z_cache = z_cache.transpose(row_dim, col_dim)
                z_cache = z_cache[..., :(n // 2), :, :]
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[first_half_slicer] = quadrant_3
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)
                z_cache[quadrant_3_slicer] = quadrant_4
                return z_cache

            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache.copy_(z[z_cache_slicer])
            z_cache_rotated = False

            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [i_2 - i_1 for i_1, i_2 in zip(i_range, i_range[1:] + [half_n])]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(i_range + after_half, initial_offsets + after_half_offsets)
            for i, offset in combined_range_with_offsets:
                if(not z_cache_rotated and i >= half_n):
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True

                z_chunk_b = slice_tensor(z, i, i + offset, b_chunk_dim)
                mask_chunk = slice_tensor(mask, i, i + offset, b_chunk_dim)
                z_chunk_b = z_chunk_b.clone()
                if(b_chunk_dim == col_dim):
                    z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
                else: 
                    if(not z_cache_rotated):
                        z_chunk_slicer = empty_slicer(z_chunk_b)
                        z_chunk_slicer[col_dim] = slice(0, half_n)
                        z_chunk_b[z_chunk_slicer] = slice_tensor(z_cache, i, i + offset, row_dim)
                    else:
                        z_cache_offset = i - half_n
                        z_chunk_b = slice_tensor(z_cache, z_cache_offset, z_cache_offset + offset, row_dim)

                b_chunk = compute_projection(z_chunk_b, mask_chunk, a=False, chunked=False)
                del z_chunk_b

                x_chunk = torch.matmul(a, b_chunk)
                x_chunk = af2_permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)
                z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                g_chunk = self.sigmoid(self.linear_g(self.layer_norm_in(z_chunk_g))) 
                del z_chunk_g
                x_chunk *= g_chunk
                z_slicer = empty_slicer(z)
                z_slicer[col_dim] = slice(i, i + offset)
                if(with_add):
                    z[z_slicer] += x_chunk
                else:
                    z[z_slicer] = x_chunk
        else:
            b = compute_projection(z, mask, False, False)
            x = torch.matmul(a, b)
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.sigmoid(self.linear_g(z))
            x = x * g
            if(with_add):
                z += x
            else:
                z = x
        return z

    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
    ) -> torch.Tensor:
        if inplace_safe:
            return self._inference_forward(z, mask, inplace_chunk_size=_inplace_chunk_size, with_add=_add_with_inplace)

        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask * self.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = mask * self.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)
        x = self._combine_projections(a, b)
        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g
        return x


class AF2TriangleMultiplicationOutgoing(AF2TriangleMultiplicativeUpdate):
    __init__ = partialmethod(AF2TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class AF2TriangleMultiplicationIncoming(AF2TriangleMultiplicativeUpdate):
    __init__ = partialmethod(AF2TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class AF2FusedTriangleMultiplicativeUpdate(AF2BaseTriangleMultiplicativeUpdate):
    def __init__(self, c_z, c_hidden, _outgoing=True):
        super(AF2FusedTriangleMultiplicativeUpdate, self).__init__(c_z=c_z,
                                                                c_hidden=c_hidden,
                                                                _outgoing=_outgoing)
        self.linear_ab_p = Linear(self.c_z, self.c_hidden * 2)
        self.linear_ab_g = Linear(self.c_z, self.c_hidden * 2)

    def _inference_forward(self,
                           z: torch.Tensor,
                           mask: Optional[torch.Tensor] = None,
                           _inplace_chunk_size: Optional[int] = None,
                           with_add: bool = True,
                           ):
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        def compute_projection(pair, mask):
            p = self.sigmoid(self.linear_ab_g(pair)) * self.linear_ab_p(pair) * mask
            left = p[..., :self.c_hidden]
            right = p[..., self.c_hidden:]
            return left, right

        z_norm_in = self.layer_norm_in(z)
        a, b = compute_projection(z_norm_in, mask)
        x = self._combine_projections(a, b, _inplace_chunk_size=_inplace_chunk_size)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z_norm_in))
        x = x * g
        if (with_add):
            z += x
        else:
            z = x
        return z

    def forward(self,
                z: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                inplace_safe: bool = False,
                _add_with_inplace: bool = False,
                _inplace_chunk_size: Optional[int] = 256
                ) -> torch.Tensor:
        if (inplace_safe):
            return self._inference_forward(z, mask, _inplace_chunk_size=_inplace_chunk_size, with_add=_add_with_inplace)

        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        ab = mask * self.sigmoid(self.linear_ab_g(z)) * self.linear_ab_p(z)
        a = ab[..., :self.c_hidden]
        b = ab[..., self.c_hidden:]
        x = self._combine_projections(a, b)
        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g
        return x


class AF2FusedTriangleMultiplicationOutgoing(AF2FusedTriangleMultiplicativeUpdate):
    __init__ = partialmethod(AF2FusedTriangleMultiplicativeUpdate.__init__, _outgoing=True)


class AF2FusedTriangleMultiplicationIncoming(AF2FusedTriangleMultiplicativeUpdate):
    __init__ = partialmethod(AF2FusedTriangleMultiplicativeUpdate.__init__, _outgoing=False)
"""
