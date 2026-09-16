"""
    Attention(chn_q, chn_k, chn_v, head_dim, num_heads; use_bias=false, fuse_qkv=true,
        use_gate=static(true), bias_layout=:kq, chunk_size=nothing, use_kernel=static(false))
    Attention(chn_q, chn_kv, head_dim, num_heads; kwargs...)
    Attention(chn_in, head_dim, num_heads; kwargs...)

Multi-head attention layer. Supports self-attention, cross-attention, and gated attention.

## Arguments
- `chn_q`: Channels for the query input.
- `chn_k`: Channels for the key input.
- `chn_v`: Channels for the value input.
- `head_dim`: Dimension of each attention head.
- `num_heads`: Number of attention heads.

## Keyword Arguments
- `use_bias`: A `NamedTuple` or `Bool` specifying which internal layers should use bias. 
  Defaults to `false`.
- `fuse_qkv`: If `true`, fuses the Q, K, and V projections into a single dense layer 
  (where possible).
- `use_gate`: If `true` (or `static(true)`), applies a sigmoid gating to the attention output.
- `use_kernel`: If `true` (or `static(true)`), routes attention through the default (or user-defined) 
  fused kernel (see [`register_kernel!`](@ref) / [`use_kernel!`](@ref) / [`tune_flash_kernel!`](@ref)) 
  instead of `Lux.scaled_dot_product_attention`. Defaults to `false`.

  This only takes effect on GPU arrays. On CPU, [`_sdpa_kernel`](@ref) always runs the ordinary
  softmax regardless of this flag. A model built with `use_kernel=true` still runs (correctly, 
  just without the speedup); the flag applies the moment the arrays are actually on a device with a 
  usable kernel selected.

  Mutually exclusive with `chunk_size`: a fused kernel tiles the keys internally, so there is nothing
  left for `chunk_size` to slice, and combining the two raises `AssertionError` at construction.
- `bias_layout`: A `Symbol` (either `:qk` or `:kq`) indicating how the attention bias tensor 
  should be prepared internally. Defaults to `:kq`.
  - `:qk`: Used in `TriangleAttention`. Expects a bias of shape `[H, Ni, Nj, B]`, which is 
    permuted to `[Ni, Nj, H, 1, B]` to match the Julia Lux 3rd-dimension attention default. 
    Since the input is also permuted in `TriangleAttention`, no additional transpose of the 
    bias is required.
  - `:kq`: Used in pair-bias implementations (e.g., MSA or general pair representations). 
    Expects a bias of shape `[H, Nq, Nk, B]`, which is permuted/transposed to `[Nk, Nq, H, 1, B]` 
    to correct for the difference between Julia Lux's 3rd-dimension attention and Python's
    4th-dimension attention when working with inputs of shape `[C, N, S, B]` (vs Python's `[B, S, N, C]`).
- `chunk_size`: If given, run the attention in slices of `chunk_size` along dim 4 of the prepared
  q/k/v instead of all at once. Defaults to `nothing` (no chunking). The result is unchanged but the 
  live score tensor shrinks from `[Nq, Nk, H, dim4, …]` to `chunk_size` entries at a time.

  Attention runs over `token_dim=3`, so dim 4 is the first axis it batches over and for a pair input 
  `[C, Ni, Nj, B]` that is **`Nj`, a token axis, not the sample batch**. Chunking therefore helps at 
  `B == 1`, whose scores are O(N³·H), ~1 GiB per op at N=400. For a 3-D input `[C, N, B]` dim 4 *is* 
  the sample batch, so `chunk_size` is a no-op at `B == 1`. Equivalent to `chunk_size_tri_attn` in the 
  openfold/boltz references.

  Under `Reactant.@compile` the Reactant extension rolls the chunk loop into a `stablehlo.while`;
  the plain Julia loop here would be unrolled and scheduled concurrently, preventing any reductions in 
  memory usage.

  !!! warning "Verify the REVERSE before relying on a specific chunk size"
      Differentiating the rolled loop hits an unfixed Reactant/Enzyme defect for some
      configurations: the chunk write lowers to a `dynamic_update_slice` whose update has one rank
      too many (`update rank does not match operand rank: 6 vs 5`). The FORWARD is unaffected.

      No rule predicts it. Measured on a bare `Attention` (chn=32, head_dim=8, num_heads=4), the
      same at every `N` tried (8, 12, 16, 20, 24):

      | nchunks | 2 | 3 | 4 | 5 | 6 |
      |:--------|:-:|:-:|:-:|:-:|:-:|
      | reverse | ✗ | ✗ | ✓ | ✗ | ✓ |

      A specific nchunks might work or fail depending on the surrounding layer (e.g. a PairFormer block
      featuring TriangleAttention layers). Eliminated as causes: the chunked axis's position and
      rank, the loop body, the number of carried buffers, mask, bias, `num_heads`, and absolute `N`.

## Inputs
- `x`: Input tensor(s). Can be an `AbstractArray` for self-attention, or a `Tuple` of 
  arrays for cross-attention (e.g., `(q, kv)` or `(q, k, v)`). 
  Expected shape: `[C, N, (N or S, ) B]`.
- `bias`: Optional attention bias tensor. Expected shape: `[num_heads, Nq, Nk, B]`.
  Must have the correct shape for broadcasting. See `prep_bias`.
- `mask`: Optional attention mask. Expected shape: `[Nq, (S, ) B]` or `[Ni, Nj, B]`.
  Masks are automatically reshaped to the internal 4D/5D attention score format.

## Returns
- `y`: The output tensor. Shape matches `q` (typically `[chn_q, N, (N or S, ) B]`).
- `st`: Updated state.

## Example
```julia
using Lux, LuxTriangleAttention, Random

# Self-attention with mask
model = Attention(64, 32, 4)
ps, st = Lux.setup(Random.default_rng(), model)

x = randn(Float32, 64, 32, 1)
mask = rand(Bool, 32, 1)

# Pass inputs as a NamedTuple
y, st = model((; x, mask), ps, st)
```

## Note on Dimensions
This implementation performs attention over the 3rd dimension (`token_dim=3`).
This is different from most Python references (e.g., Boltz-2, AlphaFold2 and 3) which 
typically attend over the 4th dimension.

- **TriangleAttention (`bias_layout=:qk`)**:
  In `TriangleAttention`, we pass a bias of shape `[H, Ni, Nj, B]` which is permuted internally to enable the model to attend to the 3rd dimension in `q, k, v` (representing `Nj`). The input is of size `[C, Ni, Nj, B]`. Permuting the input also permutes the resulting bias correctly, meaning no additional changes to the bias orientation are needed.
- **Generic pair-bias attention (`bias_layout=:kq`)**, matching the layout of the `AttentionPairBias`
  module in the AlphaFold3/Boltz Python references.
  For generic pair bias attention, the input is of shape `[C, N, S, B]` (where `S` is the pair/MSA dimension), and we attend over the `N` dimension. In Python, the equivalent shape is `[B, S, N, C]`. While the Julia input layout works naturally with Lux's 3rd-dimension attention default, the bias tensor of shape `[H, Nq, Nk, B]` must be permuted to `[Nk, Nq, H, 1, B]` to correct for the transposition. This is handled by the `:kq` layout.
"""
struct Attention{SG,QKV,G,O,BIAS,CS<:Union{Nothing,Int},UK<:StaticBool} <:
       Lux.AbstractLuxContainerLayer{(:qkv, :gate, :out)}
    should_gate::SG
    qkv::QKV
    gate::G
    out::O
    head_dim::Int
    num_heads::Int # H
    bias_layout::BIAS
    chunk_size::CS
    use_kernel::UK
end

Attention(chn_in::Int, head_dim::Int, num_heads::Int; kwargs...) =
    Attention(chn_in, chn_in, chn_in, head_dim, num_heads; kwargs...)

Attention(chn_q::Int, chn_kv::Int, head_dim::Int, num_heads::Int; kwargs...) =
    Attention(chn_in, chn_kv, chn_kv, head_dim, num_heads; kwargs...)

"""
Ideally we combine with TriAttnCore at some point, as they do exactly the same aside 
from the triangle_attention vs Lux.scaled_dot_product_attention calls.

Note: the forward function expects bias to be in the correct shape!!
"""
function Attention(
    chn_q::Int, chn_k::Int, chn_v::Int, head_dim::Int, num_heads::Int;
    use_bias=false, fuse_qkv::Bool=true, use_gate=static(true), bias_layout::Symbol=:kq,
    chunk_size::Union{Nothing,Int}=nothing, use_kernel::Union{Bool,StaticBool}=static(false)
)
    @assert bias_layout == :qk || bias_layout == :kq "`bias_layout` must be :qk or :kq."
    @assert isnothing(chunk_size) || chunk_size > 0 "`chunk_size` must be a positive integer."
    @assert !(dynamic(use_kernel) && !isnothing(chunk_size)) "`use_kernel` and \
        `chunk_size` are alternative ways to keep the score tensor small and cannot be combined: \
        a kernel tiles the keys internally, so there is nothing left for `chunk_size` to slice."

    use_bias = resolve_defaults(use_bias, (:qkv, :q, :kv, :k, :v, :gate, :out))
    use_gate_static = static(use_gate)

    qkv = if fuse_qkv
        fuse_all = chn_q == chn_k == chn_v
        fuse_kv = (chn_q !== chn_k) && (chn_k == chn_v)

        if fuse_all
            Lux.Dense(chn_q => 3 * num_heads * head_dim; use_bias=use_bias.qkv)
        elseif fuse_kv
            Lux.Chain(
                q=Lux.Dense(chn_q => num_heads * head_dim; use_bias=use_bias.q),
                kv=Lux.Dense(chn_k => 2 * num_heads * head_dim; use_bias=use_bias.kv)
            )
        else
            throw(ErrorException("When fuse_qkv = true, the inputs channels for q, k, and v should either all match, or k and v should be equal."))
        end
    else
        Lux.BranchLayer(
            q=Lux.Dense(chn_q => num_heads * head_dim; use_bias=use_bias.q),
            k=Lux.Dense(chn_k => num_heads * head_dim; use_bias=use_bias.k),
            v=Lux.Dense(chn_v => num_heads * head_dim; use_bias=use_bias.v)
        )
    end

    gate = if known(use_gate_static)
        Lux.Dense(chn_q => num_heads * head_dim, Lux.sigmoid; use_bias=use_bias.gate)
    else
        Lux.NoOpLayer()
    end

    out = Lux.Dense(num_heads * head_dim => chn_q; use_bias=use_bias.out)

    return Attention(
        use_gate_static,
        qkv,
        gate,
        out,
        head_dim,
        num_heads,
        static(bias_layout),
        chunk_size,
        static(use_kernel)
    )
end


(l::Attention)(inputs::NamedTuple, ps, st) = l(
    inputs.x, # Either a tuple or an AbstractArray
    get(inputs, :bias, nothing),
    get(inputs, :mask, nothing),
    ps, st
)

(l::Attention)(x, ps, st) = l(x, nothing, nothing, ps, st)

(l::Attention)(x, bias::AbstractArray{T}, ps, st) where T<:Number = 
    l(x, bias, nothing, ps, st)

(l::Attention)(x, mask::AbstractArray{Bool}, ps, st) = l(x, nothing, mask, ps, st)

# x is either [C, N, B], [C, N, N, B] or [C, N, S, B]
function (l::Attention)(x, bias, mask, ps, st)
    (q, k, v), st_qkv = _prep_qkv(l.qkv, x, ps.qkv, st.qkv; head_dim=l.head_dim, num_heads=l.num_heads)

    mask = prep_mask(mask)
    bias = prep_bias(bias, x, l.bias_layout)

    attn = _sdpa(l.use_kernel, l.chunk_size, q, k, v, mask, bias)

    _dims = size(attn)[3:end] # [N, B] or [N, S, B] dims
    attn = reshape(attn, l.head_dim * l.num_heads, _dims...)
    attn, st_gate = _gate_maybe(l.gate, attn, x, ps.gate, st.gate)

    y, st_out = l.out(attn, ps.out, st.out)

    return y, (qkv=st_qkv, gate=st_gate, out=st_out)
end

"""
    _sdpa(use_kernel, chunk_size, q, k, v, mask, bias)

Entry point from the layer forward: pick how the score tensor is kept small, by the *types* of the
two knobs rather than by branching on their values.

| `use_kernel` | `chunk_size` | route |
|:--|:--|:--|
| `False` | `nothing` | [`_sdpa`](@ref) — `Lux.scaled_dot_product_attention` |
| `False` | `Int`     | [`_chunked_sdpa`](@ref) — slice dim 4 |
| `True`  | `nothing` | [`_sdpa_kernel`](@ref) — a fused kernel for this device |
| `True`  | `Int`     | error: the two are alternatives, not a combination |

Every route here returns just the bare output array: `Attention`'s forward reshapes and gates it
directly. `_sdpa_kernel` itself always returns an `(output, lse)` PAIR — `lse` is the residual a
device kernel's reverse rule needs, and `nothing` on the host path, where there is no reverse rule
to feed — so the `use_kernel=true` route below unwraps it with a plain `first`, unconditionally,
rather than testing the shape of the result. See [`_sdpa_kernel`](@ref)'s docstring in
`src/kernels/backend.jl` for the full routing.
"""
_sdpa(::False, chunk_size::Int, q, k, v, mask, bias) =
    _chunked_sdpa(chunk_size, q, k, v, mask, bias)

# `_sdpa_kernel` should always return an `(output, lse)` pair: `lse` is the reverse rule's residual 
# on a device path and `nothing` on the host path, where there is no reverse rule to feed. Because 
# the shape is unconditional, unwrapping is a plain `first` rather than a runtime test of the result.
function _sdpa(::True, ::Nothing, q, k, v, mask, bias)
    # The kernel is SELF-attention-shaped: keys are read at the queries' own extent (`n`), so a
    # cross-attention call (Boltz2/AF3's windowed atom attention: 32 queries against 128 keys)
    # would be mis-addressed rather than rejected. Refuse it here, where the shapes are still the
    # layer's own and the message can say which knob to turn off.
    size(k)[3:end] == size(q)[3:end] || throw(ArgumentError(
        "`use_kernel=true` needs self-attention-shaped q/k (the fused kernel reads keys at the " *
        "query extent); got q $(size(q)) and k $(size(k)). Use `use_kernel=false` for this " *
        "(cross/windowed) attention."))
    out, _ = _sdpa_kernel(q, k, v, mask, bias)
    return out
end

_sdpa(::True, ::Int, args...) = throw(ArgumentError(
    "`use_kernel=true` cannot be combined with a `chunk_size`: a fused kernel already " *
    "tiles the keys internally, so there is no full score tensor left for `chunk_size` to " *
    "slice. Drop one of the two."))

_sdpa(::False, ::Nothing, q, k, v, mask, bias) = _sdpa_default(q, k, v, mask, bias)

# The bare 5-arg form `_chunked_sdpa`/`layers/chunk.jl` calls when there is nothing left to chunk
# (`chunk_size >= L`, or `chunk_size === nothing`)
_sdpa(q, k, v, mask, bias) = _sdpa_default(q, k, v, mask, bias)

function _sdpa_default(q, k, v, mask, bias)
    output, _scores = Lux.scaled_dot_product_attention(
        q, k, v; head_dim=1, token_dim=3, mask, bias
    )   # attends over token dim (= 4 in python)
    return output
end

# fused qkv
function _prep_qkv(qkv::Lux.Dense, x::AbstractArray, ps, st; head_dim, num_heads)
    _qkv, st_qkv = qkv(x, ps, st)# [3 * H * head_dim, N, B] or [3 * H * head_dim, N, N, B]

    _qkv_reshaped = reshape(_qkv, head_dim, num_heads, 3, size(_qkv)[2:end]...)
    q = view(_qkv_reshaped, :, :, 1, ntuple(_ -> Colon(), ndims(_qkv_reshaped) - 3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    k = view(_qkv_reshaped, :, :, 2, ntuple(_ -> Colon(), ndims(_qkv_reshaped) - 3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    v = view(_qkv_reshaped, :, :, 3, ntuple(_ -> Colon(), ndims(_qkv_reshaped) - 3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    return (q, k, v), st_qkv
end

function _prep_qkv(qkv::Lux.Chain, x::Tuple, ps, st; head_dim, num_heads)
    x_q, x_kv = x
    q, st_q = qkv.layers.q(x_q, ps.q, st.q) # [H * head_dim, N, B]
    _kv, st_kv = qkv.layers.kv(x_kv, ps.kv, st.kv) # [2 * H * head_dim, N, B]

    _kv_reshaped = reshape(_kv, head_dim, num_heads, 2, size(_kv)[2:end]...)
    q = reshape(q, head_dim, num_heads, size(q)[2:end]...) # [H, head_dim, N, B]
    k = view(_kv_reshaped, :, :, 1, ntuple(_ -> Colon(), ndims(_kv_reshaped) - 3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    v = view(_kv_reshaped, :, :, 2, ntuple(_ -> Colon(), ndims(_kv_reshaped) - 3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    return (q, k, v), (q=st_q, kv=st_kv)
end

# nonfused qkv
function _prep_qkv(qkv::Lux.BranchLayer, x::AbstractArray, ps, st; head_dim, num_heads)
    (q, k, v), st_qkv = qkv(x, ps, st)

    q = reshape(q, head_dim, num_heads, size(q)[2:end]...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    k = reshape(k, head_dim, num_heads, size(k)[2:end]...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    v = reshape(v, head_dim, num_heads, size(v)[2:end]...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    return (q, k, v), st_qkv
end

function _prep_qkv(qkv::Lux.BranchLayer, x::Tuple, ps, st; head_dim, num_heads)
    x_q, x_k, x_v = x
    q, st_q = qkv.layers.q(x_q, ps.q, st.q)
    k, st_k = qkv.layers.k(x_k, ps.k, st.k)
    v, st_v = qkv.layers.v(x_v, ps.v, st.v)

    q = reshape(q, head_dim, num_heads, size(q)[2:end]...) # [head_dim, H, N, B]
    k = reshape(k, head_dim, num_heads, size(k)[2:end]...) # [head_dim, H, N, B]
    v = reshape(v, head_dim, num_heads, size(v)[2:end]...) # [head_dim, H, N, B]
    return (q, k, v), (q=st_q, k=st_k, v=st_v)
end

"""
    prep_mask(mask)

Utility to reshape various mask shapes into the internal 4D/5D format expected by 
`Lux.scaled_dot_product_attention`.

## Arguments
- `mask`: Input mask.

## Returns
Supported conversions:
- `[N, B]` -> `[N, 1, 1, B]`
- `[Ni, Nj, B]` -> `[Ni, 1, 1, Nj, B]`
"""
prep_mask(::Nothing) = nothing
function prep_mask(mask::AbstractArray{T,2}) where T
    N, B = size(mask)
    return reshape(mask, N, 1, 1, B)
end

function prep_mask(mask::AbstractArray{T,3}) where T
    Ni, Nj, B = size(mask) # Or N, S, B
    return reshape(mask, Ni, 1, 1, Nj, B) # or [N, 1, 1, S, B]
end

prep_mask(mask::AbstractArray{T,5}) where T = mask # return as-is


"""
    prep_bias(bias, x, bias_layout)

Utility to prepare and reshape the attention bias tensor based on the input shape and the specified `bias_layout`.

## Arguments
- `bias`: The raw bias tensor (or `nothing`).
- `x`: The input tensor to the attention layer.
- `bias_layout`: A `StaticSymbol` specifying the permutation strategy (`:qk` or `:kq`).

## Permutation Strategies
- `:qk` (Symmetric/Triangle Attention):
  For an input of shape `[C, Ni, Nj, B]` and bias of shape `[H, Ni, Nj, B]`, permutes the bias to `[Ni, Nj, H, 1, B]` (preserving the spatial dimensions `Ni` and `Nj`).
- `:kq` (Asymmetric/Pair Attention):
  For an input of shape `[C, N, S, B]` and bias of shape `[H, Nq, Nk, B]`, permutes the bias to `[Nk, Nq, H, 1, B]` (transposing spatial dimensions to correct for Julia's 3rd-dimension attention vs Python's 4th-dimension attention).

## Notes on Dimensions
Python packages (such as AlphaFold and Boltz) typically perform attention over the 4th dimension (using `[B, S, N, C]`).
Julia's `Lux` default is to attend over the 3rd dimension (`[C, N, S, B]`).
The `bias_layout` determines how the bias tensor is adjusted to maintain parity with the Python attention mechanisms under these different dimension conventions.
"""
prep_bias(::Nothing, args...) = nothing

# Cross-attention: `x` is the `(q, k, v)` tuple. The bias layout is decided by the query, so
# dispatch on `q` — same convention as `_gate_maybe` below.
prep_bias(bias::AbstractArray, x::Tuple, bias_layout::StaticSymbol) =
    prep_bias(bias, first(x), bias_layout)

# As in AttentionPairBias:
function prep_bias(bias::AbstractArray{T,4}, ::AbstractArray{T,4}, ::StaticSymbol{:kq}) where T
    H, Nq, Nk, B = size(bias)
    return reshape(permutedims(bias, (3, 2, 1, 4)), Nk, Nq, H, 1, B)
end

# As in TriangleAttention:
function prep_bias(bias::AbstractArray{T,4}, ::AbstractArray{T,4}, ::StaticSymbol{:qk}) where T
    H, Nq, Nk, B = size(bias)
    return reshape(permutedims(bias, (2, 3, 1, 4)), Nq, Nk, H, 1, B)
end

# User already put the bias in the correct shapes, return as-is
prep_bias(bias::AbstractArray{T,4}, ::AbstractArray{T,3}, ::StaticSymbol{:kq}) where T =
    permutedims(bias, (3, 2, 1, 4)) # -> [Nj, Ni, H, B]

prep_bias(bias::AbstractArray{T,4}, ::AbstractArray{T,3}, ::StaticSymbol{:qk}) where T =
    permutedims(bias, (2, 3, 1, 4)) # -> [Ni, Nj, H, B]

prep_bias(bias::AbstractArray{T,5}, ::AbstractArray{T,4}, ::StaticSymbol) where T = bias

_gate_maybe(::Lux.NoOpLayer, x::AbstractArray, g::Tuple, ps, st) = x, st
_gate_maybe(::Lux.NoOpLayer, x::AbstractArray, g::AbstractArray, ps, st) = x, st

_gate_maybe(l::Lux.AbstractLuxLayer, x::AbstractArray, g::Tuple, ps, st) =
    _gate_maybe(l, x, first(g), ps, st) # take the first element of g (i.e. q) for gating.

function _gate_maybe(l::Lux.AbstractLuxLayer, x::AbstractArray, g::AbstractArray, ps, st)
    g, st_gate = l(g, ps, st)
    y = @. x * g # TODO: consider in-place operation here.
    return y, st_gate
end