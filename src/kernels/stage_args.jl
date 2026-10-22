# The argument list of every built-in binary, written ONCE. Every launch of a built-in stage builds
# its arguments here: the op's `forward`/`backward` (outputs as `OutArray`s, `src/kernels/sdpa_op.jl`)
# and the tuner's candidate launches (preallocated buffers on a packed problem). So what the tuner
# verifies and times is bound exactly as what the op calls, and `test/kernels/stage_args.test.jl` pins one list.
#
# The order is the compiled kernels' (`kernel_argtypes`, `mma_argtypes`). KernelDispatch checks the TYPES
# of the list against the binary's recorded `signature` (a `Float16` buffer for a `Float32` slot, an `Int64`
# for an `Int32`, is refused naming the argument); it cannot tell two buffers of one type apart, so a swap
# of those still binds without error — `test/kernels/stage_args.test.jl` pins the order, and the launch
# tests compare every kernel's results with a reference, which a swapped pair would break.

"""
    stage_args(kernel, Val(stage), a, s) -> Tuple

The arguments of one built-in stage's binary, in slot order. `a` holds the buffers by role — `o`,
`lse`, `q`, `k`, `v`, `bias` (keys fastest), `bias_t` (queries fastest), `mask`, `dO`, `delta`, `dq`,
`dk`, `dv`, `dbias`, `dbias_t` — of which a stage reads only its own; an output may be an `OutArray`
(allocated by the launch) or a preallocated array. `s` is `(scale, n, nouter, nh, mheads)`
(`_scalars` for the op's own layout).
"""
function stage_args end

# `:default` (`FlashScalar`): one thread per query reads the bias transposed (`bias_t`) in the forward
# and `:dq`, which writes the bias gradient transposed; `:dk`/`:dv` read it as given. Only the
# forward takes `neg_inf`.
stage_args(::FlashScalar, ::Val{:fwd}, a, s) =
    (a.o, a.lse, a.q, a.k, a.v, a.bias_t, a.mask, s[1], NEG_INF_F32, Base.tail(s)...)
stage_args(::FlashScalar, ::Val{:dq}, a, s) =
    (a.dq, a.dbias_t, a.q, a.k, a.v, a.bias_t, a.mask, a.dO, a.lse, a.delta, s...)
stage_args(::FlashScalar, ::Val{:dk}, a, s) = (a.dk, a.q, a.k, a.v, a.bias, a.mask, a.dO, a.lse, a.delta, s...)
stage_args(::FlashScalar, ::Val{:dv}, a, s) = (a.dv, a.q, a.k, a.bias, a.mask, a.dO, a.lse, s...)

# `:flash_mma` (`FlashMMA`): the bias as given everywhere; `:dkv` writes `dbias` untransposed.
stage_args(::FlashMMA, ::Val{:fwd}, a, s) = (a.o, a.lse, a.q, a.k, a.v, a.bias, a.mask, s...)
stage_args(::FlashMMA, ::Val{:dkv}, a, s) =
    (a.dk, a.dv, a.dbias, a.q, a.k, a.v, a.bias, a.mask, a.dO, a.lse, a.delta, s...)
stage_args(::FlashMMA, ::Val{:dq}, a, s) = (a.dq, a.q, a.k, a.v, a.bias, a.mask, a.dO, a.lse, a.delta, s...)

"""
    stage_outputs(kernel, stage) -> Tuple{Vararg{Symbol}}

The roles one stage writes, in the order its binary returns them (the leading arguments of
[`stage_args`](@ref)).
"""
stage_outputs(::FlashScalar, stage::Symbol) =
    stage === :fwd ? (:o, :lse) : stage === :dq ? (:dq, :dbias_t) : stage === :dk ? (:dk,) :
    stage === :dv ? (:dv,) : throw(ArgumentError("no `:default` stage `:$stage`"))
stage_outputs(::FlashMMA, stage::Symbol) =
    stage === :fwd ? (:o, :lse) : stage === :dkv ? (:dk, :dv, :dbias) : stage === :dq ? (:dq,) :
    throw(ArgumentError("no `:flash_mma` stage `:$stage`"))
