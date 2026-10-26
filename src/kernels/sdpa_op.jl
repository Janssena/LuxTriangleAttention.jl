# Attention as a KernelDispatch op: `SDPA`, and the two kernels this package builds itself.
#
# Everything generic — selection, variants and their manifest, binding a binary's arguments, eager
# launch, embedding in a Reactant trace, the Enzyme rule — is KernelDispatch'. What is here is attention:
# how its operands become each binary's argument list (`forward`/`backward` below, in the exact slot
# order the compiled kernels take), and how the built-in kernels are keyed, sized and tuned.
# `test/kernels/stage_args.test.jl` pins every argument list. The
# element types are KernelDispatch' to check: a built-in binary records the `signature` it was compiled for
# (`kernel_argtypes` / `mma_argtypes`), and a launch whose buffers or scalars have other types is refused.

"""
    SDPA

The attention op: `SDPA()(q, k, v, mask, bias) -> (o, lse)`. `q`, `k`, `v` are `[hd, H, Nq, Nj, B]`
(or a plain `[hd, H, N, B]` sequence), `mask` a KEEP mask and `bias` the pair bias, both optional.
Host arrays run `_sdpa_default` (`lse = nothing`); device arrays the selected kernel —
`use_kernel!(SDPA, :flash_mma)`.

# A fully masked key set
If every key of an `(outer index, batch)` slice is masked, its queries attend UNIFORMLY, as in OpenFold and
Boltz2 (whose finite `1e9·(mask - 1)` bias makes all logits equal): `o` is the mean of `v` over the `n` keys,
never NaN. The reverse gives `dV = dO / n` to every key and nothing to `q`, `k` or `bias` (the logits were
replaced). Every path follows it: `_sdpa_default`, the rolled chunk path and both kernels, which write
`lse = -Inf` for such a query as the marker their backward reads. Only a WHOLE slice is dead this way;
a single query made all `-Inf` by its own bias is not.
"""
struct SDPA <: KernelDispatch.AbstractKernelOp end
KernelDispatch.opname(::SDPA) = :sdpa
KernelDispatch.host(::SDPA, q, k, v, mask, bias) = (_sdpa_default(q, k, v, mask, bias), nothing)
# The registry (and each built-in kernel's calibration, beside its binaries) lives in KernelDispatch'
# default cache, `<cache root>/ops/LuxTriangleAttention.sdpa/<kernel>/`; `KERNELDISPATCH_CACHE` moves the root.

"""
    FlashScalar

`:default`: this package's flash-attention kernel with SCALAR arithmetic, built by [`tune_flash_kernel!`](@ref).

Each thread owns one query. A threadgroup stages a tile of keys and values in local memory and every thread
walks it with ordinary fused multiply-adds, keeping its running softmax state and output row in registers;
the work is spread over queries, never over the matrix units. Contrast [`FlashMMA`](@ref), which computes
the same algorithm on the GPU's matrix units. Both are written in KernelAbstractions; what differs is how
the products are computed. The tuned parameters are the threadgroup size `tg`, the key-tile size `kt` and
the sub-block size `sb` (`flash/`).
"""
struct FlashScalar <: KernelDispatch.AbstractKernel end

"""
    FlashMMA

`:flash_mma`: the flash-attention kernel on the GPU's MATRIX units, built by [`tune_flash_mma!`](@ref).

MMA is *matrix multiply-accumulate*: the hardware operation `C += A·B` on a small tile (8×8
`simdgroup_matrix` on Metal; tensor cores on NVIDIA). A simdgroup owns a block of query rows and computes
`Q·Kᵀ` and `P·V` (and the backward products) a fragment at a time; only the row-wise softmax reductions
run on the lanes that hold the fragments. The fragment operations are per-backend hooks (`flash_mma/`),
Metal's in the extension. Contrast [`FlashScalar`](@ref), which uses plain per-thread arithmetic.
"""
struct FlashMMA <: KernelDispatch.AbstractKernel end
const BuiltinKernel = Union{FlashScalar,FlashMMA}

const RESERVED_KERNEL_NAME = :default
KernelDispatch.kernelname(::FlashScalar) = RESERVED_KERNEL_NAME
KernelDispatch.kernelname(::FlashMMA) = :flash_mma
# What runs unless `use_kernel!` says otherwise: `:default`, or the kernel the package preference
# `kernel.sdpa` names (`KernelDispatch.set_default_kernel!(SDPA, :flash_mma)`).
KernelDispatch.@default_kernel SDPA FlashScalar() FlashMMA()
# The variant-key format is part of what a manifest means: bumped when `_key`'s shape or the binaries'
# launch description changes, so an old manifest is dropped (and the kernel re-tuned on demand) rather
# than misread. k3: launch size from `KernelBinary.tile` (KernelDispatch' `extent`), `is_ka` replaces `ka`.
# k4: the length element is the tuned length itself (was a power-of-two bucket).
# k5: KernelDispatch 0.2 — every binary records its `signature` and a launch is checked against it. A
# variant registered before has none (it loads unchecked and warns on its first launch), so those are
# dropped and re-tuned rather than kept.
# k6: a variant's binaries also record the tuned point as their `params` (`build_stage`).
const _KEY_FORMAT = "k6"
KernelDispatch.source_tag(::FlashScalar) = kernel_source_tag() * "-" * _KEY_FORMAT
KernelDispatch.source_tag(::FlashMMA) = flash_mma_source_tag() * "-" * _KEY_FORMAT

# --- operands -------------------------------------------------------------------------------------


"""
    _seq5(x)

A plain sequence's `[hd, H, N, B]` operand as the rank-5 `[hd, H, N, 1, B]` every kernel speaks:
attention over `N` batched over an outer extent of ONE. A pure reshape — the memory order is
identical.
"""
function _seq5(x::AbstractArray{T,4}) where {T}
    hd, H, N, B = size(x)
    return reshape(x, hd, H, N, 1, B)
end
_seq(x) = ndims(x) == 4 ? _seq5(x) : x
_f32(x) = realtype(eltype(x)) === Float32 ? x : Float32.(x)
const NEG_INF_F32 = -floatmax(Float32)

"""
    _sdpa_prep(q, k, v, mask, bias, skip) -> (q, k, v, mask, bias, d)

The caller's operands as every attention binary reads them: rank-5 `q/k/v` and a `[Nq, Nq, H, B]` bias,
all in the storage type (`q`'s: a `k`, `v` or bias of another precision is converted, see `_store`
and `_to_storage`), and a DROP mask `[Nq, Nj, B]` (the caller's is a KEEP mask). An absent bias or mask
is a 1-element placeholder when `skip` (the kernel has a specialisation that never reads it), else a
zero bias (adds nothing to any score) and an all-live mask. `d` holds the extents and flags.
"""
function _sdpa_prep(q, k, v, mask, bias, skip::Bool)
    q = _seq(q)
    T = realtype(eltype(q))
    _check_eltype(T)
    k, v = _store(T, _seq(k)), _store(T, _seq(v))
    hd, H, Nq, Nj, B = size(q)
    b = bias === nothing ?
        (skip ? KernelDispatch.device_zeros(q, T, 1) : KernelDispatch.device_zeros(q, T, Nq, Nq, H, B)) :
        reshape(realtype(eltype(bias)) === T ? bias : _to_storage(T, bias), Nq, Nq, H, B)
    m = mask === nothing ?
        (skip ? KernelDispatch.device_zeros(q, Bool, 1) : KernelDispatch.device_zeros(q, Bool, Nq, Nj, B)) :
        .!reshape(mask, Nq, Nj, B)
    d = (; T, hd, H, Nq, Nj, B, hasbias=bias !== nothing, hasmask=mask !== nothing)
    return q, k, v, m, b, d
end

# `x` in the storage type `T`. The binaries are compiled for ONE storage type per launch (their recorded
# `signature` has `T` for `q`, `k`, `v`, `dO`, `o` and the bias), and KernelDispatch refuses a launch that
# binds another; a `Float32` `k` beside `Float16` activations is the caller's mixed precision, so it is
# converted here, as the bias is, not reinterpreted and not refused.
_store(::Type{T}, x) where {T} = realtype(eltype(x)) === T ? x : T.(x)

# A bias converted to the storage type SATURATES at `±floatmax(T)/2` (`mask_value`) instead of
# overflowing to `±Inf`. The case it is for: a mask folded into the bias as `1e9·(mask - 1)` (OpenFold,
# Boltz) is finite in Float32 but `-Inf` in Float16 (max 65504), and a query whose every key got `-Inf`
# is `NaN`. Saturated, a masked key still gets weight `exp(-32752) = 0` beside any live key, exactly as
# masking. A query whose EVERY key is masked through the bias alone stays finite but attends by
# `softmax(q·k)`: the uniform weights of the `mask` argument (and, by rounding, of a Float32 `-1e9`) are
# not reproduced; such rows are padding. `max`/`min` rather than `clamp`, which traced arrays also take.
function _to_storage(::Type{T}, b) where {T}
    lo = Float32(mask_value(T))
    return T.(max.(min.(b, -lo), lo))
end

# The scalars every built-in stage takes after its buffers: scale, n, nouter, nh, mheads (see `stage_args`).
_scalars(d) = (Float32(inv(sqrt(d.hd))), Int32(d.Nq), Int32(d.Nj), Int32(d.H), Int32(d.H))
# `Σ_d o·dO`, laid out like `lse`, in Float32 whatever the storage type: the kernels read it as such.
_delta(o, dO) = permutedims(sum(_f32(o) .* _f32(dO); dims=1), (1, 3, 2, 4, 5))
# The pair bias with its two pair axes swapped (queries fastest), and back.
_pair_t(x, d) = d.hasbias ? permutedims(reshape(x, d.Nq, d.Nq, :), (2, 1, 3)) : x
# A plain sequence's results back at its own rank.
_finish(o, lse, q) = ndims(q) == 4 ?
    (reshape(o, size(q)), reshape(lse, 1, size(q, 3), size(q, 2), size(q, 4))) : (o, lse)
# A gradient in its argument's shape and element type (`dbias` accumulates in Float32).
_as(g, x) = (g = reshape(g, size(x)); realtype(eltype(g)) === realtype(eltype(x)) ? g : realtype(eltype(x)).(g))

_res(d) = (1, d.Nq, d.H, d.Nj, d.B)       # lse / delta
_full(d) = (d.hd, d.H, d.Nq, d.Nj, d.B)   # q-shaped

# --- the built-in kernels -------------------------------------------------------------------------

# Variant key: storage type, head dim, sequence length, and whether a bias / mask is present —
# absent-operand specialisations are separate binaries, so they are separate variants. A call keys
# on its own length; the variants are keyed on the lengths they were tuned for (`tune_lengths`), and
# `nearest` picks the closest. Not the outer
# extent or head count: measured on an M1 Pro over triangle, MSA-row and plain-sequence shapes, the
# best configuration chosen per (device, dtype, dim) and length costs at most ~10% against one tuned
# per geometry (mean ≤3%), so geometry is not worth a tuning dimension (docs/done/FEATURE_flash_mma.md).
_key(d) = (nameof(d.T), d.hd, d.Nq, d.hasbias, d.hasmask)

# The launch size every built-in stage covers: one row per query, every outer row and head slice. A
# binary's `tile` is `(rows, 1, 1)`, its queries per group, so it launches `(cld(Nq, rows), Nj, H·B)`.
_extent(d) = (d.Nq, d.Nj, d.H * d.B)

"""
    nearest(::SDPA, ::BuiltinKernel, keys, key)

The variant for a call with no exact one: same dtype, head dim and bias/mask presence (a different
ABI or binary is never substituted), then the tuned length closest to the call's, ties to the larger.
Every variant computes every length; the length only decides which tuning fits best.
"""
function KernelDispatch.nearest(::SDPA, ::BuiltinKernel, keys, key)
    cands = [k for k in keys if length(k) == length(key) &&
                                (k[1], k[2], k[4], k[5]) == (key[1], key[2], key[4], key[5])]
    isempty(cands) && return nothing
    return argmin(k -> (abs(k[3] - key[3]), -k[3]), cands)
end

# The storage type a variant key names (`nameof(T)`): BFloat16 is BFloat16s.jl's (`Core.BFloat16` from Julia 1.12), which `Base`
# does not export on every Julia version.
_storage_type(name::Symbol) = name === :BFloat16 ? BFloat16 : getfield(Base, name)

# No variant at all for this dtype, head dim and bias/mask presence: tune the default lengths now
# (`tune_lengths`) — the kernels are ours, and a fresh install should just work. Never mid-trace:
# tuning is eager device work the tracer would capture.
function KernelDispatch.build!(::SDPA, k::BuiltinKernel, key)
    T, dim, n = _storage_type(key[1]), key[2], key[3]
    name, ls = KernelDispatch.kernelname(k), tune_lengths()
    tune = (k isa FlashScalar ? "tune_flash_kernel!" : "tune_flash_mma!") * "($ls; dim=$dim, T=$T)"
    KernelDispatch.tracing() && throw(ErrorException(
        "no compiled `:$name` kernel for (T=$T, dim=$dim), and it cannot be built during tracing — " *
        "compiling it runs eager device work, which the tracer would capture. Run `$tune` before " *
        "compiling (any lengths: a call runs the closest tuned one)."))
    @info "Tuning `:$name` for dim=$dim, T=$T at lengths $ls (the call has n=$n; it runs the closest). " *
          "This runs once and is cached; call `$tune` ahead of time to avoid it here, and " *
          "`set_tune_lengths!` to change the default lengths."
    tune!(k, ls; dim, T)
    return nothing
end

# Each launch's arguments come from `stage_args` (`stage_args.jl`), the one place they are written.

function KernelDispatch.forward(op::SDPA, k::FlashScalar, q0, k0, v0, mask, bias)
    q, kk, v, m, b, d = _sdpa_prep(q0, k0, v0, mask, bias, true)
    a = (; o=OutArray(d.T, _full(d)), lse=OutArray(Float32, _res(d)), q, k=kk, v, bias_t=_pair_t(b, d), mask=m)
    o, lse = call_binary(op, k, :fwd, stage_args(k, Val(:fwd), a, _scalars(d))...;
        key=_key(d), extent=_extent(d))
    return _finish(o, lse, q0)
end

function KernelDispatch.forward(op::SDPA, k::FlashMMA, q0, k0, v0, mask, bias)
    T = eltype(q0)
    Tc = mma_compute_type(mma_backend(), T)
    (Tc === nothing || !mma_dim_runs(size(q0, 1))) && return _mma_fallback_forward(op, q0, k0, v0, mask, bias)
    if Tc !== T
        o, lse = _mma_forward(op, k, _narrow(Tc, q0), _narrow(Tc, k0), _narrow(Tc, v0), mask, _narrow(Tc, bias))
        return _widen(T, o), lse
    end
    return _mma_forward(op, k, q0, k0, v0, mask, bias)
end

function _mma_forward(op, k, q0, k0, v0, mask, bias)
    q, kk, v, m, b, d = _sdpa_prep(q0, k0, v0, mask, bias, true)
    a = (; o=OutArray(d.T, _full(d)), lse=OutArray(Float32, _res(d)), q, k=kk, v, bias=b, mask=m)
    o, lse = call_binary(op, k, :fwd, stage_args(k, Val(:fwd), a, _scalars(d))...;
        key=_key(d), extent=_extent(d))
    return _finish(o, lse, q0)
end

function KernelDispatch.backward(op::SDPA, k::FlashScalar, (q0, k0, v0, mask, bias), (o, lse), (dO, _))
    q, kk, v, m, b, d = _sdpa_prep(q0, k0, v0, mask, bias, true)
    o, dO, lse = reshape(o, _full(d)), _store(d.T, reshape(dO, _full(d))), reshape(lse, _res(d))
    s, key, extent = _scalars(d), _key(d), _extent(d)
    a = (; q, k=kk, v, bias=b, bias_t=_pair_t(b, d), mask=m, dO, lse, delta=_delta(o, dO),
        dq=OutArray(d.T, _full(d)), dk=OutArray(d.T, _full(d)), dv=OutArray(d.T, _full(d)),
        dbias_t=OutArray(Float32, d.hasbias ? (d.Nq, d.Nq, d.H * d.B) : (1,); zero=true))
    dq, dbt = call_binary(op, k, :dq, stage_args(k, Val(:dq), a, s)...; key, extent)
    dk, = call_binary(op, k, :dk, stage_args(k, Val(:dk), a, s)...; key, extent)
    dv, = call_binary(op, k, :dv, stage_args(k, Val(:dv), a, s)...; key, extent)
    db = d.hasbias ? _as(permutedims(dbt, (2, 1, 3)), bias) : nothing
    return (_as(dq, q0), _as(dk, k0), _as(dv, v0), nothing, db)
end

function KernelDispatch.backward(op::SDPA, k::FlashMMA, (q0, k0, v0, mask, bias), (o, lse), (dO, _))
    T = eltype(q0)
    Tc = mma_compute_type(mma_backend(), T)
    (isnothing(Tc) || !mma_dim_runs(size(q0, 1))) &&
        return _mma_fallback_backward(op, (q0, k0, v0, mask, bias), (o, lse), (dO, nothing))
    if Tc !== T
        dq, dk, dv, _, db = _mma_backward(op, k, (_narrow(Tc, q0), _narrow(Tc, k0), _narrow(Tc, v0), mask,
            _narrow(Tc, bias)), (_narrow(Tc, o), lse), (_narrow(Tc, dO), nothing))
        return (_widen(T, dq), _widen(T, dk), _widen(T, dv), nothing, db === nothing ? nothing : _widen(eltype(bias), db))
    end
    return _mma_backward(op, k, (q0, k0, v0, mask, bias), (o, lse), (dO, nothing))
end

function _mma_backward(op, k, (q0, k0, v0, mask, bias), (o, lse), (dO, _))
    q, kk, v, m, b, d = _sdpa_prep(q0, k0, v0, mask, bias, true)
    o, dO, lse =reshape(o, _full(d)), _store(d.T, reshape(dO, _full(d))), reshape(lse, _res(d))
    s, key, extent = _scalars(d), _key(d), _extent(d)
    a = (; q, k=kk, v, bias=b, mask=m, dO, lse, delta=_delta(o, dO),
        dq=OutArray(d.T, _full(d)), dk=OutArray(d.T, _full(d)), dv=OutArray(d.T, _full(d)),
        dbias=OutArray(Float32, d.hasbias ? (d.Nq, d.Nq, d.H, d.B) : (1,); zero=true))
    dk, dv, db = call_binary(op, k, :dkv, stage_args(k, Val(:dkv), a, s)...; key, extent)
    dq, = call_binary(op, k, :dq, stage_args(k, Val(:dq), a, s)...; key, extent)
    return (_as(dq, q0), _as(dk, k0), _as(dv, v0), nothing, d.hasbias ? _as(db, bias) : nothing)
end

# Where a device's matrix units do not take a storage type (CUDA: Float32, whose tensor-core operand would be TF32,
# too coarse to use silently), `:flash_mma` computes the call at the type they do take (`mma_compute_type`):
# the floating-point inputs are cast down, saturating, and the results cast back to the caller's types. Float32 data
# on an NVIDIA GPU is therefore rounded to Float16 INSIDE this kernel (relative error ~1e-3, like Float16 storage),
# accumulated in Float32 as always. `:default` is the kernel that keeps Float32 exact. A type with no such
# fallback type (Float64, or a GPU without tensor cores) runs `:default` instead.
#
# SATURATING because a plain cast sends |x| > floatmax(Float16) = 65504 to ±Inf, and a Float32 bias carries
# masking values like -1e9: saturated they still mean "masked" and stay finite, so a fully masked row stays uniform.
_narrow(::Type, ::Nothing) = nothing
_narrow(::Type{Tc}, x::AbstractArray) where {Tc} =
    eltype(x) <: AbstractFloat && eltype(x) !== Tc ? Tc.(clamp.(x, -floatmax(Tc), floatmax(Tc))) : x
_widen(::Type, ::Nothing) = nothing
_widen(::Type{T}, x::AbstractArray) where {T} = eltype(x) === T ? x : T.(x)

function _mma_fallback_forward(op, args...)
    @info "`:flash_mma` cannot run this storage type or head dimension on this device; running `:default`" maxlog = 1
    return KernelDispatch.forward(op, FlashScalar(), args...)
end
function _mma_fallback_backward(op, args...)
    @info "`:flash_mma` cannot run this storage type or head dimension on this device; running `:default`" maxlog = 1
    return KernelDispatch.backward(op, FlashScalar(), args...)
end

# --- this package's kernel-selection API ----------------------------------------------------------

"""
    use_kernel!(name)

Select the attention kernel for the rest of the session (`use_kernel!(SDPA, name)`). The selection is a
method, so it takes effect from the NEXT top-level expression: an attention call later in the same
function or block throws instead of running the previous kernel. To run one call with a particular
kernel, call it directly: `KernelDispatch.forward(SDPA(), LuxTriangleAttention.FlashMMA(), q, k, v, mask, bias)`.
"""
use_kernel!(name::Symbol) = KernelDispatch.use_kernel!(SDPA, name)
"""`current_kernel()`: the selected attention kernel."""
current_kernel() = something(KernelDispatch.current_kernel(SDPA), RESERVED_KERNEL_NAME)
"""`list_kernels()`: the attention kernels, with their variant counts and the selected one."""
list_kernels() = KernelDispatch.list_kernels(SDPA)

_attention_variant_key(k::BuiltinKernel, ::Type{T}, dim, n; hasbias=true, hasmask=true) where {T} =
    (nameof(T), Int(dim), Int(n), hasbias, hasmask)

function _register_builtin_kernels!()
    KernelDispatch.register_kernel!(SDPA(), FlashScalar())
    KernelDispatch.register_kernel!(SDPA(), FlashMMA())
    return nothing
end
