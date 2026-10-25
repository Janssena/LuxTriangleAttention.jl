# ---------------------------------------------------------------------------------------------
# Configurations, binaries, and the registry entry
# ---------------------------------------------------------------------------------------------

"""
    FLASH_MMA_NAME

The registry name of the matrix-unit kernel. Like `:$(RESERVED_KERNEL_NAME)` it is built by this
package ([`tune_flash_mma!`](@ref)), never registered from user files.
"""
const FLASH_MMA_NAME = :flash_mma

"""The stages of `:flash_mma`, in launch order: the forward, then `:dkv` and `:dq`."""
const FLASH_MMA_STAGES = (:fwd, :dkv, :dq)

flash_mma_kernel(stage::Symbol) =
    stage === :fwd ? flash_mma_fwd! :
    stage === :dkv ? flash_mma_bwd_kv! :
    stage === :dq ? flash_mma_bwd_q! :
    throw(ArgumentError("no `:$FLASH_MMA_NAME` stage `:$stage`; stages are $(FLASH_MMA_STAGES)"))

"""
    mma_argtypes(stage, T, dim, p, hasbias, hasmask, backend::KernelBackend) -> Tuple

The argument types of one `:flash_mma` stage after KernelAbstractions' context, for `KernelDispatch.ka_compile`
(see [`kernel_argtypes`](@ref)): the buffers in role order, the scale and the four `Int32`s, then the
compile-time parameters, `Val{typeof(backend)}` first (the tag the fragment hooks dispatch on), `Val{PE}` last (the
second products' operand type, `cfg.ops`; `Float32` when the point has no `ops`).
"""
function mma_argtypes(stage::Symbol, ::Type{T}, dim::Integer, cfg, hb::Bool, hm::Bool,
    backend::KernelBackend) where {T}
    A(x) = AbstractArray{x}
    bufs = stage === :fwd ? (A(T), A(Float32), A(T), A(T), A(T), A(T), A(Bool)) :
           stage === :dkv ? (A(T), A(T), A(Float32), A(T), A(T), A(T), A(T), A(Bool), A(T), A(Float32), A(Float32)) :
           stage === :dq ? (A(T), A(T), A(T), A(T), A(T), A(Bool), A(T), A(Float32), A(Float32)) :
           throw(ArgumentError("no `:$FLASH_MMA_NAME` stage `:$stage`; stages are $(FLASH_MMA_STAGES)"))
    return (bufs..., Float32, Int32, Int32, Int32, Int32, Val{typeof(backend)}, Val{T}, Val{dim}, Val{cfg.nsg},
        Val{cfg.r}, Val{cfg.tile}, Val{hb}, Val{hm}, Val{mma_ops(cfg)})
end

"""`mma_ops(p)`: the operand type of the second matrix products at point `p` (its `ops`, default `Float32`)."""
mma_ops(p) = get(p, :ops, Float32)

# A `:flash_mma` configuration is a parameter point `(; nsg, r, tile, ops)`: `nsg` simdgroups per
# group, `r` fragments of `FM` rows per simdgroup (a group covers `FM·r·nsg` rows), `tile` keys (or, for `:dkv`,
# queries) per staged tile, and `ops` the second products' operand type (`mma_ops`).

# The loaded device backend; `CPUBackend()` when there is none (a host-only machine, the mock-device tests), whose
# fragment shape and operand types are the defaults.
mma_backend() = try
    KernelDispatch.device_backend()
catch
    CPUBackend()
end

# The fragment shape of that backend.
mma_frag_shape() = _mma_shape(mma_backend())

"""
    mma_operand_types(be, T) -> Vector{DataType}

The operand types `ops` of the second matrix products the backend can run for storage type `T`: `Float32` and, for
16-bit storage, `T` itself (the default). A backend extension narrows it; an empty list means `:flash_mma` has no
configuration for `T` there (CUDA, Float32).
"""
mma_operand_types(be, ::Type{T}) where {T} = T === Float32 ? DataType[Float32] : DataType[Float32, T]

"""
    mma_compute_type(be, T) -> Union{DataType,Nothing}

The storage type `:flash_mma` computes a call of storage type `T` at on this device: `T` itself (the default, Metal), or the
type its matrix units take instead, to which the call's inputs are cast (saturating) and from which the results are cast back
(CUDA: Float32 runs at Float16); `nothing` when there is none, and the call runs `:default`. See `_narrow` in `sdpa_op.jl`.
"""
mma_compute_type(be, ::Type{T}) where {T} = T

mma_rows(c) = first(mma_frag_shape()) * c.r * c.nsg
mma_threads(c) = 32 * c.nsg

"""
    mma_shmem(stage, dim, T, cfg) -> Int

Local memory one group of `stage` uses at `cfg`, in bytes (see each kernel's `@localmem`).
"""
function mma_shmem(stage::Symbol, dim::Integer, ::Type{T}, c) where {T}
    st, sp, rows, b = sizeof(T), sizeof(mma_ops(c)), mma_rows(c), c.tile   # sp: the `PE` operand tiles
    stage === :fwd && return st * (rows * dim + dim * b) + sp * b * dim + 4b
    stage === :dkv && return st * (2 * dim * b + rows * dim) + sp * 2 * dim * b + 8b + 4dim
    stage === :dq && return st * (rows * dim + 2 * dim * b) + sp * dim * b + 4b
    throw(ArgumentError("unknown stage `:$stage`"))
end

"""
    flash_mma_source_tag() -> String

Short hash of `forward.jl` and `backward.jl`, in every `:flash_mma` binary's name and in its registry manifest, so an
edited kernel is rebuilt rather than served from a stale binary (see [`kernel_source_tag`](@ref)).
"""
flash_mma_source_tag() = FLASH_MMA_SOURCE_TAG
const FLASH_MMA_SOURCE_TAG = let h = UInt64(0)
    for f in ("forward.jl", "backward.jl")
        h = hash(read(joinpath(@__DIR__, f), String), h)
    end
    string(h % UInt32; base=36)
end

# ---------------------------------------------------------------------------------------------
# What is tuned (the tuning itself is the shared skeleton, `src/kernels/tune.jl`)
# ---------------------------------------------------------------------------------------------
#
# Measured on an M1 Pro (docs/done/FEATURE_flash_mma.md), against tuning every geometry separately:
# at most 11% slower (mean ~1%) over triangle, MSA-row and plain-sequence shapes. The length matters:
# a configuration covering 256 rows per group is the fastest at N = 256 and 1.85× slower than the best
# at N = 384, where a third of its rows are padding — hence tuning per length.

tune_stages(::FlashMMA) = FLASH_MMA_STAGES

"""
    tune_grid(::FlashMMA, stage, dim, T, caps)

2–16 simdgroups per group, one or two 8-row blocks per simdgroup, staged tiles of 16–64, and for
16-bit storage the operand type of the second matrix products, `ops`: `Float32` or the storage type
(see "Precision" in `forward.jl`). Each candidate is verified, so a 16-bit choice that is too
inaccurate, or whose Float16 `dS` overflows on a large `dO` (`DO_SCALE`), is never picked.
"""
function tune_grid(::FlashMMA, stage::Symbol, dim::Integer, ::Type{T}, caps::DeviceCaps) where {T}
    stage in FLASH_MMA_STAGES || throw(ArgumentError("unknown `:$FLASH_MMA_NAME` stage `:$stage`"))
    ops = mma_operand_types(mma_backend(), T)
    return (; nsg=[2, 4, 8, 16], r=[1, 2], tile=[16, 32, 64], ops)
end

"""`admissible(::FlashMMA, stage, dim, T, caps, p)`: within the device's thread cap and local memory."""
admissible(::FlashMMA, stage::Symbol, dim::Integer, ::Type{T}, caps::DeviceCaps, p) where {T} =
    mma_threads(p) <= caps.max_threads && mma_shmem(stage, dim, T, p) <= caps.max_shmem

# How a point becomes a binary (`launch.jl`'s `build_stage`). The backend tag selects the fragment
# hooks, so it is the loaded backend's.
stage_kernel(::FlashMMA, stage::Symbol) = flash_mma_kernel(stage)
stage_argtypes(::FlashMMA, stage::Symbol, ::Type{T}, dim::Int, p, hb::Bool, hm::Bool) where {T} =
    mma_argtypes(stage, T, dim, p, hb, hm, KernelDispatch.device_backend())
stage_threads(::FlashMMA, p) = mma_threads(p)
stage_rows(::FlashMMA, p) = mma_rows(p)

# --- the public names --------------------------------------------------------------------------------

"""
    calibrate_mma!([T=Float32]; dim, n=first(tune_lengths()), force=false) -> Dict(stage => Dict(point => seconds))

[`calibrate!`](@ref)`(FlashMMA(), T; dim, n, force)`: measure `:$FLASH_MMA_NAME` on this device for
`(dim, T)` at length `n` and persist it.
"""
calibrate_mma!(::Type{T}=Float32; dim::Integer, n::Integer=first(tune_lengths()), force::Bool=false) where {T} =
    calibrate!(FlashMMA(), T; dim, n, force)

"""`flash_mma_config(T, dim, n) -> Dict(stage => (; nsg, r, tile))`: what [`tune_flash_mma!`](@ref) picks."""
flash_mma_config(::Type{T}, dim::Integer, n::Integer) where {T} = tuned_config(FlashMMA(), T, dim, n)

"""
    tune_flash_mma!(lengths=tune_lengths(); dim, T=Float32, force=false) -> variants

[`tune!`](@ref)`(FlashMMA(), lengths; dim, T, force)`: tune, compile and register the matrix-unit kernel
`:$FLASH_MMA_NAME` at each length in `lengths`; select it with `use_kernel!(:$FLASH_MMA_NAME)`. A call of
any length runs the closest tuned one; an eager call with none tunes [`tune_lengths`](@ref) lazily, a
traced one refuses. `dim` must be a multiple of the device's fragment contraction width (8 on Metal, 16 on NVIDIA).
Where the device computes `T` at another type (Float32 at Float16 on CUDA, see `mma_compute_type`) that type is tuned; where
it has none, a call runs `:default` and this returns without tuning.
"""
function tune_flash_mma!(lengths=tune_lengths(); dim::Integer, T::Type=Float32, force::Bool=false)
    device_kernel_available() || throw(ErrorException(
        "no usable device backend is loaded, so `:$FLASH_MMA_NAME` cannot be compiled. Load the " *
        "package for your GPU (e.g. `using Metal`) first."))
    dim % last(mma_frag_shape()) == 0 || throw(ArgumentError(
        "`:$FLASH_MMA_NAME` needs a head dimension that is a multiple of $(last(mma_frag_shape())) on this device; got $dim"))
    Tc = mma_compute_type(mma_backend(), T)       # the type a call of storage type `T` runs at
    if Tc === nothing
        @info "`:$FLASH_MMA_NAME` cannot run $T on this device; nothing to tune, `:default` runs for it" maxlog = 1
        return KernelDispatch.variants(SDPA(), FlashMMA())
    end
    return tune!(FlashMMA(), lengths; dim, T=Tc, force)
end
