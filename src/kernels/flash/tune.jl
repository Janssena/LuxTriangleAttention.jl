# What is tuned for `:default` (`FlashScalar`), the scalar flash kernel. The tuning itself — search,
# verification, timing, calibration, registration — is the shared skeleton in `src/kernels/tune.jl`.
#
# Every stage — `:fwd`, `:dq`, `:dk`, `:dv` — gets its OWN configuration: the group size `tg`, the
# key-tile size `kt`, and for the forward the softmax sub-block `sb`. They differ because the kernels'
# register footprints differ (`2*DIM` floats per thread for `:fwd`/`:dv`, `3*DIM` for `:dq`/`:dk`, see
# `backward.jl`), and a device's register ceiling is not something to declare: it is measured, as
# what makes an over-full configuration slow or undispatchable. `sb` is the forward's one knob that
# trades registers for arithmetic (one softmax rescale per `sb` keys instead of per key); on an M1 Pro
# `sb = 4` wins and `sb = 8` spills, and a device with a larger register file can be expected to want
# more. One thread per query, so a group covers `tg` queries.

"""
    FLASH_STAGES

The stages of the built-in kernel, in launch order: the forward, then the three backward passes.
"""
const FLASH_STAGES = (:fwd, :dq, :dk, :dv)

tune_stages(::FlashScalar) = FLASH_STAGES

"""
    tune_grid(::FlashScalar, stage, dim, T, caps)

Group sizes that are multiples of the SIMD width up to the device maximum, key tiles of 16–128, and
for the forward a softmax sub-block of 1–8. Independent of `n` by construction.
"""
function tune_grid(::FlashScalar, stage::Symbol, dim::Integer, ::Type{T}, caps::DeviceCaps) where {T}
    stage in FLASH_STAGES || throw(ArgumentError("unknown `:default` stage `:$stage`"))
    tg = filter(<=(caps.max_threads), caps.warp .* [1, 2, 4, 8, 16, 32])
    kt = [16, 32, 64, 128]
    return stage === :fwd ? (; tg, kt, sb=[1, 2, 4, 8]) : (; tg, kt)
end

"""
    admissible(::FlashScalar, stage, dim, T, caps, p)

What is decidable without compiling: the staged tiles fit in local memory, and `sb` divides `kt` (the
forward walks each tile in whole sub-blocks). Whether the COMPILED pipeline admits `tg` threads —
register pressure drives that well below `caps.max_threads`: for the forward at dim=32 on an M1 Pro,
384 against 1024 — is checked per candidate, since answering it means compiling. `T` changes the space:
the tiles are held in storage precision, so a narrower `T` legalises larger `kt`.
"""
admissible(::FlashScalar, stage::Symbol, dim::Integer, ::Type{T}, caps::DeviceCaps, p) where {T} =
    (!haskey(p, :sb) || (p.sb <= p.kt && p.kt % p.sb == 0)) &&
    stage_shmem(stage, dim, T, p) <= caps.max_shmem

"""
    stage_shmem(stage, dim, T, p) -> Int

Local memory, in bytes, one group of `stage` uses: two staged `[dim, kt]` tiles in the storage type
plus the per-key `Float32` vectors (the forward's and `:dq`'s staged mask, `:dk`'s `lse`+`delta`, `:dv`'s
`lse`).
"""
function stage_shmem(stage::Symbol, dim::Integer, ::Type{T}, p) where {T}
    tiles = 2 * dim * p.kt * sizeof(T)
    stage === :dk && return tiles + 2 * 4 * p.kt
    return tiles + 4 * p.kt
end

# How a point becomes a binary (`launch.jl`'s `build_stage`).
stage_kernel(::FlashScalar, stage::Symbol) = stage_kernel(stage)
stage_argtypes(::FlashScalar, stage::Symbol, ::Type{T}, dim::Int, p, hb::Bool, hm::Bool) where {T} =
    kernel_argtypes(stage, T, dim, p, hb, hm)
stage_threads(::FlashScalar, p) = p.tg
stage_rows(::FlashScalar, p) = p.tg

# --- the public names --------------------------------------------------------------------------------

"""
    calibrate!([T=Float32]; dim, n=first(tune_lengths()), force=false) -> Dict(stage => Dict(point => seconds))

[`calibrate!`](@ref)`(FlashScalar(), T; dim, n, force)`: measure `:$(RESERVED_KERNEL_NAME)` on this
device for `(dim, T)` at length `n` and persist it, so a later call is a file read. Run it
deliberately: it compiles, verifies and times candidate configurations for all four stages.
"""
calibrate!(::Type{T}=Float32; dim::Integer, n::Integer=first(tune_lengths()), force::Bool=false) where {T} =
    calibrate!(FlashScalar(), T; dim, n, force)

"""
    tune_flash_kernel!(lengths=tune_lengths(); dim, T=Float32, force=false) -> variants

[`tune!`](@ref)`(FlashScalar(), lengths; dim, T, force)`: tune, compile and register this package's own
flash kernel `:$(RESERVED_KERNEL_NAME)` at each sequence length in `lengths` (an integer or a
collection; default [`tune_lengths`](@ref)). A call of any length runs the closest. Requires a backend
extension.
"""
function tune_flash_kernel!(lengths=tune_lengths(); dim::Integer, T::Type=Float32, force::Bool=false)
    device_kernel_available() || throw(ErrorException(
        "no usable device backend is loaded, so the built-in flash kernel cannot be compiled. " *
        "Load the package for your GPU (e.g. `using Metal`) first."))
    return tune!(FlashScalar(), lengths; dim, T, force)
end

"""
    flash_config(T, dim, n) -> Dict(stage => (; tg, kt[, sb]))

The configuration each stage of the `:$(RESERVED_KERNEL_NAME)` variant tuned at length `n` was (or
would be) compiled with, from the calibration exactly as [`tune_flash_kernel!`](@ref) chose it.
"""
flash_config(::Type{T}, dim::Integer, n::Integer) where {T} = tuned_config(FlashScalar(), T, dim, n)
