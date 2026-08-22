"""
    device_kernel_available() -> Bool

Whether a backend extension is loaded AND its device is usable. False on a host-only setup.
"""
device_kernel_available() = _DEVICE_AVAILABLE[]()

const _DEVICE_AVAILABLE = Ref{Function}(() -> false)

"""
    _DEVICE_BINARY[]

`(T, dim, tg, kt) -> (path, entry)`, installed by a backend extension.

This exists because the Reactant extension has to name a compiled binary in the `custom_call` it
emits, but it cannot ask the backend extension directly. Julia has no mechanism for one extension
to import another, and here neither is even guaranteed to be loaded. Routing through a `Ref` owned
by the package keeps both extensions depending only on `LuxTriangleAttention`.

Returns the cached binary or builds it. Left as the throwing default when no backend is loaded, so
a caller that needs it fails with a clear message instead of emitting a call to a file that does not
exist.
"""
const _DEVICE_BINARY = Ref{Function}((args...) ->
    error("no backend extension is loaded; cannot locate a compiled kernel"))

"""
    _KA_BACKEND_OBJECT[]

`() -> KernelAbstractions.Backend`, installed by a backend extension.

`KernelAbstractions.mkcontext` needs a concrete KernelAbstractions backend object (`MetalBackend()`, `CUDABackend()`
…), which only the extension that depends on that vendor package can construct. Our `KernelBackend`
tags deliberately cannot: they are a routing trait, not a handle.
"""
const _KA_BACKEND_OBJECT = Ref{Function}(() ->
    error("no backend extension is loaded; cannot construct a KernelAbstractions backend"))

"""
    kernel_source_tag() -> String

Short hash of the kernel sources, for a backend to put in its binary cache key.

The cache key that matters — `(T, dim, tg, kt)` — says what a binary was compiled FOR, not what it
was compiled FROM. Without this, editing `flash/forward.jl` leaves every previously built binary on disk
under a key that still matches, so the next run silently launches the OLD kernel: no error, no
recompile, and results that quietly belong to code that no longer exists. Only the numbers move,
which makes it a genuinely nasty way to lose an afternoon.

Computed once, when the package is precompiled, over the kernel source files — so it tracks the code
actually baked into the loaded image. It does NOT cover the toolchain (Julia, GPUCompiler,
KernelAbstractions, or the backend's own compiler), which can also change codegen; a backend that
wants to be strict about that should mix its own versions in on top.
"""
kernel_source_tag() = KERNEL_SOURCE_TAG

const KERNEL_SOURCE_TAG = let h = UInt64(0)
    for f in ("forward.jl", "backward.jl")
        h = hash(read(joinpath(@__DIR__, "flash", f), String), h)
    end
    string(h % UInt32; base=36)
end

"""
    device_binary(T, dim, tg, kt) -> (path, entry)

Path and entry-point name of the compiled kernel for one configuration, building it if absent.
"""
device_binary(::Type{T}, dim::Integer, tg::Integer, kt::Integer) where {T} =
    _DEVICE_BINARY[](T, Int(dim), Int(tg), Int(kt))

"""
    _DEVICE_BINARY_BWD[]

`(T, dim, tg, kt) -> (; dq, dk, dv)`, each a `(path, entry)` pair, installed by a backend extension.

The backward analogue of [`_DEVICE_BINARY`](@ref): `tune_flash_kernel!` and the traced backward
walk (`flash_bwd_dq!`/`flash_bwd_dk!`/`flash_bwd_dv!` in `flash/backward.jl`) both need to name THREE
compiled binaries — one per backward pass — and cannot reach the backend extension that compiled
them directly, for the same reason `_DEVICE_BINARY` cannot. Left as the throwing default when no
backend is loaded, so a caller that needs it fails loudly rather than naming a file that does not
exist.
"""
const _DEVICE_BINARY_BWD = Ref{Function}((args...) ->
    error("no backend extension is loaded; cannot locate compiled backward kernels"))

"""
    device_binary_bwd(T, dim, tg, kt) -> (; dq, dk, dv)

Paths and entry-point names of the compiled backward kernels for one configuration, building them if
absent. Each field is a `(path, entry)` pair.
"""
device_binary_bwd(::Type{T}, dim::Integer, tg::Integer, kt::Integer) where {T} =
    _DEVICE_BINARY_BWD[](T, Int(dim), Int(tg), Int(kt))

"""
    calibrate!([T=Float32]; dim, force=false)

Measure this device once and persist the result, so a later first call is a file read.

Run this deliberately, ahead of any latency-sensitive work: it compiles and times every candidate
configuration, which takes seconds. It is keyed by `(device, dim, T)` and is **not** keyed by
sequence length — the compiled binary takes `n` as a runtime argument and choosing a configuration
for a given `n` is an argmin over the cached numbers, so there is nothing to pre-tune per `n`.

`T` is the STORAGE element type; accumulation is always `Float32` (see `src/kernels/flash/forward.jl`). It is
part of the key because it changes which configurations are legal, not merely how fast they are.

Implemented by the loaded backend extension; throws if none is.
"""
function calibrate! end

"""
    _sdpa_kernel_bwd!(q, k, v, mask, bias, o, lse, dO) -> (dq, dk, dv, dbias)

The reverse of [`_sdpa_kernel`](@ref): run the selected entry's backward stages.

**Empty here on purpose.** This is a seam, not an implementation. A backend extension adds a method
for its own array type (`LuxTriangleAttentionMetalExt` for `MtlArray`, and the Reactant extension
for traced arrays), and each does the same thing: look up the kernel [`LTA_KERNEL`](@ref) names,
resolve its compiled stages for this problem, and run them in the order the entry declares.

Everything that varies between kernels — how many launches the backward takes, which gradient each
one writes, what residuals it reads — is DATA on the entry (see `KernelStage`), so a backend
implements the walk once and never needs to know which kernel it is running.

`o` and `lse` are the forward's own outputs for these operands, both required residuals: `o` builds
`delta`, and `lse` is used as-is to recompute attention weights a tile at a time. `dO` is the
incoming cotangent. There is deliberately no `dout` stage — `dO` is an input to the backward, not
something a kernel produces — and no `dbias` stage in our own entry, because our `dq` kernel writes
`dbias` alongside `dq` via atomics.

Callers are [`LuxTriangleAttentionEnzymeExt`](@ref)'s reverse rule on the concrete path, and the
`enzyme.reverse` function the Reactant extension attaches to its `stablehlo.custom_call`.
"""
function _sdpa_kernel_bwd! end

