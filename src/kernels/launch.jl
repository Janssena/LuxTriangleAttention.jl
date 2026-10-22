# Building and launching the built-in kernels' stages outside the registry: what the tuner (and the
# kernel tests) run. Backend-agnostic: binaries come from `KernelDispatch.ka_compile` for the loaded
# device backend, launches from `KernelDispatch.prepare_launch`/`execute`, their arguments from
# `stage_args` — the same list the op's `forward`/`backward` bind.
#
# A kernel says how a parameter point becomes a binary with four methods (in its `tune.jl`):
# `stage_kernel(k, stage)`, `stage_argtypes(k, stage, T, dim, p, hasbias, hasmask)`,
# `stage_threads(k, p)` and `stage_rows(k, p)` (queries per group: the binary's `tile`).

# Where candidates are compiled. A tuner builds every configuration it measures, and only the
# WINNERS are worth keeping: `add_variant!` copies those into the registry. So candidates go into a
# scratch directory of this session, deleted at exit, rather than a cache that grows with every
# configuration ever measured. Within a session `ka_compile` rebuilds only absent files, so the
# binary's NAME is the cache key. `ka_compile` appends the tag of every argument type and compile-time
# `Val` to what is given here (`_f16x4_f32_..._vFloat16_v32_v16`: the storage type, head dimension, tile
# sizes, `hasbias`/`hasmask`), so a build at another type or configuration never shares a file; the name
# says which stage of which kernel it is, and from which source.
const _STAGING = Ref("")
_staging_dir() = (isdir(_STAGING[]) || (_STAGING[] = mktempdir()); _STAGING[])

# The tuned point as the binary's `params` (what its manifest keeps): a type is its name, since a manifest is TOML.
_stored_params(p::NamedTuple) = map(v -> v isa Type ? String(nameof(v)) : v, p)

"""
    build_stage(kernel, stage, T, dim, p; hasbias=true, hasmask=true) -> KernelDispatch.KernelBinary

Compile one stage of a built-in kernel at the parameter point `p` (a NamedTuple of its tuned
parameters) for the loaded device backend, or return the binary already built this session. Its
`tile` is `(stage_rows(kernel, p), 1, 1)`, so a launch of `extent = (n, nouter, heads)` dispatches
`(cld(n, rows), nouter, heads)` groups. `hasbias = false` / `hasmask = false` select the
specialisation that never reads the bias / mask (the kernels' `HB`/`HM` flags).

`T`, `dim` and the parameters are compile-time constants, so each is a separate compilation (the
tile loops unroll against them); `n` and the head count are runtime arguments, so one binary covers
every length. Cold compilation is 0.2–0.9 s. The binary records its `signature` (the stage's
[`kernel_argtypes`](@ref) / [`mma_argtypes`](@ref) without the `Val`s) and `ka_compile` names the file
after those types, so a `Float16` build is never served for a `Float32` request, and a launch that binds
other types than it was built for is refused rather than computing garbage.
"""
function build_stage(k::BuiltinKernel, stage::Symbol, ::Type{T}, dim::Integer, p::NamedTuple;
    hasbias::Bool=true, hasmask::Bool=true) where {T}
    stag = string(stage) * (hasbias ? "" : "nb") * (hasmask ? "" : "nm")
    name = "lta_$(KernelDispatch.kernelname(k))_$(stag)_$(KernelDispatch.source_tag(k))"
    return KernelDispatch.ka_compile(KernelDispatch.device_backend(), stage_kernel(k, stage),
        stage_argtypes(k, stage, T, Int(dim), p, hasbias, hasmask);
        tg=stage_threads(k, p), name=replace(name, r"[^A-Za-z0-9_]" => "_"),
        tile=(stage_rows(k, p), 1, 1), params=_stored_params(p), dir=_staging_dir())
end

"""
    packed_problem(q, k, v, bias, mask, dO, o, lse, delta) -> NamedTuple

Device buffers for one attention problem in the packed `[dim, n, nouter, bh]` layout (`nh = 1`), by
the roles [`stage_args`](@ref) reads, plus zeroed outputs. `o`, `lse` and `delta` are INPUTS of the
backward stages here, so each stage runs against fixed residuals rather than the output of whichever
forward ran last. `bias_t`/`dbias_t` are the transposed pair-bias layouts `:default` reads/writes.
"""
function packed_problem(q, k, v, bias, mask, dO, o, lse, delta)
    be = KernelDispatch.device_backend()
    d(x) = to_device(be, x)
    n, nb = size(q, 2), size(bias, 3)
    return (; q=d(q), k=d(k), v=d(v), dO=d(dO), bias=d(bias), bias_t=d(permutedims(bias, (2, 1, 3))),
        mask=d(mask), o=d(o), lse=d(lse), delta=d(delta),
        dq=d(zero(q)), dk=d(zero(k)), dv=d(zero(v)),
        dbias=d(zeros(Float32, n, n, nb)), dbias_t=d(zeros(Float32, n, n, nb)))
end

"""
    stage_launch(kernel, stage, b, P; n, nouter, heads, mheads) -> KernelDispatch.Launch

One launch of binary `b` (stage `stage`) on a packed problem `P`, prepared but not run: its arguments
from [`stage_args`](@ref) (`nh = 1`), sized by `extent = (n, nouter, heads)`.
"""
function stage_launch(k::BuiltinKernel, stage::Symbol, b::KernelDispatch.KernelBinary, P;
    n::Integer, nouter::Integer, heads::Integer, mheads::Integer)
    dim = size(P.q, 1)
    s = (Float32(inv(sqrt(dim))), Int32(n), Int32(nouter), Int32(1), Int32(mheads))
    return KernelDispatch.prepare_launch(b, stage_args(k, Val(stage), P, s)...;
        extent=(n, nouter, heads), label=stage)
end

"""
    run_stage!(kernel, stage, b, P; n, nouter, heads, mheads) -> outputs

Run binary `b` (stage `stage`) on a packed problem `P` and return its outputs, in
[`stage_outputs`](@ref) order. The outputs are zeroed first: the bias gradients accumulate.
"""
function run_stage!(k::BuiltinKernel, stage::Symbol, b::KernelDispatch.KernelBinary, P; geom...)
    outs = map(Base.Fix1(getfield, P), stage_outputs(k, stage))
    foreach(x -> fill!(x, zero(eltype(x))), outs)
    KernelDispatch.execute(stage_launch(k, stage, b, P; geom...))
    return outs
end
