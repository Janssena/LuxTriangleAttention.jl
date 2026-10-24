# Which GPU backend, if any, the `gpu` units of the suite run on. As in KernelDispatch.jl's test suite.
#
# Candidates are tried in order, Metal (Apple only) then CUDA, and the first whose package is in the test environment,
# loads, reports a functional device AND has an adapter, `test/ext/<backend>/backend.jl`, wins. Nothing else is loaded,
# so a run without `gpu` units never touches a GPU package.
#
#     LTA_TEST_GPU=none    skip the gpu units even on a GPU machine
#     LTA_TEST_GPU=metal   require Metal (error if it cannot be used);  likewise `cuda`
#
# Defines `GPU` (`:metal`, `:cuda` or `nothing`) and, when there is one, `GPU_PKG` (the package) and the adapter's names
# (see `test/ext/metal/backend.jl`). Reactant is loaded too, so that the gpu files can trace with its macros; a machine
# without a traced client skips their traced parts.

const _GPU_PACKAGES = (metal=:Metal, cuda=:CUDA)

# -> (backend::Union{Symbol,Nothing}, module or nothing, why-not::String)
function _detect_gpu()
    want = lowercase(get(ENV, "LTA_TEST_GPU", ""))
    want == "none" && return (nothing, nothing, "LTA_TEST_GPU=none")
    want in ("", "metal", "cuda") || error("LTA_TEST_GPU must be none, metal or cuda; got `$want`")
    names = isempty(want) ? collect(keys(_GPU_PACKAGES)) : [Symbol(want)]
    why = String[]
    for name in names
        pkg = _GPU_PACKAGES[name]
        if name === :metal && !Sys.isapple()
            push!(why, "Metal: not an Apple machine")
        elseif !isfile(joinpath(@__DIR__, "..", "ext", String(name), "backend.jl"))
            push!(why, "$pkg: no adapter in test/ext/$name/")
        elseif (mod = _load(pkg)) === nothing
            push!(why, "$pkg: not in the test environment, or failed to load")
        elseif !(try
                Base.invokelatest(() -> mod.functional())
            catch
                false
            end)
            push!(why, "$pkg: no functional device")
        elseif !Base.invokelatest(KernelDispatch.device_available)
            push!(why, "$pkg: functional, but KernelDispatch's extension did not register the device")
        else
            return (name, mod, "")
        end
    end
    isempty(want) || error("LTA_TEST_GPU=$want requested, but: $(join(why, "; "))")
    return (nothing, nothing, join(why, "; "))
end

# `GPU` is defined by the probe, which can run in the same top-level expression that then reads it.
gpu_backend() = Base.invokelatest(getglobal, Main, :GPU)

function load_gpu_backend!()
    gpu, mod, why = _detect_gpu()
    Core.eval(Main, :(const GPU = $(QuoteNode(gpu))))
    if gpu === nothing
        @info "no GPU backend: the gpu units are skipped" reason = why
        return false
    end
    Core.eval(Main, :(const GPU_PKG = $mod))
    _load(:Reactant)                               # the gpu files trace with its macros
    @info "GPU backend for the gpu units" backend = gpu
    include(joinpath(@__DIR__, "..", "ext", String(gpu), "backend.jl"))
    return true
end
