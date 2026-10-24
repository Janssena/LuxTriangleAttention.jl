# What backends the machine can run, for `selected` (test/sections.jl) and for the files that run on one.
#
#   cpu       always.
#   gpu       a GPU backend with an adapter (test/ext/<backend>/backend.jl) is found and functional, see
#             `support/backend_gpu.jl`; the traced parts of the gpu files further need a traced client.
#   reactant  Reactant is in the test environment and loads (its default client is the CPU one).
#
# Probing loads the package, so it is only done for a backend some selected unit asks for, and only once.
# Included at top level by `test/runtests.jl`, so that the packages are visible, in world age, to the files
# included afterwards.

const _BACKEND_AVAILABLE = Dict{Symbol,Bool}(:cpu => true)

function _load(pkg::Symbol)
    Base.find_package(String(pkg)) === nothing && return nothing
    return try
        Core.eval(Main, :(using $pkg))      # binds the name in Main too, which `mps_client.jl` relies on
        Base.invokelatest(getfield, Main, pkg)      # the binding was created a moment ago, in a newer world
    catch err
        @warn "$pkg failed to load" exception = (err, catch_backtrace())
        nothing
    end
end

function _probe(b::Symbol)
    b === :reactant && return _load(:Reactant) !== nothing
    b === :gpu && return load_gpu_backend!()
    error("unknown backend $b")
end

"""
    backend_available(b::Symbol) -> Bool
"""
backend_available(b::Symbol) = get!(() -> _probe(b), _BACKEND_AVAILABLE, b)
backend_available(b::AbstractString) = backend_available(Symbol(b))
