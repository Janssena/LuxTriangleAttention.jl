# The recorded outputs of the Python references, which the parity sweeps (support/parity.jl) compare with.
#
# Running the references takes PyCall and torch, which a test run should not need. So their outputs are recorded once,
# one file per `<component>.<model>` in `test/fixtures/python/`, and the sweeps read them back. The files are the
# `python_fixtures` lazy artifact (test/Artifacts.toml): downloaded when the first parity unit runs, never in a run that
# has none. `LTA_FIXTURE_DIR` reads (and records) in another directory, e.g. `test/fixtures/python` for the checkout's own
# files, before a changed set is published.
#
# `LTA_GENERATE_FIXTURES=true` records them again: the sweeps then run Python (test/python/references.jl) for each case
# and write what it returned. See test/python/README.md.
#
# A case is a tensor `<key>/y` (what Python returned) and `<key>/fingerprint` (what the inputs summed to when it was
# recorded). The inputs are not stored, the sweeps draw them from a seeded generator: if that changes (a new Julia) the
# recorded outputs are of other numbers than the Julia layer gets, and the fingerprint says so instead of the comparison
# failing in a way that looks like a bug in the layer.
#
# The weights are stored (`weights/<layout>/<path>`), not drawn by `Lux.setup`, whose initializers make every bias zero and
# every LayerNorm bias zero (and may change): zeros let a layer that ignores its bias pass. They are random, none of them
# zero (`random_weights`), and `weights` hands them to the sweeps in the shape `Lux.setup` makes.

using Artifacts, LazyArtifacts
import SafeTensors

const GENERATE_FIXTURES = lowercase(get(ENV, "LTA_GENERATE_FIXTURES", "false")) in ("true", "1", "yes")

"The directory of the recorded references: where they are written when generating, and read from otherwise."
fixture_dir() = get(ENV, "LTA_FIXTURE_DIR") do
    GENERATE_FIXTURES ? joinpath(@__DIR__, "..", "fixtures", "python") : artifact"python_fixtures"
end

"""
    fingerprint(args...) -> Vector{Float64}

The sum of the absolute values of every array in `args` (searching tuples and named tuples), in order. Small enough to
store and compared with `isapprox`: it changes with any value that does.
"""
fingerprint(args...) = Float64[sum(abs, Float64.(a)) for arg in args for a in _arrays(arg)]

_arrays(x::AbstractArray{<:Number}) = (x,)
_arrays(x::Union{Tuple,NamedTuple}) = Iterators.flatten(_arrays(v) for v in x)
_arrays(::Any) = ()

"""
    References(component, model)

The references of one parity unit: the recorded file when running the tests, the Python wiring (`PYTHON_REFERENCES`) and
what it returned so far when generating.
"""
struct References
    path::String
    recorded::Union{Nothing,Any}            # the (lazily read) file; nothing when generating
    wiring::Union{Nothing,NamedTuple}       # the Python side; nothing when reading
    out::Dict{String,Array}
    weights::Dict{String,Any}               # the parameters handed out so far, by layout
    rng::Random.Xoshiro                     # draws the weights when generating
end

function References(component::AbstractString, model::AbstractString)
    path = joinpath(fixture_dir(), "$component.$model.safetensors")
    rng = Random.Xoshiro(2024)
    if GENERATE_FIXTURES
        return References(path, nothing, PYTHON_REFERENCES[(component, model)], Dict{String,Array}(), Dict{String,Any}(), rng)
    end
    isfile(path) || error("no recorded Python references at $path; record them with LTA_GENERATE_FIXTURES=true (test/python/README.md)")
    return References(path, SafeTensors.deserialize(path), nothing, Dict{String,Array}(), Dict{String,Any}(), rng)
end

"""
    weights(refs, layout, ps) -> NamedTuple

The parameters the Python reference of the layout `layout` was run with, in the structure and `Float32` of `ps` (a
`Lux.setup` result, which says what the layer holds). Generating draws them, once per layout (`random_weights`) and keeps
them for `save`; reading takes the recorded ones. None is zero.
"""
function weights(refs::References, layout::AbstractString, ps)
    get!(refs.weights, layout) do
        map_leaves(ps, "weights/$layout") do name, leaf
            if refs.wiring !== nothing
                w = random_weights(refs.rng, name, leaf)
                refs.out[name] = w
            else
                haskey(refs.recorded, name) || error("no recorded weights `$name` in $(refs.path); record them again with LTA_GENERATE_FIXTURES=true")
                w = collect(refs.recorded[name])
            end
            size(w) == size(leaf) && eltype(w) === eltype(leaf) ||
                error("the recorded weights `$name` are $(eltype(w)) $(size(w)), the layer's $(eltype(leaf)) $(size(leaf)); record them again with LTA_GENERATE_FIXTURES=true")
            any(iszero, w) && error("the weights `$name` contain zeros, which hide a layer that ignores them")
            return w
        end
    end
end

# `f(name, leaf)` on every array of a (nested) named tuple, rebuilt in its structure; `name` is the path joined by `/`.
map_leaves(f, x::NamedTuple, name) = NamedTuple{keys(x)}(map(k -> map_leaves(f, x[k], "$name/$k"), keys(x)))
map_leaves(f, x::AbstractArray{<:Number}, name) = f(name, x)
map_leaves(f, x, name) = x

# Weights of the shape of `leaf`, each of magnitude 0.5 to 1.5 times a scale, so none is near zero: a LayerNorm `scale`
# around 1, a matrix `1/√fan_in` (what keeps the activations of a layer near 1), a bias 0.1, with random signs apart
# from the `scale`.
function random_weights(rng, name::AbstractString, leaf::AbstractArray{T}) where {T}
    kind = last(split(name, "/"))
    scale = kind == "scale" ? 1 : ndims(leaf) ≥ 2 ? 1 / sqrt(size(leaf, 2)) : 0.1
    signs = kind == "scale" ? ones(T, size(leaf)) : T.(2 .* rand(rng, Bool, size(leaf)) .- 1)
    return T(scale) .* (T(0.5) .+ rand(rng, T, size(leaf))) .* signs
end

"""
    reference(f, refs, key, fingerprint) -> Array

The Python output of the case `key`. When reading, the recorded one, after checking that the case was recorded for the
inputs and weights `fingerprint` describes; when generating, `f(wiring)`, which runs Python and returns an array, and is
kept for `save`.
"""
function reference(f, refs::References, key::AbstractString, fp::Vector{Float64})
    if refs.wiring !== nothing
        y = collect(f(refs.wiring))
        refs.out["$key/y"] = y
        refs.out["$key/fingerprint"] = fp
        return y
    end
    haskey(refs.recorded, "$key/y") || error("no recorded Python reference `$key` in $(refs.path); record them again with LTA_GENERATE_FIXTURES=true")
    isapprox(collect(refs.recorded["$key/fingerprint"]), fp; rtol=1e-9) ||
        error("the recorded Python reference `$key` is of other inputs or weights than this run draws (a new Julia or Lux?); " *
              "record them again with LTA_GENERATE_FIXTURES=true")
    return collect(refs.recorded["$key/y"])
end

"Write what `reference` collected, when generating."
function save(refs::References)
    refs.wiring === nothing && return nothing
    mkpath(dirname(refs.path))
    SafeTensors.serialize(refs.path, refs.out)
    @info "recorded $(length(refs.out) ÷ 2) Python references" path = refs.path
    return nothing
end
