# Which parts of the suite a run executes. Included by `test/runtests.jl` before anything else.
#
# Every test unit is tagged on up to three axes:
#
#   component  the src/ file (or ext/ file) it tests, e.g. `attention`, `flash`, `ext_metal`. A GROUP name
#              (`kernels`, `layers`, `ext`) stands for all of its components.
#   backend    what it needs to run: `cpu` (host arrays), `gpu` (whichever GPU backend the machine has, found by
#              `support/backend_gpu.jl`: eager device arrays and the GPU's traced client) or `reactant` (traced on
#              Reactant's default client). The files that need a GPU are named `<file>.gpu.test.jl`; what is specific to one
#              backend is in `test/ext/<backend>/`, which runs when that backend is the one found (`ext_metal`).
#   model      `af2`, `af3`, `boltz2`: the Python reference a parity unit compares with (on the host). Naming one of
#              these is also what loads PyCall + torch.
#
# A token on the command line filters its axis; no token on an axis leaves it unfiltered. Tokens of one axis
# are OR-ed, the axes are AND-ed: `attention af3 cpu` is the AF3 attention parity on the host. When a model
# is named, units that have no model (the Julia-only tests) are left out. Unknown tokens are an error.
#
# Tokens come from the command line (`ARGS`) or, when there are none, from `LTA_TEST` (spaces or commas).
# The second exists for `Pkg.test`: the pkg REPL's `test` command takes no arguments, but the test process
# inherits the environment.
#
# A unit whose backend is selected but not available (no functional GPU) is skipped with a warning, never failed.
# `LTA_TEST_GPU=none|metal|cuda` overrides the GPU backend choice.

const COMPONENT_GROUPS = [
    "kernels" => ["device", "sdpa_op", "stage_args", "verify", "launch", "tune", "flash", "flash_mma"],
    "layers" => ["attention", "triangle_attention", "triangle_multiplication", "tri_mul_core", "glu",
        "channels_last", "primitives", "chunk"],
    "ext" => ["ext_metal", "ext_cuda", "ext_reactant"],
]
const ROOT_COMPONENTS = ["precision", "utils"]
const COMPONENTS = [reduce(vcat, last.(COMPONENT_GROUPS)); ROOT_COMPONENTS]
const BACKENDS = ["cpu", "gpu", "reactant"]
const MODELS = ["af2", "af3", "boltz2"]
const KNOWN_TOKENS = [first.(COMPONENT_GROUPS); COMPONENTS; BACKENDS; MODELS]

function requested_tokens()
    tokens = isempty(ARGS) ? split(get(ENV, "LTA_TEST", ""), r"[,\s]+"; keepempty=false) : ARGS
    tokens = String.(tokens)
    unknown = setdiff(tokens, KNOWN_TOKENS)
    isempty(unknown) || error("unknown test section(s) $unknown; known: $KNOWN_TOKENS")
    return tokens
end

const TOKENS = requested_tokens()

# The component tokens, with each group expanded to its components.
const COMPONENT_TOKENS = let toks = String[]
    for t in TOKENS
        group = findfirst(g -> first(g) == t, COMPONENT_GROUPS)
        if group !== nothing
            append!(toks, last(COMPONENT_GROUPS[group]))
        elseif t in COMPONENTS
            push!(toks, t)
        end
    end
    toks
end
const BACKEND_TOKENS = filter(in(BACKENDS), TOKENS)
const MODEL_TOKENS = filter(in(MODELS), TOKENS)

const VERBOSE = lowercase(get(ENV, "JULIA_TEST_VERBOSE", "false")) in ("true", "1", "yes")

"""
    backend_selected(b) -> Bool

Whether the backend tokens allow `b` (all do when none were given). Does not check that it is available.
"""
backend_selected(b) = isempty(BACKEND_TOKENS) || string(b) in BACKEND_TOKENS

const ALL_BACKENDS = Tuple(Symbol.(BACKENDS))

"""
    named(component; backend=:cpu, model=nothing) -> Bool

Whether the tokens select the unit, ignoring whether its backend is available (so nothing is loaded). `backend` is
one backend or a tuple: any of them selected is enough.
"""
function named(component::AbstractString; backend=:cpu, model=nothing)
    component in COMPONENTS || error("`$component` is not a known test component; known: $COMPONENTS")
    (isempty(COMPONENT_TOKENS) || component in COMPONENT_TOKENS) || return false
    if !isempty(MODEL_TOKENS)
        (model !== nothing && string(model) in MODEL_TOKENS) || return false
    end
    return any(backend_selected, backend isa Tuple ? backend : (backend,))
end

"""
    selected(component; backend=:cpu, model=nothing) -> Bool

Whether the unit tagged `component` (and, if it is a Python parity unit, `model`) runs. `backend` is one backend or a
tuple: the unit runs when any of them is both selected and available (all selected ones are probed). The unit being skipped because a backend is
unavailable is reported once per backend.
"""
function selected(component::AbstractString; backend=:cpu, model=nothing)
    named(component; backend, model) || return false
    # Availability is looked at last, so a unit that is deselected, or that runs on the host, never loads a GPU package.
    # Every selected backend is probed, none skipped once one is found: a unit tagged `(:cpu, :reactant)` has a
    # compiled part that calls into support/reactant.jl, which has to be loaded before the unit's file is.
    ok = false
    for b in (backend isa Tuple ? backend : (backend,))
        backend_selected(b) || continue
        if backend_available(b)
            ok = true
        else
            @warn "backend `$b` is selected but not available; its test units are skipped" maxlog = 1 _id = b
        end
    end
    return ok
end

"""
    selected_traced_on_gpu(name) -> Bool

For a unit of `test/ext/reactant/` that traces kernels to the GPU's client (component `ext_reactant`, backend `reactant`):
selected by the tokens, and only when a GPU backend was found. Without one it is skipped with a note, as the `gpu` units are.
`name` is only what the note calls it.
"""
function selected_traced_on_gpu(name::AbstractString)
    selected("ext_reactant"; backend=:reactant) || return false
    backend_available(:gpu) && return true
    @info "no GPU backend: the traced $name kernel tests are skipped" maxlog = 1 _id = Symbol("traced_", name)
    return false
end
