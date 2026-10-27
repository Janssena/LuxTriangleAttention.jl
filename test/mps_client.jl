# Locating the jax-mps Metal PJRT plugin, for the tests and benchmarks that need a Reactant client
# that actually traces to a GPU.
#
# Reactant's default client is CPU, and `backend_of` answers `CPUBackend()` for a traced array under
# it — so `SDPA()` takes the host path and the device-kernel route is untested unless a real Metal
# client is installed first. Anything exercising that path has to build this client explicitly.
#
# The plugin is found through the environment variable `JAX_MPS_PLUGIN`, the path of `libpjrt_plugin_mps.dylib`. There
# is no default location: without the variable `mps_client()` returns `nothing` and the tests that trace kernels to the
# Metal GPU are skipped, with a warning.
#
# No side effects on include: `mps_client()` is a function, and returns `nothing` rather than
# throwing when the plugin is absent, so callers skip cleanly on a machine without it.

const JAX_MPS_PLUGIN = get(ENV, "JAX_MPS_PLUGIN", nothing)

# MLX (the engine behind the plugin) reserves ~1.5x the working set and does not release freed
# buffers between unrelated compiles. Must be set before the plugin creates its client.
get!(ENV, "JAX_MPS_CACHE_LIMIT_BYTES", string(4 * 1024^3))

const _MPS_SKIPPED = "so the tests that trace kernels to the Metal GPU are skipped (eager Metal tests do not need the plugin)"

function _make_mps_client()
    if isnothing(JAX_MPS_PLUGIN)
        @warn "JAX_MPS_PLUGIN is not set, no jax-mps Metal PJRT plugin is available, $_MPS_SKIPPED. Set it to the path of libpjrt_plugin_mps.dylib to run them."
        return nothing
    end
    if !isfile(JAX_MPS_PLUGIN)
        @warn "JAX_MPS_PLUGIN=$JAX_MPS_PLUGIN is not a file, $_MPS_SKIPPED."
        return nothing
    end
    return try
        Reactant.XLA.PJRT.MakeClientUsingPluginAPI(JAX_MPS_PLUGIN, "mps", "MPS")
    catch err
        @warn "the jax-mps plugin at $JAX_MPS_PLUGIN failed to load, $_MPS_SKIPPED." exception = (err, catch_backtrace())
        nothing
    end
end

const _MPS_CLIENT = Ref{Any}(missing)

"""
    mps_client() -> Reactant PJRT client or `nothing`

The jax-mps Metal client, or `nothing` (with a warning, once) if `JAX_MPS_PLUGIN` is not set, does not name a file, or
the plugin fails to load. Cached, so repeated calls in one process reuse the one client — PJRT clients are not free, and
two live clients for the same device is not a supported configuration.
"""
function mps_client()
    _MPS_CLIENT[] === missing || return _MPS_CLIENT[]
    _MPS_CLIENT[] = _make_mps_client()
    return _MPS_CLIENT[]
end
