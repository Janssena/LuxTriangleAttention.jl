# How much of the Phase 1 speedup does each shared-memory change give? One command, no file edits.
#
#     julia --project=test test/bench/kernels/p1_attribution.jl                  # every variant
#     julia --project=test test/bench/kernels/p1_attribution.jl base all         # just these
#     LTA_PIN=lta_p1_all.log julia --project=test test/bench/kernels/p1_attribution.jl     # same configurations in every variant
#
# With `LTA_PIN=<log>` the tuner is not run: every variant runs the forward / dkv / dq configurations printed in <log> (an
# earlier run of `cuda.bench.jl`, e.g. the `all` log of an unpinned run), so a difference between variants is the kernel's, not the
# tuner's. Pin to the `all` log: that build uses the most shared memory, so its configurations fit every other variant. The pinned
# configurations were chosen for that build, which favours it a little: compare `base`, `pad`... with each other first. No tuning
# also makes a variant much faster. Pinned logs are `lta_p1_<variant>_pinned.log`.
#
# Each variant sets the three CUDA switches of `ext/LuxTriangleAttentionCUDAExt/flash_mma.jl` through the package's
# preferences (`test/LocalPreferences.toml`, removed again at the end), and runs `cuda.bench.jl` (LTA_BENCH=main) in a fresh
# process with a fresh `KERNELDISPATCH_CACHE`. Changing a preference recompiles the extension, which takes a minute per variant.
# The child prints the switches it actually compiled with; a mismatch aborts, so a variant can never silently be another one.
# Logs: `lta_p1_<variant>.log` in the working directory. At the end: the `:flash_mma` times relative to `base`.
#
#   base       no padding, scalar loads, bias from device memory: the kernel before Phase 1
#   pad        + padded shared-memory layout (improvement 2)
#   pad_ldm    + `ldmatrix` (improvement 1)
#   pad_stage  padding + staged bias, scalar loads
#   all        everything (the default build)
#
# Run it on an idle GPU (`nvidia-smi`): the whole table is a comparison of timings.

using Preferences, LuxTriangleAttention, Printf, Statistics

const LTA = LuxTriangleAttention
const VARIANTS = [
    "base" => (pad=0, ldm=false, stage=false),
    "pad" => (pad=8, ldm=false, stage=false),
    "pad_ldm" => (pad=8, ldm=true, stage=false),
    "pad_stage" => (pad=8, ldm=false, stage=true),
    "all" => (pad=8, ldm=true, stage=true),
]
const PROJECT = abspath(joinpath(@__DIR__, "..", ".."))                 # test/
const BENCH = joinpath(@__DIR__, "cuda.bench.jl")

const PIN = get(ENV, "LTA_PIN", nothing)
PIN === nothing || isfile(PIN) || error("LTA_PIN=$PIN: no such file")
const SUFFIX = PIN === nothing ? "" : "_pinned"

wanted = isempty(ARGS) ? first.(VARIANTS) : ARGS
unknown = setdiff(wanted, first.(VARIANTS))
isempty(unknown) || error("unknown variant(s) $(unknown); known: $(first.(VARIANTS))")

function setswitches(v)
    set_preferences!(LTA, "cuda_smem_pad" => v.pad, "cuda_ldmatrix" => v.ldm, "cuda_stage_bias" => v.stage; force=true)
end
clearswitches() = delete_preferences!(LTA, "cuda_smem_pad", "cuda_ldmatrix", "cuda_stage_bias"; force=true)

# rows of one log: (type, hd, n) => (fwd mma ms, bwd mma ms)
function parselog(path)
    rows = Dict{Tuple{String,Int,Int},Tuple{Float64,Float64}}()
    for l in eachline(path)
        m = match(r"^(\w+)\s+(\d+)\s+(\d+) \|\s+[\d.]+\s+([\d.]+)\s+[\d.]+x \|\s+[\d.]+\s+([\d.]+)\s+[\d.]+x", l)
        m === nothing || (rows[(m[1], parse(Int, m[2]), parse(Int, m[3]))] = (parse(Float64, m[4]), parse(Float64, m[5])))
    end
    return rows
end

logs = Dict{String,String}()
try
    for (name, v) in VARIANTS
        name in wanted || continue
        setswitches(v)
        log = abspath("lta_p1_$name$SUFFIX.log")
        cache = mktempdir()
        println("== $name: smem_pad=$(v.pad) ldmatrix=$(v.ldm) stage_bias=$(v.stage)  ->  $log")
        env = ["KERNELDISPATCH_CACHE" => cache, "LTA_BENCH" => "main"]
        PIN === nothing || push!(env, "LTA_PIN" => abspath(PIN))
        cmd = addenv(`$(Base.julia_cmd()) --project=$PROJECT $BENCH`, env...)
        open(log, "w") do io                                   # one handle: two would overwrite each other
            run(pipeline(cmd; stdout=io, stderr=io))
        end
        want = "smem_pad=$(v.pad) ldmatrix=$(v.ldm) stage_bias=$(v.stage)"
        any(l -> occursin(want, l), eachline(log)) ||
            error("variant $name: the benchmark did not compile with `$want` (see the `shared-memory path:` line of $log)")
        logs[name] = log
        rm(cache; recursive=true, force=true)
    end
finally
    clearswitches()
end

# --- summary: time of each variant relative to `base` (below 1 is faster) -------------------------------------------------
names = [n for n in first.(VARIANTS) if haskey(logs, n)]
data = Dict(n => parselog(logs[n]) for n in names)
ref = "base" in names ? "base" : first(names)
keys_ = sort!(collect(keys(data[ref])); by=k -> (k[1] == "Float32" ? 0 : k[1] == "Float16" ? 1 : 2, k[2], k[3]))
for (pass, label) in ((1, "forward"), (2, "backward"))
    println("\n`:flash_mma` $label time relative to $ref (<1 is faster):")
    @printf("%-9s %3s %4s |", "type", "hd", "n")
    foreach(n -> @printf(" %9s", n), names)
    println()
    for k in keys_
        @printf("%-9s %3d %4d |", k...)
        for n in names
            haskey(data[n], k) ? @printf(" %9.3f", data[n][k][pass] / data[ref][k][pass]) : @printf(" %9s", "-")
        end
        println()
    end
    @printf("%-18s |", "median")
    foreach(n -> @printf(" %9.3f", median(data[n][k][pass] / data[ref][k][pass] for k in keys_ if haskey(data[n], k))), names)
    println()
end
