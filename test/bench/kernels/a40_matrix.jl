# Everything that is still open on the A40 (docs/cuda_kernel_speed_improvements.md, "To do"), in ONE command:
# tests, accuracy and speed of the three opt-in modes, each alone and together, against the default build.
#
#     julia --project=test test/bench/kernels/a40_matrix.jl                        # every variant, every stage
#     julia --project=test test/bench/kernels/a40_matrix.jl det sis_resid          # just these variants
#     LTA_STAGES=accuracy,bench julia --project=test test/bench/kernels/a40_matrix.jl     # just these stages, in every variant
#     LTA_MATRIX=summary julia --project=test test/bench/kernels/a40_matrix.jl     # no run: only re-print the tables from the logs
#     LTA_MATRIX=dry     julia --project=test test/bench/kernels/a40_matrix.jl     # no run: print what would be run
#
# Run it on an IDLE GPU (`nvidia-smi`), from the repository root, in tmux/screen: it takes hours (the CUDA test selection alone is
# ~25 min per variant, and every variant recompiles the package and its CUDA extension, about a minute). Progress goes to the
# terminal, the output of each stage to `lta_matrix_<variant>_<stage>.log` in the working directory (tail -f one to watch it).
#
# Variants (each sets its compile-time preferences through `test/LocalPreferences.toml`, which is restored at the end):
#
#   defaults    no preference set. Tests: SKIPPED (already run: lta_cuda_tests_defaults.log). Accuracy + bench: the reference of every table.
#   det         deterministic_dbias            tests must report ONE distinct dbias result in 5 runs; bench shows its cost
#   sis         scale_in_scores                accuracy (error / rounding floor) and speed of the score scaling
#   resid       output_residual                same, for the output residual (the op returns (o, lse, e))
#   sis_resid   scale_in_scores + output_residual   the candidate new default (BFloat16 reaches the floor on Metal with both)
#   all         all three
#   defaults2   no preference set again, bench only: the run-to-run noise. A difference of another variant below this one's is not real.
#
# Stages: `tests`    LTA_TEST_GPU=cuda test/runtests.jl <TESTS below>
#         `accuracy` accuracy.bench.jl (error / rounding floor of dq, dk, dv, dbias; nothing is appended to the results CSV)
#         `bench`    cuda.bench.jl with LTA_BENCH=main (`:default` against `:flash_mma`, forward and backward, tuned as always)
# Every stage is its own Julia process with its own fresh KERNELDISPATCH_CACHE (the CUDA switches and several modes are not part of a
# kernel's cache key). Before the first stage of a variant a probe process prints the modes the package was actually compiled with; a
# mismatch aborts, so a variant can never silently be another one. A failing stage does not stop the run: its exit code is in the summary.

using LuxTriangleAttention, Printf, Statistics, Dates

const LTA = LuxTriangleAttention
const Pref = LTA.Preferences             # the package's own dependency (the test project does not list Preferences)
const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const PROJECT = joinpath(ROOT, "test")
const BENCH = joinpath(@__DIR__, "cuda.bench.jl")
const ACCURACY = joinpath(@__DIR__, "accuracy.bench.jl")
const TESTS = split("ext_cuda flash_mma stage_args launch sdpa_op attention gpu")     # what the defaults run (lta_cuda_tests_defaults.log) used
const PREFKEYS = ("deterministic_dbias", "scale_in_scores", "output_residual")
const MODE = get(ENV, "LTA_MATRIX", "run")                                          # run | summary | dry

const ALL = [:tests, :accuracy, :bench]
const VARIANTS = [
    (name="defaults", det=false, sis=false, resid=false, stages=[:accuracy, :bench]),
    (name="det", det=true, sis=false, resid=false, stages=ALL),
    (name="sis", det=false, sis=true, resid=false, stages=ALL),
    (name="resid", det=false, sis=false, resid=true, stages=ALL),
    (name="sis_resid", det=false, sis=true, resid=true, stages=ALL),
    (name="all", det=true, sis=true, resid=true, stages=ALL),
    (name="defaults2", det=false, sis=false, resid=false, stages=[:bench]),
]
const NAMES = [v.name for v in VARIANTS]

wanted = isempty(ARGS) ? NAMES : ARGS
unknown = setdiff(wanted, NAMES)
isempty(unknown) || error("unknown variant(s) $unknown; known: $NAMES")
const STAGE_OVERRIDE = haskey(ENV, "LTA_STAGES") ? Symbol.(split(ENV["LTA_STAGES"], r"[,\s]+"; keepempty=false)) : nothing
STAGE_OVERRIDE === nothing || issubset(STAGE_OVERRIDE, ALL) || error("LTA_STAGES must be among $ALL")
stages_of(v) = STAGE_OVERRIDE === nothing ? v.stages : [s for s in ALL if s in STAGE_OVERRIDE]
logpath(v, stage) = abspath("lta_matrix_$(v.name)_$(stage).log")

const JULIA = Base.julia_cmd()

# --- one stage -----------------------------------------------------------------------------------------------------------------
function stagecmd(stage)
    stage == :tests && return `$JULIA --project=$PROJECT $(joinpath(PROJECT, "runtests.jl")) $TESTS`
    stage == :accuracy && return `$JULIA --project=$PROJECT $ACCURACY`
    return `$JULIA --project=$PROJECT $BENCH`
end

function runstage(v, stage)
    log = logpath(v, stage)
    cache = mktempdir()
    env = ["KERNELDISPATCH_CACHE" => cache, "LTA_TEST_GPU" => "cuda", "LTA_BENCH_GPU" => "cuda", "LTA_BENCH" => "main", "LTA_ACC_NOCSV" => "1"]
    cmd = addenv(Cmd(stagecmd(stage); dir=ROOT, ignorestatus=true), env...)
    t0 = time()
    @printf("[%s] %-9s %-8s -> %s\n", Dates.format(now(), "HH:MM:SS"), v.name, stage, log)
    code = open(log, "w") do io                               # one handle: two would overwrite each other
        run(pipeline(cmd; stdout=io, stderr=io)).exitcode
    end
    rm(cache; recursive=true, force=true)
    @printf("[%s] %-9s %-8s done in %.1f min, exit code %d\n", Dates.format(now(), "HH:MM:SS"), v.name, stage, (time() - t0) / 60, code)
    return code
end

# --- preferences ---------------------------------------------------------------------------------------------------------------
clearprefs() = Pref.delete_preferences!(LTA, PREFKEYS...; force=true)
function setprefs(v)
    clearprefs()
    v.det && Pref.set_preferences!(LTA, "deterministic_dbias" => true; force=true)
    v.sis && Pref.set_preferences!(LTA, "scale_in_scores" => true; force=true)
    v.resid && Pref.set_preferences!(LTA, "output_residual" => true; force=true)
end

# what the package really compiled with: a fresh process, after the preference change
function checkmodes(v)
    code = """using CUDA, LuxTriangleAttention; L = LuxTriangleAttention
        println("MODES det=", L.DET_DBIAS, " sis=", L.SCALE_IN_SCORES, " resid=", L.OUTPUT_RESID)"""
    out = read(addenv(Cmd(`$JULIA --project=$PROJECT -e $code`; dir=ROOT), "KERNELDISPATCH_CACHE" => mktempdir()), String)
    want = "MODES det=$(v.det) sis=$(v.sis) resid=$(v.resid)"
    occursin(want, out) || error("variant $(v.name): the package did not compile with `$want` (probe printed: $(strip(out)))")
    println("   compiled with: ", strip(out))
end

# --- summary -------------------------------------------------------------------------------------------------------------------
# rows of a cuda.bench.jl log: (type, hd, n) => (forward `:flash_mma` ms, backward `:flash_mma` ms)
function parsebench(path)
    rows = Dict{Tuple{String,Int,Int},Tuple{Float64,Float64}}()
    for l in eachline(path)
        m = match(r"^(\w+)\s+(\d+)\s+(\d+) \|\s+[\d.]+\s+([\d.]+)\s+[\d.]+x \|\s+[\d.]+\s+([\d.]+)\s+[\d.]+x", l)
        m === nothing || (rows[(m[1], parse(Int, m[2]), parse(Int, m[3]))] = (parse(Float64, m[4]), parse(Float64, m[5])))
    end
    return rows
end

# rows of an accuracy.bench.jl log: (type, shape, setting, gradient) => error / rounding floor
function parseacc(path)
    rows = Dict{NTuple{4,String},Float64}()
    for l in eachline(path)
        m = match(r"^(\w+)\s+(hd=\d+ n=\d+)\s+(plain|bias\+mask)\s+(\w+)\s+\|\s+\S+\s+\S+\s+([\d.]+)x", l)
        m === nothing || (rows[(String(m[1]), String(m[2]), String(m[3]), String(m[4]))] = parse(Float64, m[5]))
    end
    return rows
end

typeorder(t) = t == "Float32" ? 0 : t == "Float16" ? 1 : 2

function summarize(io, vs)
    avail(stage) = [v for v in vs if stage in stages_of(v) && isfile(logpath(v, stage))]
    println(io, "A40 matrix summary, $(Dates.format(now(), "yyyy-mm-dd HH:MM"))\n")

    # tests
    println(io, "== tests: `", join(TESTS, " "), "` ==")
    for v in avail(:tests)
        ls = readlines(logpath(v, :tests))
        i = findlast(l -> startswith(l, "Test Summary:"), ls)
        println(io, "\n[$(v.name)]")
        for l in ls
            occursin("dbias repeatability", l) && println(io, "  ", l)
        end
        if i === nothing
            println(io, "  NO `Test Summary` in the log: the run did not finish (see $(logpath(v, :tests)))")
        else
            foreach(l -> println(io, "  ", l), ls[i:min(i + 25, end)])
        end
    end
    isempty(avail(:tests)) && println(io, "(no test logs)")
    println(io, "\n(`det` and `all` must report 1 distinct dbias result in 5 runs; every other variant reports 5. The defaults' own run is lta_cuda_tests_defaults.log.)")

    # accuracy
    av = avail(:accuracy)
    if !isempty(av)
        data = Dict(v.name => parseacc(logpath(v, :accuracy)) for v in av)
        names = [v.name for v in av]
        ref = first(names)
        keys_ = sort!(collect(union(keys.(values(data))...)); by=k -> (typeorder(k[1]), k[2], k[3], k[4]))
        println(io, "\n== accuracy: RMSE / rounding floor (1.00 is perfect; lower is better) ==")
        @printf(io, "%-9s %-12s %-10s %-6s |", "type", "shape", "setting", "grad")
        foreach(n -> @printf(io, " %9s", n), names)
        println(io)
        for k in keys_
            @printf(io, "%-9s %-12s %-10s %-6s |", k...)
            for n in names
                haskey(data[n], k) ? @printf(io, " %9.2f", data[n][k]) : @printf(io, " %9s", "-")
            end
            println(io)
        end
        println(io, "median ratio per gradient over all rows:")
        for g in ("dq", "dk", "dv", "dbias")
            @printf(io, "%-37s |", g)
            for n in names
                xs = [data[n][k] for k in keys_ if k[4] == g && haskey(data[n], k)]
                isempty(xs) ? @printf(io, " %9s", "-") : @printf(io, " %9.2f", median(xs))
            end
            println(io)
        end
        println(io, "(`dv` is the control: no mode may change it much. Reference column: $ref.)")
    end

    # speed
    bv = avail(:bench)
    if !isempty(bv)
        data = Dict(v.name => parsebench(logpath(v, :bench)) for v in bv)
        names = [v.name for v in bv]
        ref = "defaults" in names ? "defaults" : first(names)
        keys_ = sort!(collect(keys(data[ref])); by=k -> (typeorder(k[1]), k[2], k[3]))
        for (pass, label) in ((1, "forward"), (2, "backward"))
            println(io, "\n== speed: `:flash_mma` $label time relative to $ref (<1 is faster; `defaults2` is the same build again: that is the noise) ==")
            @printf(io, "%-9s %3s %4s |", "type", "hd", "n")
            foreach(n -> @printf(io, " %9s", n), names)
            println(io)
            for k in keys_
                @printf(io, "%-9s %3d %4d |", k...)
                for n in names
                    haskey(data[n], k) ? @printf(io, " %9.3f", data[n][k][pass] / data[ref][k][pass]) : @printf(io, " %9s", "-")
                end
                println(io)
            end
            @printf(io, "%-18s |", "median")
            foreach(n -> @printf(io, " %9.3f", median(data[n][k][pass] / data[ref][k][pass] for k in keys_ if haskey(data[n], k))), names)
            println(io)
        end
    end

    # exit codes of the stages that were run in this process
    println(io, "\n== stage exit codes (0 = ok) ==")
    for v in vs, s in stages_of(v)
        haskey(EXITCODES, (v.name, s)) && @printf(io, "%-10s %-9s %d\n", v.name, s, EXITCODES[(v.name, s)])
    end
end

const EXITCODES = Dict{Tuple{String,Symbol},Int}()

function writesummary(vs; echo=true)
    buf = IOBuffer()
    summarize(buf, vs)
    s = String(take!(buf))
    echo && print(s)
    write("lta_matrix_summary.txt", s)
    println("\nsummary so far: lta_matrix_summary.txt")
end

# --- main ----------------------------------------------------------------------------------------------------------------------
vs = [v for v in VARIANTS if v.name in wanted]

if MODE == "dry"
    for v in vs
        println("== $(v.name): deterministic_dbias=$(v.det) scale_in_scores=$(v.sis) output_residual=$(v.resid)")
        for s in stages_of(v)
            println("   $s: ", stagecmd(s), "   -> ", logpath(v, s))
        end
    end
elseif MODE == "summary"
    writesummary(vs)
elseif MODE == "run"
    try
        println(read(`nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv`, String))
        println("(the GPU must be idle: every table is a comparison of timings)\n")
    catch
        println("nvidia-smi not found: could not check that the GPU is idle\n")
    end
    saved = Dict(k => Pref.load_preference(LTA, k, nothing) for k in PREFKEYS)     # restored at the end
    try
        for v in vs
            println("== $(v.name): deterministic_dbias=$(v.det) scale_in_scores=$(v.sis) output_residual=$(v.resid)")
            setprefs(v)
            checkmodes(v)
            for s in stages_of(v)
                EXITCODES[(v.name, s)] = runstage(v, s)
            end
            writesummary(vs[1:findfirst(x -> x.name == v.name, vs)]; echo=false)    # the tables so far: a partial result survives an abort
        end
    finally
        clearprefs()
        for (k, x) in saved
            x === nothing || Pref.set_preferences!(LTA, k => x; force=true)
        end
    end
    println("\n\n", "="^100, "\nALL DONE")
    writesummary(vs)
else
    error("LTA_MATRIX must be run, summary or dry (got $MODE)")
end
