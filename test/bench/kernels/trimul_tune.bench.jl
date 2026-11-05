# Tune `TriMul`'s `:default` kernel on the machine's GPU (Metal or CUDA): the forward and the backward over the tunables of
# `trimul_tunables`, one at a time around the current default (a full grid would be hundreds of compiles). Writes a CSV on
# stdout and, with a path argument, to that file; paste it back to read off the best `trimul_tunables(::CUDABackendTag)`.
#
#     julia --project=test test/bench/kernels/trimul_tune.bench.jl [out.csv] [N=384] [C=128] [H=128]
#
# `LTA_BENCH_GPU=metal|cuda` overrides the backend (Metal on a Mac, else CUDA). Times are the median of `REPS` calls of the
# whole forward / backward, host-synchronised (the way a layer is called), after one warm-up that also compiles the variant.
# Differences under ~10% between rows are noise on a shared machine; only a larger shift is a finding.

using LuxTriangleAttention, KernelDispatch, Random, Printf, Statistics
import LuxTriangleAttention: Lux
const LTA = LuxTriangleAttention

const GPU = Symbol(lowercase(get(ENV, "LTA_BENCH_GPU", Sys.isapple() ? "metal" : "cuda")))
if GPU === :metal
    @eval using Metal
    const DeviceArray = Metal.MtlArray
    sync() = Metal.synchronize()
else
    @eval using CUDA
    const DeviceArray = CUDA.CuArray
    sync() = CUDA.synchronize()
end

const OUT = length(ARGS) >= 1 ? ARGS[1] : nothing
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 384
const C = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 128
const H = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 128
const REPS = 7

function main()
    kern = TriangleMultiplication(C, H; use_kernel=true)
    ps, st = Lux.setup(Random.Xoshiro(1), kern)
    dev(t) = Lux.fmap(a -> a isa Array ? DeviceArray(a) : a, t)
    rng = Random.Xoshiro(2)
    x = DeviceArray(randn(rng, Float32, C, N, N, 1))
    mask = DeviceArray(rand(rng, N, N, 1) .< 0.8)
    w = DeviceArray(randn(rng, Float32, C, N, N, 1))
    ps, st = dev(ps), dev(st)
    args = (kern, x, mask, ps, st)
    fwd() = (kern(x, mask, ps, st); sync())
    bwd() = (KernelDispatch.backward(LTA.TriMul(), LTA.TriMulStaged(), args, nothing, (w, nothing)); sync())
    timeit(f) = (f(); median([(t = time_ns(); f(); (time_ns() - t) / 1e6) for _ in 1:REPS]))

    be = KernelDispatch.device_backend()
    base = LTA.trimul_tunables(be)
    println("# TriMul tune: ", GPU, " ", LTA.KernelDispatch.device_caps(be).name, ", N=$N C=$C H=$H B=1, defaults ", base)
    rows = ["param,value,fwd_ms,bwd_ms"]
    emit(param, value, f, b) = (r = @sprintf("%s,%s,%.2f,%.2f", param, value, f, b); println(r); push!(rows, r))
    try
        LTA.set_trimul_tunables!()
        emit("default", "-", timeit(fwd), timeit(bwd))
        for (param, values) in (:nb => (4096, 8192, 16384, 32768, 65536, 131072), :NTE => (128, 256, 512, 1024),
            :NT => (128, 256, 512), :TB => (8, 16, 32))
            for v in values
                getproperty(base, param) == v && continue
                LTA.set_trimul_tunables!(; param => v)
                f = try
                    (timeit(fwd), timeit(bwd))
                catch e
                    println("# $param=$v failed: ", first(sprint(showerror, e), 200)); continue
                end
                emit(param, v, f...)
            end
        end
    finally
        LTA.set_trimul_tunables!()
    end
    OUT === nothing || write(OUT, join(rows, "\n") * "\n")
end

main()
