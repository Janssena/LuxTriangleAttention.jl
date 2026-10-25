# Why does `:flash_mma` find no dispatchable, verified configuration on CUDA? The tuner swallows each candidate's failure
# (`measure` turns any error into `Inf`); this builds ONE small configuration per stage and dtype and says what happens:
# the full GPUCompiler error if it does not compile, else the pipeline's thread limit and the verification result.
#
#   julia --project=test test/scratch/cuda_mma_spike.jl 2>&1 | tee lta_cuda_mma_spike.log

using CUDA
using KernelAbstractions
using KernelDispatch
using LuxTriangleAttention

const LTA = LuxTriangleAttention
const KD = KernelDispatch

println("device: ", CUDA.name(CUDA.device()), "  fragment shape ", LTA._mma_shape(KD.device_backend()),
    "  operand types: Float16 => ", LTA.mma_operand_types(KD.device_backend(), Float16),
    ", Float32 => ", LTA.mma_operand_types(KD.device_backend(), Float32))

# GPUCompiler's own error for one stage, as KernelDispatch's CUDA extension compiles it.
function diagnose(k, stage, T, dim, p)
    ext = Base.get_extension(KD, :KernelDispatchCUDAExt)
    GPUCompiler = ext.GPUCompiler
    argtypes = LTA.stage_argtypes(k, stage, T, dim, p, true, true)
    tg = LTA.stage_threads(k, p)
    obj = LTA.stage_kernel(k, stage)(CUDA.CUDABackend(), tg)
    ftype = ext.FlatKA{typeof(obj.f),tg}
    sig = Tuple{Int,Int,Int,Int,Int,Int,map(ext._ka_argtype, argtypes)...}
    job = GPUCompiler.CompilerJob(GPUCompiler.methodinstance(ftype, sig), ext._stateless_config("diag_$stage"))
    try
        GPUCompiler.JuliaContext() do _
            Base.invokelatest(GPUCompiler.compile, :asm, job)
        end
        println("    compiles")
        return true
    catch err
        lines = split(sprint(showerror, err), '\n')
        println("    does NOT compile — ", length(lines), " lines of error; first 60:")
        foreach(l -> println("      ", l), first(lines, 60))
        return false
    end
end

k = LTA.FlashMMA()
dim = 32
for T in (Float16, LTA.BFloat16)
    println("\n=== $T, dim $dim")
    V = LTA.verification_problem(dim, T)
    for stage in LTA.FLASH_MMA_STAGES
        p = (; nsg=2, r=1, tile=16, ops=T)
        println("  :$stage at $p")
        diagnose(k, stage, T, dim, p) || continue
        try
            b = LTA.build_stage(k, stage, T, dim, p)
            lim = KD.max_threads(b)
            println("    binary built; thread limit $lim, group $(b.threadgroup)")
            ok = LTA.verify_stage(k, stage, b, V)
            println("    verification: ", ok ? "PASSED" : "FAILED")
        catch err
            println("    after compiling: ")
            showerror(stdout, err, catch_backtrace())
            println()
        end
    end
end
