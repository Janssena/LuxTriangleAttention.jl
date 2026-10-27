# Mirrors `ext/`: what each extension's OWN glue code does (registration hooks, the hooks `:flash_mma` relies on, the rolled
# chunk loop) and what only a trace can get wrong — as opposed to `test/kernels/` and `test/layers/`, which test kernel and layer
# behaviour natively. The GPU extensions' tests are in a folder per backend, next to the adapter that the generic `.gpu.test.jl`
# files use, and run when that backend is the one found. Everything traced by Reactant is in `reactant/`: against the native
# layer in Float32, or at 16 bits where the compiler's rewrites matter. The traced KERNEL tests run only with a GPU.
#
# `device_call` was removed from KernelDispatch.jl (see `metal/flash_mma.test.jl` for the `ka_compile` replacement).

# XLA multiplies Float32 matrices at TF32 precision on NVIDIA GPUs from Ampere on (its default `dot_general` precision): a
# relative error of ~3e-4 against Float32 (measured on an A40), which the Float32 tolerances of
# the traced tests do not allow and which says nothing about this package. The Reactant units therefore run at HIGHEST precision
# on CUDA, so what they measure is our code. (Metal and the CPU client multiply in full Float32 already.)
function reactant_exact(f)
    (isdefined(Main, :GPU) && gpu_backend() === :cuda) || return f()
    return Base.invokelatest() do
        R = Main.Reactant
        R.with_config(f; dot_general_precision=R.PrecisionConfig.HIGHEST, convolution_precision=R.PrecisionConfig.HIGHEST)
    end
end

selected("ext_metal"; backend=:gpu) && gpu_backend() === :metal && @testset verbose = VERBOSE "Metal" begin
    include("metal/flash_mma.test.jl")        # ext/LuxTriangleAttentionMetalExt/flash_mma.jl
    include("metal/tune.test.jl")             # what the Apple compiler does with the tuned kernels
end

selected("ext_cuda"; backend=:gpu) && gpu_backend() === :cuda && @testset verbose = VERBOSE "CUDA" begin
    include("cuda/device.test.jl")            # ext/LuxTriangleAttentionCUDAExt.jl
    include("cuda/flash_mma.test.jl")         # ext/LuxTriangleAttentionCUDAExt/flash_mma.jl
end

selected("ext_reactant"; backend=:reactant) && @testset verbose = VERBOSE "Reactant" begin
    reactant_exact() do
        @testset "SDPA paths" begin include("reactant/sdpa_op.test.jl") end                  # against the native paths
        @testset "chunked attention" begin include("reactant/chunk.test.jl") end             # ext/LuxTriangleAttentionReactantExt/chunk.jl
        @testset "Attention, 16-bit" begin include("reactant/attention.test.jl") end
        @testset "TriangleAttention, 16-bit" begin include("reactant/triangle_attention.test.jl") end
        @testset "StableLayerNorm" begin include("reactant/primitives.test.jl") end
        @testset "TriMulCore" begin include("reactant/tri_mul_core.test.jl") end             # Folded and HalfSafe
    end
end

selected_traced_on_gpu("kernel") && @testset verbose = VERBOSE "Reactant, kernels in the trace" begin
    reactant_exact() do
        @testset "SDPA kernel launches" begin include("reactant/kernel_launch.test.jl") end
        @testset "Attention(use_kernel=true)" begin include("reactant/attention_kernel.test.jl") end
        @testset ":flash_mma" begin include("reactant/flash_mma.test.jl") end
    end
end
