# Mirrors `ext/`: what each extension's OWN glue code does (registration hooks, the hooks `:flash_mma` relies on, the rolled
# chunk loop) and what only a trace can get wrong — as opposed to `test/kernels/` and `test/layers/`, which test kernel and layer
# behaviour natively. The GPU extensions' tests are in a folder per backend, next to the adapter that the generic `.gpu.test.jl`
# files use, and run when that backend is the one found. Everything traced by Reactant is in `reactant/`: against the native
# layer in Float32, or at 16 bits where the compiler's rewrites matter. The traced KERNEL tests run only with a GPU.
#
# `device_call` was removed from KernelDispatch.jl (see `metal/flash_mma.test.jl` for the `ka_compile` replacement).

selected("ext_metal"; backend=:gpu) && gpu_backend() === :metal && @testset verbose = VERBOSE "Metal" begin
    include("metal/flash_mma.test.jl")        # ext/LuxTriangleAttentionMetalExt/flash_mma.jl
    include("metal/tune.test.jl")             # what the Apple compiler does with the tuned kernels
end

selected("ext_cuda"; backend=:gpu) && gpu_backend() === :cuda && @testset verbose = VERBOSE "CUDA" begin
    include("cuda/device.test.jl")            # ext/LuxTriangleAttentionCUDAExt.jl
    include("cuda/flash_mma.test.jl")         # ext/LuxTriangleAttentionCUDAExt/flash_mma.jl
end

selected("ext_reactant"; backend=:reactant) && @testset verbose = VERBOSE "Reactant" begin
    @testset "SDPA paths" begin include("reactant/sdpa_op.test.jl") end                  # against the native paths
    @testset "chunked attention" begin include("reactant/chunk.test.jl") end             # ext/LuxTriangleAttentionReactantExt/chunk.jl
    @testset "Attention, 16-bit" begin include("reactant/attention.test.jl") end
    @testset "TriangleAttention, 16-bit" begin include("reactant/triangle_attention.test.jl") end
    @testset "StableLayerNorm" begin include("reactant/primitives.test.jl") end
    @testset "TriMulCore" begin include("reactant/tri_mul_core.test.jl") end             # Folded and HalfSafe
end

selected_traced_on_gpu("kernel") && @testset verbose = VERBOSE "Reactant, kernels in the trace" begin
    @testset "SDPA kernel launches" begin include("reactant/kernel_launch.test.jl") end
    @testset "Attention(use_kernel=true)" begin include("reactant/attention_kernel.test.jl") end
    @testset ":flash_mma" begin include("reactant/flash_mma.test.jl") end
end
