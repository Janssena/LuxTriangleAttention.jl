# Mirrors `ext/`: what each extension's OWN glue code does (registration hooks, AOT build/cache
# mechanics, MLIR emission, launch bookkeeping) — as opposed to `test/kernels/`, which tests kernel
# math using an extension only as the vehicle to run it.

@testset "Enzyme" begin
    include("enzyme.test.jl")
end

@testset "Metal" begin
    include("metal/launch.test.jl")
end

@testset "Reactant" begin
    include("reactant/custom_call.test.jl")
    include("reactant/chunk.test.jl")
    include("reactant/kernel_launch.test.jl")
    include("reactant/attention_kernel.test.jl")
end
