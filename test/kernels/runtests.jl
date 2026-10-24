# Mirrors `src/kernels/`: each `<file>.test.jl` tests `src/kernels/<file>.jl`, and each subfolder
# has its own `runtests.jl`. Backend-agnostic where the concept allows it (`sdpa_op`, `device`);
# `flash/` and `flash_mma/` need SOME device to launch a `@kernel` at all and use Metal for that
# today, but what is under test there is the kernel's own math, not Metal's launch machinery —
# contrast `test/ext/metal/`.
#
# `<file>.gpu.test.jl` needs the machine's GPU backend (found by `support/backend_gpu.jl`; the tests use only its adapter's names,
# `test/ext/<backend>/backend.jl`). Everything that traces with Reactant is in `test/ext/reactant/`, and what is specific to one GPU
# backend in `test/ext/<backend>/`.
#
# `reference_sdpa` (`test/support/sdpa_reference.jl`) is not included here: it is shared with
# `test/layers/` and `test/ext/`, so `test/runtests.jl` includes it ahead of everything.
#
# Each unit is tagged with its component and the backends it needs (see `test/sections.jl`). Order matters:
# "device" must run with no backend loaded, and it does because nothing before it asks for one.

selected("device") && @testset verbose = VERBOSE "device" begin
    include("device.test.jl")
end

selected("sdpa_op"; backend=(:cpu, :reactant)) && @testset verbose = VERBOSE "sdpa_op" begin
    include("sdpa_op.test.jl")
end

selected("sdpa_op"; backend=:gpu) && @testset verbose = VERBOSE "sdpa_op, on the GPU" begin
    include("sdpa_op.gpu.test.jl")
end

selected("verify") && @testset verbose = VERBOSE "verify" begin
    include("verify.test.jl")
end

selected("stage_args") && @testset verbose = VERBOSE "stage_args" begin
    include("stage_args.test.jl")
end

selected("launch"; backend=:gpu) && @testset verbose = VERBOSE "launch" begin
    include("launch.gpu.test.jl")
end

selected("tune") && @testset verbose = VERBOSE "tune" begin
    include("tune.test.jl")
end

selected("flash"; backend=:gpu) && @testset verbose = VERBOSE "flash" begin
    include("flash/runtests.jl")
end

selected("flash_mma"; backend=:gpu) && @testset verbose = VERBOSE "flash_mma" begin
    include("flash_mma/runtests.jl")
end
