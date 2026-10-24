# Mirrors `src/kernels/flash/`. `forward`/`backward` need SOME device to launch a `@kernel` at all and
# use the machine's GPU for that, but what is under test is the kernel's own math, not Metal's launch
# machinery — see `backward.gpu.test.jl`'s header, and contrast `test/ext/metal/`. The tuning tests are
# `test/kernels/tune.test.jl` (no device: every backend callback is a mock) and, on a real device, `test/ext/metal/tune.test.jl`.

@testset verbose = VERBOSE "forward" begin
    include("forward.gpu.test.jl")
end

@testset verbose = VERBOSE "backward" begin
    include("backward.gpu.test.jl")
end
