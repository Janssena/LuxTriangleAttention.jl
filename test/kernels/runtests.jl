# Mirrors `src/kernels/`. Backend-agnostic where the concept allows it (`registry.jl`, `device.jl`);
# `flash/` needs SOME device to launch a `@kernel` at all and uses Metal for that today, but what is
# under test there is the kernel's own math, not Metal's launch machinery — see
# `test/kernels/flash/backward.test.jl`'s header, and contrast `test/ext/metal/`.

@testset "Registry" begin
    include("registry.test.jl")
end

@testset "Backend routing" begin
    # `reference_sdpa` — not `using`-ed or `include`d again by name; shared with `test/ext/` the same
    # way `test/python/setup.jl`'s `convert_types` is, by running ahead of it in `test/runtests.jl`.
    include("sdpa_reference.jl")
    include("backend.test.jl")
end

@testset "Device seams" begin
    include("device.test.jl")
end

@testset "Flash kernels" begin
    include("flash/backward.test.jl")
end
