# Run everything:         julia --project=test test/runtests.jl
# One group:              julia --project=test test/runtests.jl kernels          (also: layers, ext)
# Some components:        julia --project=test test/runtests.jl attention flash
# One backend:            julia --project=test test/runtests.jl cpu              (also: gpu, reactant)
# Components x backend:   julia --project=test test/runtests.jl kernels gpu
# Python parity of one model, on the host:   julia --project=test test/runtests.jl attention af3 cpu
#
# From Julia, `Pkg.test("LuxTriangleAttention"; test_args=["attention", "cpu"])`. The pkg REPL's `test` takes no
# arguments, so there: `ENV["LTA_TEST"] = "attention cpu"` first (read when there are no command-line arguments).
#
# The tokens, what each axis means and how they combine are in `test/sections.jl`. In short: `component` is the
# src/ (or ext/) file under test, `backend` what it needs to run (a backend that is not available is skipped with
# a warning), `model` the Python reference of a parity unit (its recorded outputs are read, Python is not run).
# `JULIA_TEST_VERBOSE=true` prints the counts of the nested testsets too, also when they pass (Julia's
# Test hides them; the variable is read in sections.jl, not by Julia).

import LuxTriangleAttention: Lux, Static.static
import KernelDispatch, Random

using Test, LuxTriangleAttention, BenchmarkTools

include("sections.jl")
include("support/backends.jl")
include("support/backend_gpu.jl")

# The Python references are not run by the tests: their outputs are recorded in test/fixtures/python/ and read back
# (support/fixtures.jl). Running Python is only for recording them again, which takes `LTA_GENERATE_FIXTURES=true`, PyCall
# and torch (see test/python/README.md).
const PARITY_COMPONENTS = ("attention", "triangle_attention", "triangle_multiplication")
include("support/fixtures.jl")
if GENERATE_FIXTURES
    include("python/setup.jl")
    include("python/references.jl")
end

include("support/convert_types.jl")    # convert_types — shared with test/layers/ and test/ext/
include("support/sdpa_bwd.jl")         # sdpa_bwd — the device reverse, shared with test/ext/
include("support/sdpa_reference.jl")   # reference_sdpa — shared with test/kernels/ and test/ext/
include("support/f16_attention.jl")    # f16_overflow_attention, relerr — shared with test/layers/ and test/ext/
include("support/tri_mul_core.jl")     # tmc_reference, tmc_fold, … — shared by test/layers/ and test/ext/reactant/

@testset verbose = VERBOSE "LuxTriangleAttention.jl" begin
    # Mirrors `src/kernels/`. See its own `runtests.jl` header for what needs a device.
    @testset verbose = VERBOSE "Kernels" begin
        include("kernels/runtests.jl")
    end

    @testset verbose = VERBOSE "Layers" begin
        include("layers/runtests.jl")
    end

    # Mirrors `src/precision.jl`: contraction modes, `ishalf`, `mask_value`, `fold_trimul`. (The
    # Float32-statistics LayerNorm is tested with `StableLayerNorm`, in layers/primitives.test.jl.)
    selected("precision") && @testset verbose = VERBOSE "precision" begin
        include("precision.test.jl")
    end

    # Mirrors `src/utils.jl`.
    selected("utils") && @testset verbose = VERBOSE "utils" begin
        include("utils.test.jl")
    end

    # Mirrors `ext/`. Each extension's own glue code, as opposed to kernel math (`Kernels`, above)
    # or layer dispatch (`Layers`, above).
    @testset verbose = VERBOSE "Extensions" begin
        include("ext/runtests.jl")
    end
end
