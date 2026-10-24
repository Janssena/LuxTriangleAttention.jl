# The tuning skeleton on a real device (`src/kernels/tune.jl`), and the codegen property the kernels
# depend on.
#
# The search and persistence are tested without a device in `test/kernels/tune.test.jl`; this
# file covers what only a real device can: that every stage compiles to straight-line register code,
# that the correctness gate passes real kernels and FAILS wrong answers (both kernels), and that a
# tuned entry carries each stage's own configuration.

using Test, Metal
import KernelDispatch
import Metal: GPUCompiler

const LTAt2 = LuxTriangleAttention

if Metal.functional()
    @testset "Metal tuning" begin
        @testset "no stage keeps a register tuple on the stack (dim=$dim)" for dim in (8, 16, 24, 32)
            # See note 4 at the top of `src/kernels/flash/forward.jl`: an `alloca` left in the IR
            # made the forward 5-6x slower on an M1 Pro, with correct results — so only this check
            # would ever notice it coming back.
            for stage in LTAt2.FLASH_STAGES, cfg in ((; tg=128, kt=32, sb=4), (; tg=64, kt=16, sb=8))
                # What `KernelDispatch.ka_compile` compiles: KA's context from KernelDispatch' probe, then the
                # stage's argument types with every buffer a raw device pointer.
                ctx, _ = KernelDispatch.ka_context(KernelDispatch.MetalBackendTag(), (1, 1, 1), cfg.tg)
                ptr(A::Type{<:AbstractArray}) = Core.LLVMPtr{eltype(A),Metal.AS.Device}
                ptr(T) = T
                tt = Tuple{typeof(ctx),map(ptr, LTAt2.kernel_argtypes(stage, Float32, dim, cfg))...}
                obj = LTAt2.stage_kernel(stage)(MetalBackend(), (cfg.tg, 1, 1))
                job = GPUCompiler.CompilerJob(GPUCompiler.methodinstance(typeof(obj.f), tt),
                    Metal.compiler_config(Metal.device(); name="probe", kernel=true))
                ir = sprint(io -> GPUCompiler.code_llvm(io, job; optimize=true, dump_module=true))
                @test !occursin("alloca", ir)
            end
        end

        @testset "the correctness gate passes real kernels and fails wrong answers" begin
            dim = 8
            V = LTAt2.verification_problem(dim, Float32)
            # The same launch judged against a reference that is off by 1%: must be rejected.
            off = merge(V, (; ref=map(x -> x .* 1.01, V.ref)))
            points = ((LTAt2.FlashScalar(), s -> s === :fwd ? (; tg=32, kt=16, sb=2) : (; tg=32, kt=16)),
                (LTAt2.FlashMMA(), s -> (; nsg=2, r=1, tile=16)))
            for (k, p) in points, stage in LTAt2.tune_stages(k)
                b = LTAt2.build_stage(k, stage, Float32, dim, p(stage))
                @test LTAt2.verify_stage(k, stage, b, V)
                @test !LTAt2.verify_stage(k, stage, b, off)
            end
        end

        # `:flash_mma`'s operand type for the second matrix products (`ops`), forced, at 16-bit storage.
        # The tuner chooses among these; each must pass the gate on its own merits.
        @testset "operand types at 16-bit storage" begin
            dim, BF = 16, LTAt2.BFloat16
            mma(T, ops) = s -> LTAt2.build_stage(LTAt2.FlashMMA(), s, T, dim, (; nsg=2, r=1, tile=16, ops))
            for (T, ops) in ((Float16, Float32), (Float16, Float16), (BF, Float32))
                V = LTAt2.verification_problem(dim, T)
                for stage in LTAt2.FLASH_MMA_STAGES
                    @test LTAt2.verify_stage(LTAt2.FlashMMA(), stage, mma(T, ops)(stage), V)
                end
            end
            # BFloat16 operands: Apple's compiler for the M1 GPU cannot build a bfloat fragment from
            # register values (it computes zeros; see `ext/LuxTriangleAttentionMetalExt/flash_mma.jl`), so the gate rejects
            # them and BFloat16 storage runs with Float32 operands here. Broken until Apple's is fixed.
            V = LTAt2.verification_problem(dim, BF)
            @test_broken LTAt2.verify_stage(LTAt2.FlashMMA(), :fwd, mma(BF, BF)(:fwd), V)
        end

        @testset "a tuned variant carries each binary's own configuration" begin
            dim, n = 8, 48
            vs = tune_flash_kernel!(n; dim, T=Float32)
            cfgs = LTAt2.flash_config(Float32, dim, n)
            # One variant per bias/mask presence, each with all four binaries at the tuned configs.
            for hb in (true, false), hm in (true, false)
                v = vs[LTAt2._attention_variant_key(LTAt2.FlashScalar(), Float32, dim, n; hasbias=hb, hasmask=hm)]
                @test sort(collect(keys(v))) == sort(collect(LTAt2.FLASH_STAGES))
                for (s, b) in v
                    @test b.threadgroup == cfgs[s].tg == b.tile[1]
                    @test b.is_ka && b.tile == (b.threadgroup, 1, 1)
                    @test b.params.tg == cfgs[s].tg && b.params.kt == cfgs[s].kt     # the binary's recorded tuning point
                    @test isfile(b.file)
                end
            end
        end
    end
else
    @warn "Metal not functional; skipping Metal tuning tests" maxlog = 1
end
