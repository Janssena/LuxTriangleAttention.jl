# The registry itself: registration, selection, the derived ABI order, and persistence.
#
# No GPU and no compiled kernel is needed for any of this — the files registered below are empty
# placeholders with the right extensions, because everything under test here is about the TABLE, not
# about running anything. That is deliberate: these are the checks that catch a malformed entry
# before it ever reaches a launcher, where the same mistakes are silent.

using Test, LuxTriangleAttention
const LTA = LuxTriangleAttention

# An isolated cache root, so these tests neither read nor clobber the user's real registry.
const _REG_TMP = mktempdir()

function with_tmp_registry(f)
    old = get(ENV, "LTA_KERNEL_CACHE", nothing)
    ENV["LTA_KERNEL_CACHE"] = _REG_TMP
    saved = copy(LTA._KERNELS)
    saved_sel = LTA.LTA_KERNEL[]
    try
        empty!(LTA._KERNELS)
        f()
    finally
        empty!(LTA._KERNELS)
        merge!(LTA._KERNELS, saved)
        LTA.LTA_KERNEL[] = saved_sel
        old === nothing ? delete!(ENV, "LTA_KERNEL_CACHE") : (ENV["LTA_KERNEL_CACHE"] = old)
    end
end

"""Empty files with a given extension, one per stage — enough to register, never launched."""
function fake_binaries(dir, ext; stages=(:fwd, :dq, :dk, :dv))
    mkpath(dir)
    return NamedTuple(s => (p = joinpath(dir, "$(s)$(ext)"); write(p, UInt8[]); p)
                      for s in stages)
end

@testset "kernel registry" begin
    @testset "the reserved name is not writable by users" begin
        with_tmp_registry() do
            f = fake_binaries(joinpath(_REG_TMP, "src1"), ".metallib")
            @test_throws ArgumentError register_kernel!(;
                name=LTA.RESERVED_KERNEL_NAME, fwd=f.fwd, dq=f.dq, dk=f.dk, dv=f.dv, dim=8)
        end
    end

    @testset "register, select, unregister" begin
        with_tmp_registry() do
            f = fake_binaries(joinpath(_REG_TMP, "src2"), ".metallib")
            e = register_kernel!(; name=:probe, fwd=f.fwd, dq=f.dq, dk=f.dk, dv=f.dv,
                dim=8, description="test entry")

            @test e.backend === :metal          # inferred from the extension, not declared
            @test e.format === :metallib
            @test e.keys == [:dtype, :dim]      # no `n` given, so not keyed on length
            @test length(e.variants) == 1
            @test LTA.has_kernel(:probe)

            # The binaries are COPIED into the cache, so the entry survives the originals moving.
            rm(joinpath(_REG_TMP, "src2"); recursive=true)
            @test all(isfile(LTA.stage_path(e, s)) for s in first(values(e.variants)))

            # Selection checks the backend is actually available.
            if :metal in LTA._LOADED_BACKENDS
                @test use_kernel!(:probe) === :probe
                @test LTA.current_kernel() === :probe
            else
                @test_throws ArgumentError use_kernel!(:probe)
            end

            @test_throws ArgumentError use_kernel!(:not_registered)

            @test unregister_kernel!(:probe)
            @test !LTA.has_kernel(:probe)
            # Unregistering the SELECTED kernel must leave a working selection, never an empty one.
            @test LTA.current_kernel() === LTA.RESERVED_KERNEL_NAME
            @test !unregister_kernel!(:probe)   # idempotent
        end
    end

    @testset "mixed binary formats are rejected" begin
        with_tmp_registry() do
            d = joinpath(_REG_TMP, "src3")
            mkpath(d)
            a = joinpath(d, "fwd.metallib"); write(a, UInt8[])
            b = joinpath(d, "dq.cubin"); write(b, UInt8[])
            c = joinpath(d, "dk.metallib"); write(c, UInt8[])
            e = joinpath(d, "dv.metallib"); write(e, UInt8[])
            @test_throws ArgumentError register_kernel!(;
                name=:mixed, fwd=a, dq=b, dk=c, dv=e, dim=8)
            @test_throws ArgumentError register_kernel!(;
                name=:missing_file, fwd=joinpath(d, "nope.metallib"), dim=8)
        end
    end

    @testset "each gradient is written exactly once" begin
        with_tmp_registry() do
            f = fake_binaries(joinpath(_REG_TMP, "src4"), ".metallib")
            roles(o, i) = (outputs=o, inputs=i, scalars=[:scale, :n, :nh, :mheads])
            # `:dv` written by nobody: the gradient would silently be whatever the fresh buffer held.
            @test_throws ArgumentError register_kernel!(; name=:incomplete, dim=8,
                stages=[(; name=:fwd, file=f.fwd,
                        roles([:o, :lse], [:q, :k, :v, :bias, :mask])...),
                    (; name=:dq, file=f.dq,
                        roles([:dq, :dbias], [:q, :k, :v, :bias, :mask, :dO, :lse_in])...),
                    (; name=:dk, file=f.dk,
                        roles([:dk], [:q, :k, :v, :bias, :mask, :dO, :lse_in])...)])

            # `:dq` written twice: one launch silently overwrites the other.
            @test_throws ArgumentError register_kernel!(; name=:twice, dim=8,
                stages=[(; name=:fwd, file=f.fwd,
                        roles([:o, :lse], [:q, :k, :v, :bias, :mask])...),
                    (; name=:dq, file=f.dq,
                        roles([:dq, :dbias], [:q, :k, :v, :bias, :mask, :dO, :lse_in])...),
                    (; name=:dk, file=f.dk,
                        roles([:dq, :dk, :dv], [:q, :k, :v, :bias, :mask, :dO, :lse_in])...)])
        end
    end

    @testset "a residual cannot be read before it is written" begin
        with_tmp_registry() do
            f = fake_binaries(joinpath(_REG_TMP, "src5"), ".metallib")
            roles(o, i) = (outputs=o, inputs=i, scalars=[:scale, :n, :nh, :mheads])
            @test_throws ArgumentError register_kernel!(; name=:outoforder, dim=8,
                stages=[(; name=:fwd, file=f.fwd,
                        roles([:o, :lse], [:q, :k, :v, :bias, :mask])...),
                    # reads `delta`…
                    (; name=:dq, file=f.dq,
                        roles([:dq, :dbias], [:q, :k, :v, :bias, :mask, :dO, :delta_in])...),
                    # …which is not produced until here.
                    (; name=:dk, file=f.dk,
                        roles([:dk, :dv, :delta], [:q, :k, :v, :bias, :mask, :dO])...)])
        end
    end

    @testset "unknown roles and grid axes are rejected" begin
        with_tmp_registry() do
            f = fake_binaries(joinpath(_REG_TMP, "src6"), ".metallib")
            @test_throws ArgumentError register_kernel!(; name=:badrole, dim=8,
                stages=[(; name=:fwd, file=f.fwd, outputs=[:o, :lse],
                    inputs=[:q, :k, :v, :bias, :nonsense], scalars=[:scale])])
            @test_throws ArgumentError register_kernel!(; name=:badgrid, dim=8,
                stages=[(; name=:fwd, file=f.fwd, outputs=[:o, :lse],
                    inputs=[:q, :k, :v, :bias, :mask], scalars=[:scale],
                    grid=(:nonsense, :one, :one))])
        end
    end

    @testset "the ABI order is derived, not taken from the caller" begin
        with_tmp_registry() do
            f = fake_binaries(joinpath(_REG_TMP, "src7"), ".metallib")
            # Roles given in a DELIBERATELY scrambled order; the table must come out canonical.
            e = register_kernel!(; name=:ordered, dim=8,
                stages=[(; name=:fwd, file=f.fwd,
                    outputs=[:lse, :o], inputs=[:mask, :v, :q, :bias, :k],
                    scalars=[:mheads, :scale, :n, :neg_inf, :nh])])
            st = only(first(values(e.variants)))
            @test LTA.abi_order(e, st) == [
                :out => :o, :out => :lse,
                :in => :q, :in => :k, :in => :v, :in => :bias, :in => :mask,
                :scalar => :scale, :scalar => :neg_inf, :scalar => :n,
                :scalar => :nh, :scalar => :mheads]
        end
    end

    @testset "variant keying and nbucket fallback" begin
        with_tmp_registry() do
            f = fake_binaries(joinpath(_REG_TMP, "src8"), ".metallib")
            e = register_kernel!(; name=:buckets, fwd=f.fwd, dq=f.dq, dk=f.dk, dv=f.dv,
                dim=8, n=128)
            @test e.keys == [:dtype, :dim, :nbucket]
            @test LTA.nbucket(100) == 128
            @test LTA.nbucket(128) == 128
            @test LTA.nbucket(129) == 256

            @test LTA.resolve_variant(e, Float32, 8, 128) !== nothing
            @test LTA.resolve_variant(e, Float32, 8, 100) !== nothing   # same bucket
            @test LTA.resolve_variant(e, Float32, 8, 500) !== nothing   # falls back to 128
            # A different dtype or head dim is a different ABI and must NEVER be substituted.
            @test LTA.resolve_variant(e, Float16, 8, 128) === nothing
            @test LTA.resolve_variant(e, Float32, 16, 128) === nothing

            register_kernel!(; name=:buckets, fwd=f.fwd, dq=f.dq, dk=f.dk, dv=f.dv,
                dim=8, n=512)
            @test length(LTA.kernel_entry(:buckets).variants) == 2
        end
    end

    @testset "variants do not overwrite each other's binaries" begin
        # The bug this guards: every stage used to be copied in as `<name>_<stage>.<ext>`, so tuning
        # dim=16 and then dim=32 left both variants in the manifest pointing at the SECOND binary.
        with_tmp_registry() do
            a = fake_binaries(joinpath(_REG_TMP, "src10a"), ".metallib")
            b = fake_binaries(joinpath(_REG_TMP, "src10b"), ".metallib")
            write(b.fwd, UInt8[0x01])   # distinguishable content, so a swap is visible

            register_kernel!(; name=:dims, fwd=a.fwd, dq=a.dq, dk=a.dk, dv=a.dv, dim=16)
            e = register_kernel!(; name=:dims, fwd=b.fwd, dq=b.dq, dk=b.dk, dv=b.dv, dim=32)

            s16 = LTA.find_stage(LTA.resolve_variant(e, Float32, 16, 128), :fwd)
            s32 = LTA.find_stage(LTA.resolve_variant(e, Float32, 32, 128), :fwd)
            @test s16.id != s32.id
            @test s16.file != s32.file
            @test occursin(s16.id, s16.file) && occursin(s32.id, s32.file)
            # Both binaries are still there, and each variant still holds its OWN bytes.
            @test read(LTA.stage_path(e, s16)) == UInt8[]
            @test read(LTA.stage_path(e, s32)) == UInt8[0x01]

            # Re-registering the same variant with a NEW binary replaces it and prunes the old file.
            stale = LTA.stage_path(e, s32)
            write(b.fwd, UInt8[0x02])
            e = register_kernel!(; name=:dims, fwd=b.fwd, dq=b.dq, dk=b.dk, dv=b.dv, dim=32)
            s32b = LTA.find_stage(LTA.resolve_variant(e, Float32, 32, 128), :fwd)
            @test read(LTA.stage_path(e, s32b)) == UInt8[0x02]
            @test !isfile(stale)
            @test length(e.variants) == 2
            # …and the dim=16 variant is untouched by any of it.
            @test isfile(LTA.stage_path(e, s16))

            # The id round-trips through the manifest.
            empty!(LTA._KERNELS)
            LTA.load_registry!()
            r = LTA.kernel_entry(:dims)
            @test LTA.find_stage(LTA.resolve_variant(r, Float32, 16, 128), :fwd).id == s16.id
        end
    end

    @testset "the registry survives a restart" begin
        with_tmp_registry() do
            f = fake_binaries(joinpath(_REG_TMP, "src9"), ".metallib")
            register_kernel!(; name=:persisted, fwd=f.fwd, dq=f.dq, dk=f.dk, dv=f.dv,
                dim=8, n=64, description="round-trip")

            # Simulate a fresh process: drop everything in memory, then rehydrate from disk.
            empty!(LTA._KERNELS)
            @test !LTA.has_kernel(:persisted)
            @test :persisted in LTA.load_registry!()

            e = LTA.kernel_entry(:persisted)
            @test e.backend === :metal
            @test e.keys == [:dtype, :dim, :nbucket]
            @test e.description == "round-trip"
            st = LTA.find_stage(LTA.resolve_variant(e, Float32, 8, 64), :dq)
            @test st !== nothing
            @test st.outputs == [:dq, :dbias]
            # The backward takes no `neg_inf`; declaring one would shift `n` and silently zero every
            # gradient, which is exactly the bug this round-trip guards.
            @test :neg_inf ∉ st.scalars

            # A manifest naming a binary that has gone missing must be dropped, not selected and
            # then failed at launch.
            rm(LTA.stage_path(e, st))
            empty!(LTA._KERNELS)
            LTA.load_registry!()
            @test !LTA.has_kernel(:persisted)
        end
    end

    @testset "a `:default` manifest from other kernel sources is dropped" begin
        # The registry holds COPIES of the binaries, so an edited kernel would otherwise keep being
        # served from it. A user entry is never checked this way.
        with_tmp_registry() do
            f = fake_binaries(joinpath(_REG_TMP, "src11"), ".metallib")
            stage(name, file) = (; name, file, LTA._DEFAULT_ROLES[name]...)
            LTA._register!(LTA.RESERVED_KERNEL_NAME;
                fwd=nothing, dq=nothing, dk=nothing, dv=nothing,
                stages=[stage(s, getfield(f, s)) for s in (:fwd, :dq, :dk, :dv)],
                dim=8, dtype=Float32, n=64, layout=LTA._DEFAULT_LAYOUT, rowmajor=false,
                threadgroup=32, grid=LTA.FLASH_GRID, preamble=LTA.FLASH_PREAMBLE,
                description="")
            register_kernel!(; name=:user, fwd=f.fwd, dq=f.dq, dk=f.dk, dv=f.dv, dim=8)

            path = LTA._manifest_path(LTA.RESERVED_KERNEL_NAME)
            empty!(LTA._KERNELS)
            @test LTA.RESERVED_KERNEL_NAME in LTA.load_registry!()   # current sources: kept

            write(path, replace(read(path, String),
                "source_tag = \"$(LTA.kernel_source_tag())\"" => "source_tag = \"stale\""))
            empty!(LTA._KERNELS)
            loaded = LTA.load_registry!()
            @test LTA.RESERVED_KERNEL_NAME ∉ loaded
            @test :user in loaded
        end
    end
end
