# The sweeps behind the Python parity files, `test/layers/<layer>.<model>.test.jl`.
#
# A parity file says what is specific to the Julia side of one reference (its bias terms, the width of its hidden
# layer); the sweep says what is compared: the Julia layer against the Python one for every precision, every
# case of the reference, and every configuration of the Julia layer that must not change the result (the parameter
# layout, chunking). Python runs once per (precision, case, parameter layout) and is the ground truth.
#
# Python is not run by the tests: its outputs are recorded in `test/fixtures/python/<component>.<model>.safetensors`
# (support/fixtures.jl) and the sweeps read them back. What is specific to the Python side (the class, how the weights
# are copied in, how it is called) is in `test/python/references.jl`, loaded only to record the outputs again.
#
# Only the native layer on the host is swept. Other backends are not: Julia's own array code is trusted there, and
# tracing the same layer in another precision is Reactant's to get right. What is ours is tested against the native layer
# instead: the fused kernels (`attention.gpu.test.jl`, `test/ext/reactant/attention_kernel.test.jl`) and the chunked attention of the Reactant
# extension (`test/ext/reactant/chunk.test.jl`).

const PARITY_TYPES = (Float16, Float32, Float64)

"The relative L2 tolerance of a layer output against the Python one: `isapprox`'s default."
parity_rtol(::Type{T}) where {T} = sqrt(eps(T))

"Whether the parity unit (component, model) is selected."
parity_selected(component, model) = selected(component; model)

# ── Attention ─────────────────────────────────────────────────────────────────────────────────────

"""
    attention_parity(model; use_bias, make_py, call_py)

`Attention` against one Python reference, over the three input shapes the references use, an all-ones and a random
mask, every precision, `fuse_qkv` (the parameter layout; the weights are copied into Python from whichever the layer
has) and `chunk_size` (whole, or in slices of `chunk` along the batch axis).

The Python side is `make_py(chn, head_dim, num_heads)`, which builds the Python layer, and
`call_py(py_layer, x_py, mask, mask_additive_py, bias_py, T)`, which calls it, since the references take their mask and
bias differently (`test/python/references.jl`).
"""
function attention_parity(model::AbstractString; use_bias)
    parity_selected("attention", model) || return
    refs = References("attention", model)
    rng = Random.Xoshiro(42)
    N, S, B = 16, 8, 2
    chn_in, num_heads, head_dim = 32, 6, 8

    dims_cfg = (
        ("3D inputs", :kq, (N, B), 1),
        ("4D inputs (symmetric)", :qk, (N, N, B), 3),          # as in TriangleAttention
        ("4D inputs (non-symmetric)", :kq, (N, S, B), 3),      # as in AttentionPairBias
    )
    @testset verbose = VERBOSE "$model" begin
        for (dim_name, bias_layout, dims, chunk) in dims_cfg
            @testset verbose = VERBOSE "$dim_name" begin
                mask_cfg = (("All-ones mask", trues(dims...)), ("Random mask", rand(rng, Bool, dims...)))
                for (mname, mask) in mask_cfg
                    @testset verbose = VERBOSE "$mname" begin
                        for T in PARITY_TYPES
                            inf_val = T == Float16 ? Float16(1e4) : T(1e9)
                            x = randn(rng, T, chn_in, dims...)                      # C, dims...
                            bias = randn(rng, T, num_heads, N, N, B)
                            # The symmetric input is attended over its other token axis (see `Attention`).
                            x_jl = dim_name == "4D inputs (symmetric)" ? permutedims(x, (1, 3, 2, 4)) : x

                            for fuse_qkv in (true, false)
                                layer0 = Attention(chn_in, head_dim, num_heads; use_bias, bias_layout, fuse_qkv)
                                ps0, st0 = Lux.setup(rng, layer0)
                                ps, st = (weights(refs, "$dim_name/fuse_qkv=$fuse_qkv", ps0), st0) |> convert_types(T)
                                y_ref = reference(refs, "$dim_name/$mname/$T/fuse_qkv=$fuse_qkv", fingerprint(x, bias, mask)) do py
                                    bias_prep = prep_bias(bias, x, static(bias_layout))
                                    x_py = if dim_name == "4D inputs (non-symmetric)"
                                        to_py(permutedims(x, reverse(1:4)); swap_batch_dim=false)   # Python takes B, S, N, C
                                    else
                                        to_py(x; swap_batch_dim=true)
                                    end
                                    mask_prep = prep_mask(mask)
                                    mask_py = permutedims(mask_prep, reverse(1:ndims(mask_prep)))
                                    mask_add_py = to_py(inf_val .* (mask_py .- one(T)); swap_batch_dim=false).to(py_dtype(T))
                                    bias_py = to_py(permutedims(bias_prep, reverse(1:ndims(bias_prep))); swap_batch_dim=false)
                                    py_layer = py.make_py(chn_in, head_dim, num_heads)
                                    sync_attention!(py_layer, ps)
                                    y_py = py.call_py(py_layer, x_py, mask, mask_add_py, bias_py, T)
                                    permutedims(to_jl(y_py; swap_batch_dim=false), reverse(1:ndims(x_jl)))
                                end

                                for chunk_size in (nothing, chunk)
                                    layer = Attention(chn_in, head_dim, num_heads; use_bias, bias_layout, fuse_qkv, chunk_size)
                                    @testset "$T, fuse_qkv=$fuse_qkv, chunk_size=$chunk_size" begin
                                        y, _ = layer(x_jl, bias, mask, ps, st)
                                        @test eltype(y) === T
                                        @test isapprox(y, y_ref; rtol=parity_rtol(T))
                                        @test_nowarn @inferred layer(x_jl, bias, mask, ps, st)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    save(refs)
    return nothing
end

# ── TriangleAttention ─────────────────────────────────────────────────────────────────────────────

"""
    triangle_attention_parity(model; use_bias, make_py, sync_py)

`TriangleAttention` against one Python reference, for both directions, with and without a mask, every precision,
`fuse_qkv` and `chunk_size`.

The Python side is `make_py(chn_in, chn_hidden, num_heads, is_starting, inf)`, which builds the Python layer, and
`sync_py(py_layer, ps)`, which copies the parameters into it (`test/python/references.jl`).
"""
function triangle_attention_parity(model::AbstractString; use_bias)
    parity_selected("triangle_attention", model) || return
    refs = References("triangle_attention", model)
    rng = Random.Xoshiro(42)
    chn_in, N, B = 64, 8, 2
    chn_hidden, num_heads = 32, 4                     # chn_hidden is the head dimension
    @testset verbose = VERBOSE "$model" begin
        for T in PARITY_TYPES
            inf_val = T == Float16 ? Float16(1e4) : T(1e9)
            x = randn(rng, T, chn_in, N, N, B)
            for (mname, mask) in (("No mask", nothing), ("Random mask", rand(rng, Bool, N, N, B)))
                @testset verbose = VERBOSE "TriangleAttention ($mname)" begin
                    for is_starting in (true, false), fuse_qkv in (true, false)
                        layer0 = TriangleAttention(chn_in, chn_hidden, num_heads; is_starting, use_bias, fuse_qkv)
                        ps0, st0 = Lux.setup(rng, layer0)
                        ps, st = (weights(refs, "$(is_starting ? "starting" : "ending")/fuse_qkv=$fuse_qkv", ps0), st0) |> convert_types(T)
                        key = "$mname/$T/$(is_starting ? "starting" : "ending")/fuse_qkv=$fuse_qkv"
                        y_ref = reference(refs, key, fingerprint(x, mask)) do py
                            x_py = to_py(x; swap_batch_dim=true)
                            mask_py = isnothing(mask) ? nothing : to_py(permutedims(mask, (3, 1, 2))).to(py_dtype(T))   # Python wants floats
                            py_layer = py.make_py(chn_in, chn_hidden, num_heads, is_starting, inf_val)
                            py.sync_py(py_layer, ps)
                            to_jl(py_layer(x_py, mask_py); swap_batch_dim=true)
                        end

                        for chunk_size in (nothing, 3)
                            layer = TriangleAttention(chn_in, chn_hidden, num_heads; is_starting, use_bias, fuse_qkv, chunk_size)
                            @testset "$T, $(is_starting ? "starting" : "ending"), fuse_qkv=$fuse_qkv, chunk_size=$chunk_size" begin
                                y, _ = layer(x, mask, ps, st)
                                @test eltype(y) === T
                                @test isapprox(y, y_ref; rtol=parity_rtol(T))
                                @test_nowarn @inferred layer(x, mask, ps, st)
                            end
                        end
                    end
                end
            end
        end
    end
    save(refs)
    return nothing
end

# ── TriangleMultiplication ────────────────────────────────────────────────────────────────────────

"""
    trimul_parity(model; use_bias, chn_hidden, make_py, sync_py, mask_py, upcasts=false)

`TriangleMultiplication` against one Python reference, outgoing and incoming, with and without a mask, every
precision and all four parameter layouts (`fused` × `fused_glu`). In Float16 the layer is also run `Folded`: the
parameters prepared by `fold_trimul`, which changes how the contraction is carried out and not the result.

The Python side is `make_py(is_outgoing, chn_in, chn_hidden)`, which builds the Python layer, `sync_py(py_layer, ps)`,
which copies the parameters into it, and `mask_py(mask, T, N, B)`, which gives the mask in the form it takes
(`test/python/references.jl`). A reference that upcasts to Float32 inside
(`upcasts=true`) cannot be run in Float64 either, since PyTorch refuses its mixed dtypes; it runs in Float32 there, and the
Float64 layer is compared with it to Float32's tolerance.
"""
function trimul_parity(model::AbstractString; use_bias, chn_hidden, upcasts=false)
    parity_selected("triangle_multiplication", model) || return
    refs = References("triangle_multiplication", model)
    rng = Random.Xoshiro(42)
    chn_in, N, B = 64, 16, 2
    @testset verbose = VERBOSE "$model" begin
        for T in PARITY_TYPES
            x = randn(rng, T, chn_in, N, N, B)
            # The references upcast to Float32 inside (`.float()`) but keep their parameters as they were made, which
            # PyTorch refuses in half precision (`mixed dtype`). So at Float16 Python runs in Float32 on the same
            # Float16-rounded numbers, and the Julia layer is compared with that.
            Tpy = T === Float16 || (upcasts && T === Float64) ? Float32 : T
            rtol = Tpy === T ? parity_rtol(T) : max(parity_rtol(T), parity_rtol(Float32))
            for (mname, mask) in (("No mask", nothing), ("Random mask", rand(rng, Bool, N, N, B)))
                @testset verbose = VERBOSE "TriangleMultiplication ($mname)" begin
                    for is_outgoing in (true, false), (fused, fused_glu) in ((true, true), (true, false), (false, true), (false, false))
                        layer = TriangleMultiplication(chn_in, chn_hidden; is_outgoing, use_bias, fused, fused_glu)
                        ps32, st32 = Lux.setup(rng, layer)
                        ps32 = weights(refs, "$(is_outgoing ? "outgoing" : "incoming")/fused=$fused/fused_glu=$fused_glu", ps32)
                        ps, st = (ps32, st32) |> convert_types(T)
                        key = "$mname/$T/$(is_outgoing ? "outgoing" : "incoming")/fused=$fused/fused_glu=$fused_glu"
                        y_ref = reference(refs, key, fingerprint(x, mask)) do py
                            x_py = to_py(Tpy.(x); swap_batch_dim=true)
                            py_layer = py.make_py(is_outgoing, chn_in, chn_hidden)
                            py.sync_py(py_layer, Tpy === T ? ps : Lux.f32(ps))
                            to_jl(py_layer(x_py, py.mask_py(mask, Tpy, N, B)); swap_batch_dim=true)
                        end

                        # (name, k): Float16 data takes the contraction in Float32 by default; folding scales the weights
                        # by 2⁻ᵏ in Float32 before they are cast, and the layer undoes it.
                        modes = T === Float16 ? (("widened", nothing), ("Folded(2)", 2)) : (("plain", nothing),)
                        for (modename, k) in modes
                            psm, stm = k === nothing ? (ps, st) : (fold_trimul(ps32, st32, k) |> convert_types(T))
                            @testset "$T, $(is_outgoing ? "outgoing" : "incoming"), fused=$fused fused_glu=$fused_glu, $modename" begin
                                y, _ = layer(x, mask, psm, stm)
                                @test eltype(y) === T
                                @test isapprox(Tpy.(y), y_ref; rtol)
                                @test_nowarn @inferred layer(x, mask, psm, stm)
                            end
                        end
                    end
                end
            end
        end
    end
    save(refs)
    return nothing
end
