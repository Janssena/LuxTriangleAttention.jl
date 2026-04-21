include("../python/alphafold3.jl")

function copy_weights_to_af3_attention!(
    py_layer::PyObject,
    ps::NamedTuple
)   
    sync_dense!(py_layer.linear_q, ps.linear_q)
    sync_dense!(py_layer.linear_k, ps.linear_k)
    sync_dense!(py_layer.linear_v, ps.linear_v)
    sync_dense!(py_layer.linear_o, ps.linear_out)

    if !isempty(ps.gate)
        sync_dense!(py_layer.linear_g, ps.gate)
    end

    return nothing
end

function copy_weights_to_af3_triangle_attention!(
    py_layer::PyObject,
    ps::NamedTuple
)   
    sync_layernorm!(py_layer.layer_norm, ps.layer_norm)
    sync_dense!(py_layer.linear_z, ps.linear)
    copy_weights_to_af3_attention!(py_layer.mha, ps.mha)

    return nothing
end

@testset "AlphaFold3" begin
    # setup
    rng = Random.Xoshiro(42)
    T = Float32
    chn_in, N, B = 64, 8, 2
    chn_hidden = 32 # == D
    num_heads = 4 # == H
    inf_val = LuxTriangleAttention._safe_inf(T)
    
    x = randn(rng, T, chn_in, N, N, B)
    bias = randn(rng, T, num_heads, N, N, B)
    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(rng, Bool, N, N, B))
    )
    
    for (name, mask) in mask_cfg
        @testset "AF3 TriAttnCore ($name)" begin
            jl_layer = TriAttnCore(chn_in, chn_hidden, num_heads; inf=inf_val, qkv_use_bias=false, gate_use_bias=false, out_use_bias=false)
            ps, st = Lux.setup(rng, jl_layer)
    
            py_layer = py"AF3Attention"(chn_in, chn_in, chn_in, chn_hidden, num_heads)
            
            copy_weights_to_af3_attention!(py_layer, ps)
    
            y_jl, _ = jl_layer(x, bias, mask, ps, st)
    
            x_py = to_py(x; swap_batch_dim=true)
            if !isnothing(mask)
                mask_py = to_py(permutedims(mask, (3, 1, 2)))
                mask_py = inf_val * (mask_py - 1) # [B, N, N] 
            else
                mask_py = nothing
            end
            bias_py = to_py(bias; swap_batch_dim=true)
            
            py"""
            # mask in expected: [B, N, N]
            # bias in expected: [B, N, N, H]
            triangle_bias = af3_permute_final_dims($bias_py, (2, 0, 1))
            triangle_bias = triangle_bias.unsqueeze(-4) # [B, 1, H, N, N]
            if $mask_py is None:
                mask = $x_py.new_ones($x_py.shape[:-1])
                mask = $inf_val * (mask - 1)
            else:
                mask = $mask_py
                
            mask_bias = mask[..., :, None, None, :] # [B, N, 1, 1, N]
            biases = [mask_bias, triangle_bias]
            """
            
            py_layer.eval()
            py_out = py_layer(x_py, x_py; biases=py"biases")
    
            @testset "Parity" begin
                @test y_jl ≈ to_jl(py_out; swap_batch_dim=true)
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_layer(x, bias, mask, ps, st)
            end
        end
    
        @testset "AF3 TriangleAttention ($name)" begin
            for is_starting in [true, false]
                jl_layer = TriangleAttention(chn_in, chn_hidden, num_heads; is_starting, use_bias=false, inf=inf_val, qkv_use_bias=false, gate_use_bias=false, out_use_bias=false)
                ps, st = Lux.setup(rng, jl_layer)
    
                py_layer = py"AF3TriangleAttention"(chn_in, chn_hidden, num_heads, starting=is_starting, inf=inf_val)
                copy_weights_to_af3_triangle_attention!(py_layer, ps)
                py_layer.eval()
                
                x_py = to_py(x; swap_batch_dim=true)
                mask_py = isnothing(mask) ? nothing : to_py(permutedims(mask, (3, 1, 2)))
                
                y_jl, _ = jl_layer(x, mask, ps, st)
                py_out = py_layer(x_py, mask_py)
    
                @testset "Parity" begin
                    @test y_jl ≈ to_jl(py_out; swap_batch_dim=true)
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(x, mask, ps, st)
                end
            end
        end
    end
end