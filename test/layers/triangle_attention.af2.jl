include("../python/alphafold2.jl")

function copy_weights_to_af2_attention!(
    py_attn::PyObject,
    ps::NamedTuple
)   
    sync_dense!(py_attn.linear_q, ps.linear_q)
    sync_dense!(py_attn.linear_k, ps.linear_k)
    sync_dense!(py_attn.linear_v, ps.linear_v)
    sync_dense!(py_attn.linear_o, ps.linear_out)

    if !isempty(ps.gate)
        sync_dense!(py_attn.linear_g, ps.gate)
    end

    return nothing
end

@testset "TriAttnCore vs AF2 Attention Parity" begin
    rng = Random.Xoshiro(42)
    T = Float32
    chn_in, N, B = 64, 8, 2
    chn_hidden = 16 # == D
    num_heads = 4 # == H
    inf_val = LuxTriangleAttention._safe_inf(T)

    jl_layer = TriAttnCore(chn_in, chn_hidden, num_heads; inf=inf_val, qkv_use_bias=false, gate_use_bias=true, out_use_bias=true)
    ps, st = LuxTriangleAttention.Lux.setup(rng, jl_layer)

    py_attn = py"AF2Attention"(chn_in, chn_in, chn_in, chn_hidden, num_heads)

    copy_weights_to_af2_attention!(py_attn, ps)

    x = randn(rng, T, chn_in, N, N, B)
    bias = randn(rng, T, num_heads, N, N, B)
    mask = rand(rng, Bool, N, N, B)

    y_jl, _ = jl_layer(x, bias, mask, ps, st)

    x_py = to_py(x; swap_batch_dim=true)
    mask_py = to_py(permutedims(mask, (3, 1, 2)))
    mask_py = inf_val * (mask_py - 1) # [B, N, N] 
    bias_py = to_py(bias; swap_batch_dim=true)
    
    py"""
    # mask in expected: [B, N, N]
    # bias in expected: [B, N, N, H]
    triangle_bias = af2_permute_final_dims($bias_py, (2, 0, 1))
    triangle_bias = triangle_bias.unsqueeze(-4) # [B, 1, H, N, N]
    mask_bias = $mask_py[..., :, None, None, :] # [B, N, 1, 1, N]
    biases = [mask_bias, triangle_bias]
    """
    
    py_attn.eval()
    py_out = py_attn(x_py, x_py; biases=py"biases")

    @test y_jl ≈ to_jl(py_out; swap_batch_dim=true)
end

