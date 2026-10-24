# `TriangleMultiplication` against AlphaFold2's `TriangleMultiplicativeUpdate` (test/python/alphafold2.jl). The sweep is
# `trimul_parity` (test/support/parity.jl). AlphaFold2 projects `a` and `b` separately, with a bias on every layer.

include("../python/alphafold2.jl")

trimul_parity("af2";
    use_bias=true,
    chn_hidden=32,
    make_py=(out, c, h) -> py"AF2TriangleMultiplicativeUpdate"(c, h, _outgoing=out),
    sync_py=(py, ps) -> sync_trimul!(py, ps; norm_in=:layer_norm_in, norm_out=:layer_norm_out,
        a=(linear=:linear_a_p, gate=:linear_a_g), b=(linear=:linear_b_p, gate=:linear_b_g),
        out=(linear=:linear_z, gate=:linear_g)),
    mask_py=(mask, T, N, B) -> isnothing(mask) ? nothing :
        to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T)))
