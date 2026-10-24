# `TriangleMultiplication` against AlphaFold3's `TriangleMultiplication` (test/python/alphafold3.jl). The sweep is
# `trimul_parity` (test/support/parity.jl). AlphaFold3 projects `a` and `b` separately, with a bias on every layer, and
# its module takes a mask always, so no mask is a mask of ones.

include("../python/alphafold3.jl")

trimul_parity("af3";
    use_bias=true,
    chn_hidden=32,
    make_py=(out, c, h) -> (out ? py"AF3TriangleMultiplicationOutgoing" : py"AF3TriangleMultiplicationIncoming")(c, h),
    sync_py=(py, ps) -> sync_trimul!(py, ps; norm_in=:layer_norm_in, norm_out=:layer_norm_out,
        a=(linear=:linear_a_p, gate=:linear_a_g), b=(linear=:linear_b_p, gate=:linear_b_g),
        out=(linear=:linear_z, gate=:linear_g)),
    mask_py=(mask, T, N, B) -> isnothing(mask) ? to_py(ones(T, N, N, B); swap_batch_dim=true) :
        to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T)))
