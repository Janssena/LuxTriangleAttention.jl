# `TriangleMultiplication` against AlphaFold3's `TriangleMultiplication` (test/python/alphafold3.jl). The sweep is
# `trimul_parity` (test/support/parity.jl). AlphaFold3 projects `a` and `b` separately, with a bias on every layer, and
# its module takes a mask always, so no mask is a mask of ones. How Python is built is in test/python/references.jl.

trimul_parity("af3"; use_bias=true, chn_hidden=32)
