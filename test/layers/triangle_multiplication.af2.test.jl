# `TriangleMultiplication` against AlphaFold2's `TriangleMultiplicativeUpdate` (test/python/alphafold2.jl). The sweep is
# `trimul_parity` (test/support/parity.jl). AlphaFold2 projects `a` and `b` separately, with a bias on every layer.
# How Python is built is in test/python/references.jl.

trimul_parity("af2"; use_bias=true, chn_hidden=32)
