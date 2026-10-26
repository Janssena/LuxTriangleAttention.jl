# `Attention` against AlphaFold3's `Attention` (test/python/alphafold3.jl). The sweep is `attention_parity`
# (test/support/parity.jl); this says what is specific to the Julia side of the reference: no bias terms. How Python
# is built and called is in test/python/references.jl.

attention_parity("af3"; use_bias=false)
