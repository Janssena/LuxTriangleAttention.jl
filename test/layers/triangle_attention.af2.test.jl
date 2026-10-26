# `TriangleAttention` against AlphaFold2's `TriangleAttention` (test/python/alphafold2.jl). The sweep is
# `triangle_attention_parity` (test/support/parity.jl); how Python is built is in test/python/references.jl.

triangle_attention_parity("af2"; use_bias=(layer_norm=true, linear=false, mha=(false, (gate=true, out=true))))
