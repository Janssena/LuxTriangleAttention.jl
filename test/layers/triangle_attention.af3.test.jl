# `TriangleAttention` against AlphaFold3's `TriangleAttention` (test/python/alphafold3.jl). The sweep is
# `triangle_attention_parity` (test/support/parity.jl); how Python is built is in test/python/references.jl.

triangle_attention_parity("af3"; use_bias=(false, (layer_norm=true,)))
