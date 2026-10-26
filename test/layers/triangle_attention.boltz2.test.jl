# `TriangleAttention` against Boltz-2's `TriangleAttention` (test/python/boltz2.jl). The sweep is
# `triangle_attention_parity` (test/support/parity.jl); how Python is built is in test/python/references.jl.

triangle_attention_parity("boltz2"; use_bias=(false, (layer_norm=true,)))
