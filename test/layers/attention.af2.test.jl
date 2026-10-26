# `Attention` against AlphaFold2's `Attention` (test/python/alphafold2.jl). The sweep is `attention_parity`
# (test/support/parity.jl); this says what is specific to the Julia side of the reference: its bias terms. How Python
# is built and called is in test/python/references.jl.

attention_parity("af2"; use_bias=(false, (gate=true, out=true)))
