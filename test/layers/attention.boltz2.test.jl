# `Attention` against Boltz-2's `Attention` (test/python/boltz2.jl). The sweep is `attention_parity`
# (test/support/parity.jl); this says what is specific to the Julia side of the reference: no bias terms. How Python
# is built and called is in test/python/references.jl.

attention_parity("boltz2"; use_bias=false)
