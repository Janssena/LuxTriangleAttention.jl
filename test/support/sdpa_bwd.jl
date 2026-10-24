# The reverse of `LuxTriangleAttention.SDPA()(q, k, v, mask, bias)` on device arrays, eager or traced:
# the selected kernel's `KernelDispatch.backward`. Each gradient comes back in its argument's shape and
# element type; `dbias` is `nothing` without a bias. Shared by the device tests and the kernel benchmarks,
# which `include` it after `using LuxTriangleAttention` (`test/runtests.jl` does for the tests).

import KernelDispatch

function sdpa_bwd(q, k, v, mask, bias, o, lse, dO)
    op = LuxTriangleAttention.SDPA()
    g = KernelDispatch.backward(op, KernelDispatch.selected_kernel(op), (q, k, v, mask, bias),
        (o, lse), (dO, nothing))
    return g[1], g[2], g[3], g[5]
end
