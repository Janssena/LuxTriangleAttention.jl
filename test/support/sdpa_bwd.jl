# The reverse of `LuxTriangleAttention.SDPA()(q, k, v, mask, bias)` on device arrays, eager or traced:
# the selected kernel's `KernelDispatch.backward`. Each gradient comes back in its argument's shape and
# element type; `dbias` is `nothing` without a bias. Shared by the device tests and the kernel benchmarks,
# which `include` it after `using LuxTriangleAttention` (`test/runtests.jl` does for the tests).

import KernelDispatch

# `kernel` defaults to the one `use_kernel!` selected. A switch only takes effect from the next top-level expression,
# so code that picks the kernel and runs it in one function passes the kernel object instead.
function sdpa_bwd(q, k, v, mask, bias, o, lse, dO; kernel=nothing)
    op = LuxTriangleAttention.SDPA()
    kernel = isnothing(kernel) ? KernelDispatch.selected_kernel(op) : kernel
    g = KernelDispatch.backward(op, kernel, (q, k, v, mask, bias), (o, lse), (dO, nothing))
    return g[1], g[2], g[3], g[5]
end
