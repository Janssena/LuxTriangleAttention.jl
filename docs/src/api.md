# API reference

```@meta
CurrentModule = LuxTriangleAttention
```

Every exported and public name. Names starting with an underscore are internal and may change; they are listed on
the [Internals](internals.md) page.

## Layers

```@docs
TriangleAttention
TriangleMultiplication
Attention
TriMulCore
GatedLinearUnit
SwiGLU
ChannelsLast
```

## Masks and bias

```@docs
prep_mask
prep_bias
```

## Normalisation

```@docs
StableLayerNorm
LayerNormNoBias
layernorm_f32stats
```

## Precision

Public, not exported.

```@docs
ishalf
mask_value
fold_k
fold_trimul
ContractionMode
FullPrecision
HalfSafe
Folded
contraction_mode
```

## Chunked attention

Both need Reactant loaded.

```@docs
chunked_compile_options
check_rolled
```

## Attention kernels

```@docs
SDPA
use_kernel!
current_kernel
list_kernels
device_kernel_available
```

## Tuning

```@docs
tune_flash_kernel!
tune_flash_mma!
calibrate!
calibrate_mma!
clear_calibration!
tune_lengths
set_tune_lengths!
```

## Other

```@docs
resolve_defaults
```

`backend_of`, which says which backend an array belongs to, is KernelDispatch': see its
[documentation](https://janssena.github.io/KernelDispatch.jl/dev/api/#Backends-and-devices).

Everything else in the package, such as the tuner, the stage launcher and the kernel sources, is on the
[Internals](internals.md) page.
