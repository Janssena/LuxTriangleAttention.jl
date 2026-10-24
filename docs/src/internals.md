# Internals

```@meta
CurrentModule = LuxTriangleAttention
```

The docstrings of everything that is neither exported nor `public`: the attention routing, the tuner, the stage
launcher and the kernel sources. Documented for people reading or extending the package, and **not** covered by any
compatibility promise. The [Kernel operands](kernel-operands.md) page explains the data these functions move around.

```@autodocs
Modules = [LuxTriangleAttention]
Public = false
```
