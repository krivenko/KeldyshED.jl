# Single-particle Green's functions

```@meta
CurrentModule = KeldyshED
```

## `computegf()`

```@docs
computegf(::EDCore,
          ::BranchPoint, ::BranchPoint,
          ::IndicesType, ::IndicesType,
          ::Float64)
computegf(::EDCore,
          ::BranchPoint, ::BranchPoint,
          ::IndicesType, ::IndicesType;
          en, ρ)
computegf(::EDCore,
          ::FullTimeGrid,
          ::IndicesType, ::IndicesType;
          ::AbstractGFFiller)
computegf(::EDCore,
          ::KeldyshTimeGrid,
          ::IndicesType, ::IndicesType,
          ::Float64;
          ::AbstractGFFiller)
computegf(::EDCore,
          ::ImaginaryTimeGrid,
          ::IndicesType, ::IndicesType;
          ::AbstractGFFiller)
computegf(::EDCore,
          ::FullTimeGrid,
          ::Vector{Tuple{IndicesType, IndicesType}};
          ::AbstractGFFiller)
computegf(::EDCore,
          ::KeldyshTimeGrid,
          ::Vector{Tuple{IndicesType, IndicesType}},
          ::Float64;
          ::AbstractGFFiller)
computegf(::EDCore,
          ::ImaginaryTimeGrid,
          ::Vector{Tuple{IndicesType, IndicesType}};
          ::AbstractGFFiller)
computegf(::EDCore,
          ::FullTimeGrid,
          ::Vector{IndicesType}, ::Vector{IndicesType};
          ::AbstractGFFiller)
computegf(::EDCore,
          ::KeldyshTimeGrid,
          ::Vector{IndicesType}, ::Vector{IndicesType},
          ::Float64;
          ::AbstractGFFiller)
computegf(::EDCore,
          ::ImaginaryTimeGrid,
          ::Vector{IndicesType}, ::Vector{IndicesType};
          ::AbstractGFFiller)
```

## Auxiliary types

```@docs
AbstractGFFiller
SerialGFFiller
DistributedGFFiller
```