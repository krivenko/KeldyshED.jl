# [Density matrix and evolution operators](@id Evolution-operators)

```@meta
CurrentModule = KeldyshED
```

```@docs
partition_function
density_matrix
evolution_operator
tofockbasis(::Vector{GF}, ::EDCore) where {GF <: AbstractTimeGF}
toeigenbasis(::Vector{GF}, ::EDCore) where {GF <: AbstractTimeGF}
partial_trace
reduced_density_matrix
reduced_evolution_operator
```