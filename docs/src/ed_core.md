# Exact Diagonalization solver

```@meta
CurrentModule = KeldyshED
```

## Core ED types and functions

```@docs
EDCore
EigenSystem
EDCore(::OperatorExpr, ::SetOfIndices; ::Vector{OperatorExpr})
Base.show(io::IO, ed::EDCore)
c_connection
cdag_connection
monomial_connection
c_matrix
cdag_matrix
monomial_matrix
operator_blocks
```

## Utility functions

```@docs
fock_states
energies
unitary_matrices
tofockbasis(M::Vector{Matrix{T}}, ed::EDCore) where T <: Number
toeigenbasis(M::Vector{Matrix{T}}, ed::EDCore) where T <: Number
full_hs_matrix
```