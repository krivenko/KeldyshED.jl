# Hilbert spaces, states and linear operators

```@docs
KeldyshED.Hilbert
```
```@meta
CurrentModule = KeldyshED.Hilbert
```

## Ordered set of compound indices

```@docs
Hilbert.SetOfIndices
Base.insert!(soi::SetOfIndices, indices::IndicesType)
Base.insert!(soi::SetOfIndices, indices...)
reversemap(soi::SetOfIndices)
matching_indices
```

## Full Hilbert spaces and their subspaces

```@docs
FockState
translate
HilbertSpace
FullHilbertSpace
FullHilbertSpace(soi::SetOfIndices)
getindex(fhs::FullHilbertSpace, soi_indices::Set{IndicesType})
(⊗)(H_A::FullHilbertSpace, H_B::FullHilbertSpace)
(/)(H_AB::FullHilbertSpace, H_A::FullHilbertSpace)
product_basis_map
factorized_basis_map
HilbertSubspace
Base.insert!(hss::HilbertSubspace, fs::FockState)
getstateindex
```

## Quantum states

```@docs
State
StateVector
StateVector(hs)
StateDict
StateDict(hs)
Base.similar
dot
project
Base.show(io::IO, sv::StateVector)
Base.show(io::IO, sd::StateDict)
```

## Operators acting in Hilbert spaces

```@docs
Operator
Operator{HSType, S}(op_expr::OperatorExpr{S}, soi::SetOfIndices) where {
    HSType, S}
(*)(op::Operator, st::StateType) where {StateType <: State}
```

## [Space autopartition algorithm] (@id autopartition)

```@docs
SpacePartition
SpacePartition{HSType, S}(::HSType, ::OperatorType, ::Bool) where {
    HSType <: HilbertSpace, S <: Number, OperatorType <: Operator}
merge_subspaces!
numsubspaces
Base.getindex(sp::SpacePartition, index)
```