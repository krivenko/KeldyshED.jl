# Expressions with creation/annihilation operators of fermions

```@docs
KeldyshED.Operators
```
```@meta
CurrentModule = KeldyshED.Operators
```

## Canonical operators

```@docs
IndicesType
CanonicalOperator
dagger(op::CanonicalOperator)
isconjugate(op1::CanonicalOperator, op2::CanonicalOperator)
Base.show(io::IO, op::CanonicalOperator)
```

## Monomials

```@docs
Monomial
Base.length(m::Monomial)
Base.isempty(m::Monomial)
dagger(m::Monomial)
Base.show(io::IO, m::Monomial)
```

## Polynomial expressions

```@docs
OperatorExpr
OperatorExpr{S}(x::S) where {S <: Number}
RealOperatorExpr
ComplexOperatorExpr
scalartype
dagger(op::OperatorExpr)
Base.iszero(op::OperatorExpr)
Base.length(op::OperatorExpr)
Base.isempty(op::OperatorExpr)
Base.map(f, op::OperatorExpr)
Base.real(op::OperatorExpr)
Base.imag(op::OperatorExpr)
Base.show(io::IO, op::OperatorExpr)
```

## Factory functions for polynomial expressions

```@docs
c
c_dag
n
```