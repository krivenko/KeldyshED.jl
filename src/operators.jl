# KeldyshED.jl
#
# Copyright (C) 2019-2023 Igor Krivenko <igor.s.krivenko@gmail.com>
# Copyright (C) 2015 P. Seth, I. Krivenko, M. Ferrero and O. Parcollet
#
# KeldyshED.jl is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# KeldyshED.jl is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# KeldyshED.jl. If not, see <http://www.gnu.org/licenses/.
#
# Authors: Igor Krivenko, Hugo U.R. Strand

"""
This module implements an algebra of fermionic creation/annihilation operators
and polynomial expressions built out of such operators.
"""
module Operators

export IndicesType
export OperatorExpr, RealOperatorExpr, ComplexOperatorExpr, scalartype
export c, c_dag, n, dagger

using DataStructures
using DocStringExtensions

#################################
# Auxiliary types and functions #
#################################

"""
$(TYPEDEF)

Compound operator index ``\\alpha`` as a mixed list of `String` / `Int` indices.
"""
const IndicesType = Vector{Union{String, Int}}

function cmp(ind1::IndicesType, ind2::IndicesType)
  for (i1, i2) in zip(ind1, ind2)
    if !isequal(i1, i2)
      if typeof(i1) == typeof(i2)
        return (isless(i1, i2) ? -1 : 1)
      else
        # Assume that Int values < String values
        return (isa(i1, Int) ? -1 : 1)
      end
    end
  end
  return cmp(length(ind1), length(ind2))
end

"""
This structure represents a single creation operator ``c^\\dagger_\\alpha`` or
an annihilation operator ``c_\\alpha``.

Fields
------
$(TYPEDFIELDS)

Canonical operators are less-comparable and are ordered according to
the following rule,

``
    c^\\dagger_\\alpha < c^\\dagger_\\beta < c^\\dagger_\\gamma <
    c_\\gamma < c_\\beta < c_\\alpha,
``

where compound indices satisfy ``\\alpha < \\beta < \\gamma``.
"""
struct CanonicalOperator
  "Creation operator if `true`, annihilation operator otherwise"
  dagger::Bool
  "Compound index α carried by the operator"
  indices::IndicesType
end

function Base.:(==)(op1::CanonicalOperator, op2::CanonicalOperator)
  op1.dagger == op2.dagger && op1.indices == op2.indices
end

function Base.isless(op1::CanonicalOperator, op2::CanonicalOperator)
  # Order: dagger < non dagger, and then indices
  # c+_1 < c+_2 < c+_3 < c_3 < c_2 < c_1
  if op1.dagger != op2.dagger return op1.dagger end
  if op1.dagger
    return cmp(op1.indices, op2.indices) < 0
  else
    return cmp(op2.indices, op1.indices) < 0
  end
end

"""
$(TYPEDSIGNATURES)

Hermitian conjugate of a canonical operator.
"""
dagger(op::CanonicalOperator) = CanonicalOperator(!op.dagger, op.indices)

"""
$(TYPEDSIGNATURES)

Check if `op2` is a Hermitian conjugate of `op1`.
"""
function isconjugate(op1::CanonicalOperator, op2::CanonicalOperator)
  op1.dagger != op2.dagger && op1.indices == op2.indices
end

"""
$(TYPEDSIGNATURES)

Print a canonical operator.
"""
function Base.show(io::IO, op::CanonicalOperator)
  print(io, "c", (op.dagger ? "†" : ""), "(")
  print(io, join(map(repr, op.indices), ","), ")")
end

############################################
# Monomial: Product of canonical operators #
############################################

"""
An ordered product of canonical operators.

Fields
------
$(TYPEDFIELDS)

Monomials are less-comparable, with a shorter monomial being always considered
lesser than a longer one.
"""
struct Monomial
  "List of canonical operators in the product, left-to-right."
  ops::Vector{CanonicalOperator}
end

Monomial() = Monomial(CanonicalOperator[])

Base.:(==)(m1::Monomial, m2::Monomial) = m1.ops == m2.ops

function Base.isless(m1::Monomial, m2::Monomial)
  if length(m1.ops) != length(m2.ops)
    return length(m1.ops) < length(m2.ops)
  else
    return Base.cmp(m1.ops, m2.ops) < 0
  end
end

"""
$(TYPEDSIGNATURES)

Number of canonical operators in a given monomial.
"""
Base.length(m::Monomial) = length(m.ops)
"""
$(TYPEDSIGNATURES)

Check if a given monomial is 1, i.e. a product of zero canonical operators.
"""
Base.isempty(m::Monomial) = imempty(m.ops)

"""
$(TYPEDSIGNATURES)

Hermitian conjugate of a monomial.
"""
function dagger(m::Monomial)
  Monomial([dagger(op) for op in Iterators.reverse(m.ops)])
end

"""
$(TYPEDSIGNATURES)

Print a monomial.
"""
function Base.show(io::IO, m::Monomial)
  if ! isempty(m.ops)
    print(io, "*", join(map(repr, m.ops)))
  end
end

const const_monomial = Monomial()

################
# OperatorExpr #
################

"""
Polynomial expression built out of canonical operators and numeric coefficients,

``
M^{(0)} + \\sum_\\alpha M^{(1)}_\\alpha o_\\alpha +
    \\sum_{\\alpha\\beta} M^{(2)}_{\\alpha\\beta} o_\\alpha o_\\beta +
    \\sum_{\\alpha\\beta\\gamma} M^{(3)}_{\\alpha\\beta\\gamma}
        o_\\alpha o_\\beta o_\\gamma + \\ldots,
``

where ``o_\\alpha`` are canonical operators ``c^\\dagger_\\alpha`` / ``c_\\alpha``,
and ``M^{(n)}`` are the coefficients.

Fields
------
$(TYPEDFIELDS)

Polynomial expressions support the following arithmetic operations.

- Addition / subtraction of two expressions with the same coefficient type.
- Multiplication of two expressions with the same coefficient type.
- Addition / subtraction of a constant.
- Multiplication / division by a constant.
- Unary minus.

They also support the iteration interface with iteration element type being
`Pair{Monomial, ScalarType}`.
"""
struct OperatorExpr{ScalarType <: Number}
  "Sorted list of monomials with numeric coefficients ``M^{(n)}`` in front of them."
  monomials::SortedDict{Monomial, ScalarType}
end

"""
```julia
OperatorExpr{S}() where S
```

Construct a polynomial expression with coefficient type `S` identically equal to 0.
"""
OperatorExpr{S}() where {S <: Number} = OperatorExpr{S}(SortedDict{Monomial, S}())
"""
```julia
OperatorExpr{S}(x::S) where {S <: Number}
```

Construct a polynomial expression with coefficient type `S` identically equal to `x`.
"""
function OperatorExpr{S}(x::S) where {S <: Number}
  OperatorExpr{S}(SortedDict{Monomial, S}(Monomial() => x))
end

"""
$(TYPEDEF)

Polynomial expression with real coefficients.
"""
const RealOperatorExpr = OperatorExpr{Float64}
"""
$(TYPEDEF)

Polynomial expression with complex coefficients.
"""
const ComplexOperatorExpr = OperatorExpr{ComplexF64}

##################################
# OperatorExpr: basic operations #
##################################

"""
$(TYPEDSIGNATURES)

Determine the type of monomial coefficients of a given polynomial expression.
"""
scalartype(::OperatorExpr{S}) where S = S
"""
$(TYPEDSIGNATURES)

Determine the type of monomial coefficients of a given polynomial expression type.
"""
scalartype(::Type{OperatorExpr{S}}) where S = S

"""
$(TYPEDSIGNATURES)

Check if a given polynomial expression is identically zero.
"""
Base.iszero(op::OperatorExpr{S}) where S = isempty(op.monomials)

function Base.:(==)(op1::OperatorExpr{S}, op2::OperatorExpr{S}) where S
  iszero(op1 - op2)
end

#################################################################
# OperatorExpr: Algebraic operations involving scalar constants #
#################################################################

# Unary minus
function Base.:-(op::OperatorExpr{S}) where S
  OperatorExpr{S}(SortedDict{Monomial, S}(m => -c for (m,c) in op.monomials))
end

# Add constant
function Base.:+(op::OperatorExpr{S}, alpha::Number) where S
  isapprox(alpha, 0, atol = 100*eps(real(S))) && return op
  res = deepcopy(op)
  tok = findkey(res.monomials, const_monomial)
  if tok == pastendsemitoken(res.monomials)
    insert!(res.monomials, const_monomial, alpha)
  else
    res.monomials[tok] += alpha
    if isapprox(res.monomials[tok], 0, atol = 100*eps(real(S)))
      delete!((res.monomials, tok))
    end
  end
  res
end
Base.:+(alpha::Number, op::OperatorExpr) = op + alpha

# Subtract constant
Base.:-(op::OperatorExpr, alpha::Number) = op + (-alpha)
Base.:-(alpha::Number, op::OperatorExpr) = -op + alpha

# Multiply by constant
function Base.:*(op::OperatorExpr{S}, alpha::Number) where S
  if isapprox(alpha, 0, atol = 100*eps(real(S)))
    OperatorExpr{S}()
  else
    OperatorExpr{S}(SortedDict{Monomial, S}(m => alpha * c for (m,c) in op.monomials))
  end
end
Base.:*(alpha::Number, op::OperatorExpr{S}) where S = op * alpha

# Divide by constant
Base.:/(op::OperatorExpr, alpha::Number) = op * (one(alpha) / alpha)

######################################
# OperatorExpr: Algebraic operations #
######################################

# Addition
function Base.:+(op1::OperatorExpr{S}, op2::OperatorExpr{S}) where S
  res = deepcopy(op1)
  for (m, c) in op2.monomials
    res_tok = findkey(res.monomials, m)
    if res_tok == pastendsemitoken(res.monomials)
      insert!(res.monomials, m, c)
    else
      res.monomials[res_tok] += c
      if isapprox(res.monomials[res_tok], 0, atol = 100*eps(real(S)))
        delete!((res.monomials, res_tok))
      end
    end
  end
  res
end

# Subtraction
function Base.:-(op1::OperatorExpr{S}, op2::OperatorExpr{S}) where S
  res = deepcopy(op1)
  for (m, c) in op2.monomials
    res_tok = findkey(res.monomials, m)
    if res_tok == pastendsemitoken(res.monomials)
      insert!(res.monomials, m, -c)
    else
      res.monomials[res_tok] -= c
      if isapprox(res.monomials[res_tok], 0, atol = 100*eps(real(S)))
        delete!((res.monomials, res_tok))
      end
    end
  end
  res
end

# Multiplication
function Base.:*(op1::OperatorExpr{S}, op2::OperatorExpr{S}) where S
  product = SortedDict{Monomial, S}()  # product will be stored here
  for (m1, c1) in op1.monomials
    for (m2, c2) in op2.monomials
      # prepare an unnormalized product
      product_m = Monomial(vcat(m1.ops, m2.ops))
      normalize_and_insert(product_m, c1 * c2, product)
    end
  end
  OperatorExpr{S}(product)
end

# Normalize a monomial and insert into a dictionary
function normalize_and_insert(m::Monomial,
                              coeff::S,
                              target::SortedDict{Monomial,S}) where S
  # The normalization is done by employing a simple bubble sort algorithm.
  # Apart from sorting elements this function keeps track of the sign and
  # recursively calls itself if a permutation of two operators produces a new
  # monomial
  if length(m) >= 2
    while true
      is_swapped = false
      for n = 2:length(m)
        prev_index = m.ops[n - 1];
        cur_index  = m.ops[n];
        if prev_index == cur_index return end # The monomial is effectively zero
        if prev_index > cur_index
          # Are we swapping C and C^+ with the same indices?
          if isconjugate(cur_index, prev_index)
            new_m = Monomial(vcat(m.ops[1:n-2], m.ops[n+1:end]))
            normalize_and_insert(new_m, coeff, target);
          end
          coeff = -coeff
          m.ops[n - 1], m.ops[n] = m.ops[n], m.ops[n - 1]
          is_swapped = true
        end
      end
      if !is_swapped break end
    end
  end

  # Insert the result into target
  tok = findkey(target, m)
  if tok == pastendsemitoken(target)
    insert!(target, m, coeff)
  else
    target[tok] += coeff
    if isapprox(target[tok], 0, atol = 100*eps(real(S)))
      delete!((target, tok))
    end
  end
end

"""
$(TYPEDSIGNATURES)

Hermitian conjugate of a polynomial expression.
"""
function dagger(op::OperatorExpr{S}) where S
  monomials = SortedDict{Monomial, S}()
  for (m, c) in op.monomials
    push!(monomials, dagger(m) => conj(c))
  end
  OperatorExpr{S}(monomials)
end

#####################################
# OperatorExpr: Iteration interface #
#####################################
Base.eltype(op::OperatorExpr{S}) where S = Pair{Monomial, S}

"""
$(TYPEDSIGNATURES)

Number of monomials in a given polynomial expression `op`.
"""
Base.length(op::OperatorExpr) = length(op.monomials)
"""
$(TYPEDSIGNATURES)

Check if a given polynomial expression is identically zero.
"""
Base.isempty(op::OperatorExpr) = length(op) == 0

Base.iterate(op::OperatorExpr) = iterate(op.monomials)
Base.iterate(op::OperatorExpr, it) = iterate(op.monomials, it)

#############################################
# OperatorExpr: map() and related functions #
#############################################

"""
$(TYPEDSIGNATURES)

Transform a given polynomial expression `op` by applying a function `f`
to coefficients of all monomials in the expression. If `f` applied
to a coefficient returns zero, the corresponding monomial is omitted
from the resulting expression.
"""
function Base.map(f, op::OperatorExpr{S}) where S
  res = OperatorExpr{S}()
  for (m, c) in op
    new_c = f(m, c)
    if !isapprox(new_c, 0, atol = 100*eps(real(S)))
      push!(res.monomials, m => new_c)
    end
  end
  res
end

"""
$(TYPEDSIGNATURES)

Return a version of a polynomial expression `op` with all coefficients replaced by
their real parts.
"""
Base.real(op::OperatorExpr) = map((m, c) -> real(c), op)
"""
$(TYPEDSIGNATURES)

Return a version of a polynomial expression `op` with all coefficients replaced by
their imaginary parts.
"""
Base.imag(op::OperatorExpr) = map((m, c) -> imag(c), op)

#######################################
# OperatorExpr: String representation #
#######################################

"""
$(TYPEDSIGNATURES)

Print a polynomial expression.
"""
function Base.show(io::IO, op::OperatorExpr{S}) where S
  if isempty(op.monomials)
    print(io, zero(S))
  else
    # Add parentheses around complex numbers
    f(x) = (S <: Complex) ? "($x)" : "$x"
    print(io, join(["$(f(c))$m" for (m,c) in op.monomials], " + "))
  end
end

###################################
# OperatorExpr: Factory functions #
###################################

"""
$(TYPEDSIGNATURES)

Make an annihilation operator ``c_\\alpha`` with `indices...` being
components of the compound index ``\\alpha``. Coefficient type
of the resulting polynomial expression can be specified via
`scalar_type` (defaults to `Float64`).
"""
function c(indices...; scalar_type = Float64)
  m = Monomial([CanonicalOperator(false, collect(indices))])
  OperatorExpr{scalar_type}(
    SortedDict{Monomial,scalar_type}(m => one(scalar_type)))
end

"""
$(TYPEDSIGNATURES)

Make an creation operator ``c^\\dagger_\\alpha`` with `indices...` being
components of the compound index ``\\alpha``. Coefficient type
of the resulting polynomial expression can be specified via
`scalar_type` (defaults to `Float64`).
"""
function c_dag(indices...; scalar_type = Float64)
  m = Monomial([CanonicalOperator(true, collect(indices))])
  OperatorExpr{scalar_type}(
    SortedDict{Monomial,scalar_type}(m => one(scalar_type)))
end

"""
$(TYPEDSIGNATURES)

Make a particle number operator ``n_\\alpha = c^\\dagger_\\alpha c_\\alpha``
with `indices...` being components of the compound index ``\\alpha``.
Coefficient type of the resulting polynomial expression can be specified via
`scalar_type` (defaults to `Float64`).
"""
function n(indices...; scalar_type = Float64)
  m = Monomial([CanonicalOperator(true, collect(indices)),
                CanonicalOperator(false, collect(indices))])
  OperatorExpr{scalar_type}(
    SortedDict{Monomial,scalar_type}(m => one(scalar_type)))
end

end # module Operators
