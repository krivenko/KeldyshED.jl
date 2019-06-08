module Operators

using DataStructures

export IndicesType
export OperatorExpr, RealOperatorExpr, ComplexOperatorExpr, scalartype
export c, c_dag, n, dagger

#################################
# Auxiliary types and functions #
#################################

"""Combination of String/Int indices"""
const IndicesType = Vector{Union{String,Int}}

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

"""Canonical operator C/C^+"""
struct CanonicalOperator
  dagger::Bool          # Creation/annihilation
  indices::IndicesType  # Indices of canonical operator
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

"""Hermitian conjugate"""
dagger(op::CanonicalOperator) = CanonicalOperator(!op.dagger, op.indices)

function isconjugate(op1::CanonicalOperator, op2::CanonicalOperator)
  op1.dagger != op2.dagger && op1.indices == op2.indices
end

function Base.show(io::IO, op::CanonicalOperator)
  print(io, "c", (op.dagger ? "†" : ""), "(")
  print(io, join(map(repr, op.indices), ","), ")")
end

############################################
# Monomial: Product of canonical operators #
############################################

"""An ordered product of canonical operators"""
struct Monomial
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

Base.length(m::Monomial) = length(m.ops)
Base.isempty(m::Monomial) = imempty(m.ops)

"""Hermitian conjugate"""
function dagger(m::Monomial)
  Monomial([dagger(op) for op in Iterators.reverse(m.ops)])
end

function Base.show(io::IO, m::Monomial)
  if ! isempty(m.ops)
    print(io, "*", join(map(repr, m.ops)))
  end
end

const const_monomial = Monomial()

################
# OperatorExpr #
################

"""Polynomial built out of canonical operators"""
struct OperatorExpr{ScalarType <: Number}
  # Monomial -> coefficient in front of it
  monomials::SortedDict{Monomial, ScalarType}
end

# Construct empty expression
OperatorExpr{S}() where S = OperatorExpr{S}(SortedDict{Monomial, S}())
# Construct constant expression
function OperatorExpr{S}(x::S) where {S <: Number}
  OperatorExpr{S}(SortedDict{Monomial, S}(Monomial() => x))
end

const RealOperatorExpr = OperatorExpr{Float64}
const ComplexOperatorExpr = OperatorExpr{ComplexF64}

##################################
# OperatorExpr: basic operations #
##################################

"""Determine the type of operator monomial coefficients"""
scalartype(::OperatorExpr{S}) where S = S
scalartype(::Type{OperatorExpr{S}}) where S = S

# Is zero operator?
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
Base.:+(alpha::Number, op::OperatorExpr{S}) where S = op + alpha

# Subtract constant
Base.:-(op::OperatorExpr{S}, alpha::Number) where S = op + (-alpha)
Base.:-(alpha::Number, op::OperatorExpr{S}) where S = -op + alpha

# Multiply by constant
function Base.:*(op::OperatorExpr{S}, alpha::Number) where S
  if isapprox(alpha, 0, atol = 100*eps(real(S)))
    OperatorExpr{S}()
  else
    OperatorExpr{S}(SortedDict{Monomial, S}(m => alpha for (m,c) in op.monomials))
  end
end
Base.:*(alpha::Number, op::OperatorExpr{S}) where S = op * alpha

# Divide by constant
Base.:/(op::OperatorExpr{S}, alpha::Number) where S = op * (one(alpha) / alpha)

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

"""Hermitian conjugate"""
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

Base.length(op::OperatorExpr{S}) where S = length(op.monomials)
Base.isempty(op::OperatorExpr{S}) where S = length(op) == 0

Base.iterate(op::OperatorExpr{S}) where S = iterate(op.monomials)
Base.iterate(op::OperatorExpr{S}, it) where S = iterate(op.monomials, it)

#############################################
# OperatorExpr: map() and related functions #
#############################################
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

Base.real(op::OperatorExpr{S}) where S = map((m, c) -> real(c), op)
Base.imag(op::OperatorExpr{S}) where S = map((m, c) -> imag(c), op)

#######################################
# OperatorExpr: String representation #
#######################################

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

"""Make an annihilation operator"""
function c(indices...; scalar_type = Float64)
  m = Monomial([CanonicalOperator(false, collect(indices))])
  OperatorExpr{scalar_type}(
    SortedDict{Monomial,scalar_type}(m => one(scalar_type)))
end

"""Make a creation operator"""
function c_dag(indices...; scalar_type = Float64)
  m = Monomial([CanonicalOperator(true, collect(indices))])
  OperatorExpr{scalar_type}(
    SortedDict{Monomial,scalar_type}(m => one(scalar_type)))
end

"""Make a particle number operator"""
function n(indices...; scalar_type = Float64)
  m = Monomial([CanonicalOperator(true, collect(indices)),
                CanonicalOperator(false, collect(indices))])
  OperatorExpr{scalar_type}(
    SortedDict{Monomial,scalar_type}(m => one(scalar_type)))
end

end # module Operators
