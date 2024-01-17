# KeldyshED.jl
#
# Copyright (C) 2019-2024 Igor Krivenko <igor.s.krivenko@gmail.com>
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
# KeldyshED.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Igor Krivenko, Hugo U.R. Strand

"""
This module defines types of Hilbert spaces, state vectors and linear operators acting
in these spaces. It also implements the space autopartition algorithm described in
[Computer Physics Communications 200, March 2016, 274-284 (section 4.2)]
(https://doi.org/10.1016/j.cpc.2015.10.023).
"""
module Hilbert

export SetOfIndices, reversemap, matching_indices
export FockState, translate
export HilbertSpace, FullHilbertSpace, HilbertSubspace, getstateindex
export StateVector, StateDict, State, dot, project
export Operator
export SpacePartition, numsubspaces, merge_subspaces!
export ⊗, product_basis_map, factorized_basis_map

using LinearAlgebra
using DataStructures
using SparseArrays
using DocStringExtensions

using ..Operators: IndicesType, OperatorExpr

################
# SetOfIndices #
################

"""
Mapping from a compound operator index [`IndicesType`](@ref) to a linear integer index.
It provides a dictionary-like iteration interface, supports indexed read-only access,
functions `keys()`, `values()` and `pairs()`, as well as the operator `in`.

Fields
------
$(TYPEDFIELDS)
"""
mutable struct SetOfIndices <: AbstractDict{IndicesType, Int}
  "The [`IndicesType`](@ref) -> linear index map"
  map_index_n::SortedDict{IndicesType, Int}
end

SetOfIndices() = SetOfIndices(SortedDict{IndicesType, Int}())

"""
$(TYPEDSIGNATURES)

Construct a set from a vector of compound operator indices `v`.
"""
function SetOfIndices(v::AbstractVector)
  map_index_n = SortedDict{IndicesType, Int}(IndicesType(i) => 1 for i in v)
  # Reorder the linear indices
  map_index_n = SortedDict{IndicesType, Int}(
    k => n for (n, (k, v)) in enumerate(map_index_n)
  )
  return SetOfIndices(map_index_n)
end

"""
$(TYPEDSIGNATURES)

Insert a new compound operator index `indices` into `soi`.
"""
function Base.insert!(soi::SetOfIndices, indices::IndicesType)
  insert!(soi.map_index_n, indices, length(soi.map_index_n) + 1)
  # Reorder the linear indices
  soi.map_index_n = SortedDict{IndicesType, Int}(
    k => n for (n, (k, v)) in enumerate(soi.map_index_n)
  )
end

"""
$(TYPEDSIGNATURES)

Insert a new compound operator index built out of arguments `indices...` into `soi`.
"""
function Base.insert!(soi::SetOfIndices, indices...)
  insert!(soi, IndicesType([indices...]))
end

function Base.:(==)(soi1::SetOfIndices, soi2::SetOfIndices)
  return soi1.map_index_n == soi2.map_index_n
end

Base.getindex(soi::SetOfIndices, indices) = soi.map_index_n[indices]
function Base.getindex(soi::SetOfIndices, indices...)
  return soi.map_index_n[IndicesType([indices...])]
end

Base.in(indices, soi::SetOfIndices) = indices in keys(soi.map_index_n)

"""
$(TYPEDSIGNATURES)

Build and return the reverse map `Int` -> [`IndicesType`](@ref) out of
a [`SetOfIndices`](@ref) object.
"""
reversemap(soi::SetOfIndices)::Vector{IndicesType} = collect(keys(soi.map_index_n))

"""
$(TYPEDSIGNATURES)

For each compound index in the set `soi_from`, find the linear index of that compound index
within `soi_to` and collect the found linear indices.
"""
function matching_indices(soi_from::SetOfIndices, soi_to::SetOfIndices)::Vector{Int}
  @assert keys(soi_from) ⊆ keys(soi_to)
  return getindex.(Ref(soi_to), keys(soi_from))
end

#####################################
# SetOfIndices: Iteration interface #
#####################################
Base.eltype(soi::SetOfIndices) = Pair{IndicesType, Int}

Base.length(soi::SetOfIndices) = length(soi.map_index_n)
Base.isempty(soi::SetOfIndices) = isempty(soi.map_index_n)

Base.iterate(soi::SetOfIndices) = iterate(soi.map_index_n)
Base.iterate(soi::SetOfIndices, it) = iterate(soi.map_index_n, it)

Base.keys(soi::SetOfIndices) = keys(soi.map_index_n)
Base.values(soi::SetOfIndices) = values(soi.map_index_n)
Base.pairs(soi::SetOfIndices) = pairs(soi.map_index_n)

################
# HilbertSpace #
################

"""
$(TYPEDEF)

Fermionic Fock state encoded as a sequence of zeros (unoccupied single-particle states)
and ones (occupied states) in the binary representation of an integer.
"""
const FockState = UInt64

"Abstract supertype for Hilbert space types."
abstract type HilbertSpace <: AbstractVector{FockState} end

"""
$(TYPEDSIGNATURES)

Reshuffle single-particle states (bits) of a Fock state `fs` according to a given map
`bit_map`. `bit_map` is understood as a reverse map if `reverse = true`.
"""
function translate(fs::FockState,
                   bit_map::Vector{Int64};
                   reverse=false)::FockState
  fs_out = FockState(0)
  for (bit_from, bit_to) in pairs(bit_map)
    if reverse
      bit_from, bit_to = bit_to, bit_from
    end
    if fs & (0b1 << (bit_from - 1)) > 0
      fs_out += (0b1 << (bit_to - 1))
    end
  end
  return fs_out
end

####################
# FullHilbertSpace #
####################

"""
A Hilbert space spanned by all fermionic Fock states generated by a given set of
creation/annihilation operators. This type supports vector-like iteration over Fock states,
read-only indexed access and the operator `in`.

Fields
------
$(TYPEDFIELDS)
"""
struct FullHilbertSpace <: HilbertSpace
  "Set of compound operator indices used to generate this space"
  soi::SetOfIndices
  "Dimension of the space"
  dim::UInt64
end

"""
$(TYPEDSIGNATURES)

Make a Hilbert space generated by creation/annihilation operators carrying compound indices
from a given set `soi`.
"""
FullHilbertSpace(soi::SetOfIndices) = FullHilbertSpace(soi, 1 << length(soi))
FullHilbertSpace() = FullHilbertSpace(SetOfIndices([]))

Base.:(==)(fhs1::FullHilbertSpace, fhs2::FullHilbertSpace) = fhs1.dim == fhs2.dim

Base.in(fs::FockState, fhs::FullHilbertSpace) = fs < fhs.dim

function Base.getindex(fhs::FullHilbertSpace, index)
  if index <= fhs.dim
    return FockState(index - 1)
  else
    throw(BoundsError(x, "Fock state does not exist (index too big)"))
  end
end

"""
$(TYPEDSIGNATURES)

Return the Fock state ``|\\psi\\rangle \\propto c^\\dagger_i c^\\dagger_j c^\\dagger_k
\\ldots |0\\rangle``, where the compound indices ``i,j,k,\\ldots`` form a given set
`soi_indices` and the creation operators act in a Hilbert space `fhs`.
"""
function Base.getindex(fhs::FullHilbertSpace, soi_indices::Set{IndicesType})
  return foldl((fs, ind) -> fs + (FockState(1) << (fhs.soi[ind] - 1)),
               soi_indices;
               init = FockState(0))
end

"""
$(TYPEDSIGNATURES)

Find the index of a given Fock state `fs` within a full Hilbert space `fhs`.
"""
function getstateindex(fhs::FullHilbertSpace, fs::FockState)
  if fs < fhs.dim
    return Int(fs + 1)
  else
    throw(BoundsError(x, "Fock state is not part of this Hilbert space"))
  end
end

#########################################
# FullHilbertSpace: Iteration interface #
#########################################
Base.eltype(fhs::FullHilbertSpace) = FockState

Base.length(fhs::FullHilbertSpace)::Int = fhs.dim
Base.isempty(fhs::FullHilbertSpace) = fhs.dim == 0
Base.size(fhs::FullHilbertSpace) = (length(fhs),)

function Base.iterate(fhs::FullHilbertSpace)
  return fhs.dim > 0 ? (FockState(0), 0) : nothing
end
function Base.iterate(fhs::FullHilbertSpace, it)
  return it < fhs.dim - 1 ? (FockState(it + 1), it + 1) : nothing
end

Base.keys(fhs::FullHilbertSpace) = LinearIndices(1:fhs.dim)
Base.values(fhs::FullHilbertSpace) = [FockState(i) for i=0:fhs.dim-1]
function Base.pairs(fhs::FullHilbertSpace)
  return collect(Iterators.Pairs([FockState(i) for i=0:fhs.dim-1],
                                 LinearIndices(1:fhs.dim)))
end

###############################################
# FullHilbertSpace: Direct products of spaces #
###############################################

"""
$(TYPEDSIGNATURES)

Construct a direct product of full Hilbert spaces ``H_A \\otimes H_B`` under the
assumption that the sets of indices generating ``H_A`` and ``H_B`` are disjoint.
"""
function (⊗)(H_A::FullHilbertSpace, H_B::FullHilbertSpace)
  @assert isdisjoint(keys(H_A.soi), keys(H_B.soi))
  soi_AB = SetOfIndices(union(keys(H_A.soi), keys(H_B.soi)))
  return FullHilbertSpace(soi_AB)
end

"""
$(TYPEDSIGNATURES)

Construct a quotient space ``H_{AB} / H_A`` under the assumption that the sets of indices
generating ``H_{AB}`` and ``H_A`` satisfy ``soi(H_A) \\subseteq soi(H_{AB})``.
"""
function Base.:(/)(H_AB::FullHilbertSpace, H_A::FullHilbertSpace)
  @assert keys(H_A.soi) ⊆ keys(H_AB.soi)
  soi_B = SetOfIndices(setdiff(keys(H_AB.soi), keys(H_A.soi)))
  return FullHilbertSpace(soi_B)
end

"""
$(TYPEDSIGNATURES)

Given three Hilbert spaces ``H_A``, ``H_B`` and ``H_{AB}``, construct all product
Fock states ``|\\psi\\rangle_{AB} = |\\psi\\rangle_A ⊗ |\\psi\\rangle_B``, where
``|\\psi\\rangle_A ∈ H_A, |\\psi\\rangle_B ∈ H_B, |\\psi\\rangle_{AB} ∈ H_{AB}``.
Return a mapping (an integer-valued matrix) ``i, j \\mapsto k``, where ``i``, ``j`` and
``k`` are linear indices of ``|\\psi\\rangle_A``, ``|\\psi\\rangle_B`` and
``|\\psi\\rangle_{AB}`` within their respective spaces.

Sets of indices ``soi(H_A)``, ``soi(H_B)`` must be disjoint and satisfy
``soi(H_A) \\subseteq soi(H_{AB})``, ``soi(H_B) \\subseteq soi(H_{AB})``.
"""
function product_basis_map(H_A::FullHilbertSpace,
                           H_B::FullHilbertSpace,
                           H_AB::FullHilbertSpace)::Array{Int64, 2}
  @assert isdisjoint(keys(H_A.soi), keys(H_B.soi))

  all_fs_A_in_H_AB = translate.(H_A, Ref(matching_indices(H_A.soi, H_AB.soi)))
  all_fs_B_in_H_AB = translate.(H_B, Ref(matching_indices(H_B.soi, H_AB.soi)))

  return [getstateindex(H_AB, fs_A + fs_B) for fs_A in all_fs_A_in_H_AB,
                                               fs_B in all_fs_B_in_H_AB]
end

"""
$(TYPEDSIGNATURES)

Given a Hilbert space ``H_{AB}`` and its divisor ``H_A``, factorize all Fock states
``|\\psi\\rangle_{AB} \\in H_{AB}`` into a direct product
``|\\psi\\rangle_A \\otimes |\\psi\\rangle_B``, where ``|\\psi\\rangle_A \\in H_A``,
``|\\psi\\rangle_B \\in H_{AB} / H_A``.
Return a mapping (a vector of integer pairs) ``k \\mapsto (i, j)``, where ``i``, ``j``
and ``k`` are linear indices of ``|\\psi\\rangle_A``, ``|\\psi\\rangle_B`` and
``|\\psi\\rangle_{AB}`` within their respective spaces.

Sets of indices ``soi(H_{AB})``, ``soi(H_A)`` must satisfy
``soi(H_A) \\subseteq soi(H_{AB})``.
"""
function factorized_basis_map(H_AB::FullHilbertSpace,
                              H_A::FullHilbertSpace)
  H_B = H_AB / H_A

  fs_A_comp_of_fs_AB = translate.(H_AB, Ref(matching_indices(H_A.soi, H_AB.soi)),
                                  reverse=true)
  fs_B_comp_of_fs_AB = translate.(H_AB, Ref(matching_indices(H_B.soi, H_AB.soi)),
                                  reverse=true)

  return [(getstateindex(H_A, fs_A), getstateindex(H_B, fs_B)) for
          (fs_A, fs_B) in zip(fs_A_comp_of_fs_AB, fs_B_comp_of_fs_AB)]
end

###################
# HilbertSubspace #
###################

"""
Subspace of a Hilbert space, as a list of basis Fock states.

This type supports vector-like iteration over Fock states, read-only indexed access and
the operator `in`.

Fields
------
$(TYPEDFIELDS)
"""
struct HilbertSubspace <: HilbertSpace
  "List of all Fock states spanning the space"
  fock_states::Vector{FockState}
  "Reverse map to quickly find the index of a basis Fock state"
  fock_to_index::Dict{FockState,Int}
end

HilbertSubspace() = HilbertSubspace(Vector{FockState}(), Dict{FockState,Int}())

"""
$(TYPEDSIGNATURES)

Insert a basis Fock state `fs` into a subspace `hss`.
"""
function Base.insert!(hss::HilbertSubspace, fs::FockState)
  push!(hss.fock_states, fs)
  hss.fock_to_index[fs] = length(hss.fock_states)
end

function Base.:(==)(hss1::HilbertSubspace, hss2::HilbertSubspace)
  return hss1.fock_states == hss2.fock_states
end

Base.in(fs::FockState, hss::HilbertSubspace) = fs in hss.fock_states

Base.getindex(hss::HilbertSubspace, index) = hss.fock_states[index]

"""
$(TYPEDSIGNATURES)

Find the index of a given Fock state `fs` within a subspace `hss`.
"""
getstateindex(hss::HilbertSubspace, fs::FockState) = hss.fock_to_index[fs]

########################################
# HilbertSubspace: Iteration interface #
########################################
Base.eltype(hss::HilbertSubspace) = FockState

Base.length(hss::HilbertSubspace) = length(hss.fock_states)
Base.isempty(hss::HilbertSubspace) = isempty(hss.fock_states)
Base.size(hss::HilbertSubspace) = (length(hss),)

Base.iterate(hss::HilbertSubspace) = iterate(hss.fock_states)
Base.iterate(hss::HilbertSubspace, it) = iterate(hss.fock_states, it)

Base.keys(hss::HilbertSubspace) = keys(hss.fock_states)
Base.values(hss::HilbertSubspace) = values(hss.fock_states)
Base.pairs(hss::HilbertSubspace) = pairs(hss.fock_states)

#########
# State #
#########

"""Abstract supertype of quantum state types"""
abstract type State{HSType <: HilbertSpace, ScalarType <: Number} end

###############
# StateVector #
###############

"""
Quantum state in a Hilbert space/subspace implemented as a vector of amplitudes.

The amplitudes can be accessed using the indexing interface and iterated over.
States support addition/subtraction and multiplication/division by a constant scalar.

Fields
------
$(TYPEDFIELDS)
"""
struct StateVector{HSType <: HilbertSpace, ScalarType} <: State{HSType, ScalarType}
  "Hilbert space this state belongs to"
  hs::HSType
  "Amplitudes of basis states contributing to this state"
  amplitudes::Vector{ScalarType}
end

"""
```julia
StateVector{HSType, S}(hs::HSType) where {HSType, S}
```

Create a vector-based state in a Hilbert space/subspace `hs` with
all amplitudes set to zero.
"""
function StateVector{HSType, S}(hs::HSType) where {HSType, S}
  return StateVector{HSType, S}(hs, zeros(S, length(hs)))
end

"""
$(TYPEDSIGNATURES)

Create a vector-based state similar to `sv` with all amplitudes set to zero.
"""
function Base.similar(sv::StateVector{HSType, S}) where {HSType, S}
  return StateVector{HSType, S}(sv.hs)
end

function Base.getindex(sv::StateVector{HSType, S}, index) where {HSType, S}
  return getindex(sv.amplitudes, index)
end

function Base.setindex!(sv::StateVector{HSType, S},
                        val::S,
                        index) where {HSType, S}
  return setindex!(sv.amplitudes, val, index)
end

function Base.:+(sv1::StateVector{HSType, S},
                 sv2::StateVector{HSType, S}) where {HSType, S}
  return StateVector{HSType, S}(sv1.hs, sv1.amplitudes .+ sv2.amplitudes)
end

function Base.:-(sv1::StateVector{HSType, S},
                 sv2::StateVector{HSType, S}) where {HSType, S}
  return StateVector{HSType, S}(sv1.hs, sv1.amplitudes .- sv2.amplitudes)
end

function Base.:*(sv::StateVector{HSType, S}, x::S) where {HSType, S}
  return StateVector{HSType, S}(sv.hs, sv.amplitudes * x)
end
Base.:*(x::S, sv::StateVector{HSType, S}) where {HSType, S} = sv * x

Base.:/(sv::StateVector{HSType, S}, x::S) where {HSType, S} = sv * (one(S) / x)

"""
$(TYPEDSIGNATURES)

Compute the scalar product of two vector-based states `sv1` and `sv2`.
"""
function dot(sv1::StateVector, sv2::StateVector)
  return LinearAlgebra.dot(sv1.amplitudes, sv2.amplitudes)
end

function Base.firstindex(sv::StateVector{HSType, S}) where {HSType, S}
  return firstindex(sv.amplitudes)
end
function Base.lastindex(sv::StateVector{HSType, S}) where {HSType, S}
  return lastindex(sv.amplitudes)
end

"""
$(TYPEDSIGNATURES)

Print a vector-based state.
"""
function Base.show(io::IO, sv::StateVector{HSType, S}) where {HSType, S}
  something_written = false
  for (i, a) in pairs(sv.amplitudes)
    if !isapprox(a, 0, atol = 100*eps(S))
      print(io, " +($a)|" * repr(Int(sv.hs[i])) * ">")
      something_written = true
    end
  end
  if !something_written print(io, "0") end
end

Base.eltype(sv::StateVector{HSType, S}) where {HSType, S} = Pair{Int, S}

"""
$(TYPEDSIGNATURES)

Project a vector-based state `sv` from one Hilbert space onto another
Hilbert space/subspace `target_space`.
"""
function project(sv::StateVector{HSType, S},
                 target_space::TargetHSType) where {HSType, S, TargetHSType}
  proj_sv = StateVector{TargetHSType, S}(target_space)
  for (i, a) in pairs(sv.amplitudes)
    f = sv.hs[i]
    if f in target_space
      proj_sv[getstateindex(target_space, f)] = a
    end
  end
  return proj_sv
end

#############
# StateDict #
#############

"""
Quantum state in a Hilbert space/subspace implemented as a (sparse) dictionary
of amplitudes.

The amplitudes can be accessed using the indexing interface and iterated over.
States support addition/subtraction and multiplication/division by a constant scalar.

Fields
------
$(TYPEDFIELDS)
"""
struct StateDict{HSType <: HilbertSpace, ScalarType} <: State{HSType, ScalarType}
  "Hilbert space this state belongs to"
  hs::HSType
  """
  Non-vanishing amplitudes of basis states contributing to this state. Each element of this
  dictionary is a pair (index of the basis state within `hs`, amplitude).
  """
  amplitudes::Dict{Int, ScalarType}
end

"""
```julia
StateDict{HSType, S}(hs::HSType) where {HSType, S}
```

Create a dictionary-based state in a Hilbert space/subspace `hs` with
zero non-vanishing amplitudes.
"""
function StateDict{HSType, S}(hs::HSType) where {HSType, S}
  return StateDict{HSType, S}(hs, Dict{Int, S}())
end

"""
$(TYPEDSIGNATURES)

Create a dictionary-based state similar to `sd` with zero non-vanishing amplitudes.
"""
function Base.similar(sd::StateDict{HSType, S}) where {HSType, S}
  return StateDict{HSType, S}(sd.hs)
end

function Base.getindex(sd::StateDict{HSType, S}, index) where {HSType, S}
  return get(sd.amplitudes, index, zero(S))
end

function Base.setindex!(sd::StateDict{HSType, S},
                        val::S,
                        index) where {HSType, S}
  if isapprox(val, 0, atol = 1e-10)
    (index in keys(sd.amplitudes)) && delete!(sd.amplitudes, index)
    return zero(S)
  else
    return (sd.amplitudes[index] = val)
  end
end

function Base.:+(sd1::StateDict{HSType, S},
                 sd2::StateDict{HSType, S}) where {HSType, S}
  d = merge(+, sd1.amplitudes, sd2.amplitudes)
  filter!(p -> !isapprox(p.second, 0, atol = 1e-10), d)
  return StateDict{HSType, S}(sd1.hs, d)
end

function Base.:-(sd1::StateDict{HSType, S},
                 sd2::StateDict{HSType, S}) where {HSType, S}
  d = merge(+, sd1.amplitudes, Dict([(i=>-a) for (i,a) in sd2.amplitudes]))
  filter!(p -> !isapprox(p.second, 0, atol = 1e-10), d)
  return StateDict{HSType, S}(sd1.hs, d)
end

function Base.:*(sd::StateDict{HSType, S}, x::Number) where {HSType, S}
  if isapprox(x, 0, atol = 1e-10)
    return StateDict{HSType, S}(sd.hs)
  else
    return StateDict{HSType, S}(sd.hs, Dict([i => a*x for (i, a) in pairs(sd.amplitudes)]))
  end
end
Base.:*(x::Number, sd::StateDict{HSType, S}) where {HSType, S} = sd * x

Base.:/(sd::StateDict{HSType, S}, x::Number) where {HSType, S} = sd * (one(S)/x)

"""
$(TYPEDSIGNATURES)

Compute the scalar product of two dictionary-based states `sd1` and `sd2`.
"""
function dot(sd1::StateDict{HSType, S}, sd2::StateDict{HSType, S}) where {HSType, S}
  res = zero(S)
  for (i, a) in sd1.amplitudes
    res += conj(a) * get(sd2.amplitudes, i, 0)
  end
  return res
end

"""
$(TYPEDSIGNATURES)

Print a dictionary-based state.
"""
function Base.show(io::IO, sd::StateDict{HSType, S}) where {HSType, S}
  something_written = false
  for i in sort(collect(keys(sd.amplitudes)))
    a = sd.amplitudes[i]
    if !isapprox(a, 0, atol = 100*eps(S))
      print(io, " +($a)|" * repr(Int(sd.hs[i])) * ">")
      something_written = true
    end
  end
  if !something_written print(io, "0") end
end

Base.eltype(sd::StateDict) = eltype(sd.amplitudes)

"""
$(TYPEDSIGNATURES)

Project a dictionary-based state `sd` from one Hilbert space onto another
Hilbert space/subspace `target_space`.
"""
function project(sd::StateDict{HSType, S},
                 target_space::TargetHSType) where {HSType, S, TargetHSType}
  proj_sd = StateVector{TargetHSType, S}(target_space)
  for (i, a) in pairs(sd.amplitudes)
    f = sd.hs[i]
    if f in target_space
      proj_sd[getstateindex(target_space, f)] = a
    end
  end
  return proj_sd
end

##############################
# State: Iteration interface #
##############################

Base.length(st::State) = length(st.amplitudes)
Base.isempty(st::State) = isempty(st.amplitudes)

Base.size(st::State) = (length(st),)
Base.size(st::State, dim) = dim == 1 ? length(st) : 1

Base.iterate(st::State) = iterate(st.amplitudes)
Base.iterate(st::State, it) = iterate(st.amplitudes, it)

Base.keys(st::State) = keys(st.amplitudes)
Base.values(st::State) = values(st.amplitudes)
Base.pairs(st::State) = pairs(st.amplitudes)

############
# Operator #
############

# Fock state convention:
# |0,...,k> = C^+_0 ... C^+_k |0>
# Operator monomial convention:
# C^+_0 ... C^+_i ... C_j  ... C_0

struct OperatorTerm{ScalarType <: Number}
  coeff::ScalarType
  # Bit masks used to change bits
  annihilation_mask::FockState
  creation_mask::FockState
  # Bit masks for particle counting
  annihilation_count_mask::FockState
  creation_count_mask::FockState
end

"""
Quantum-mechanical operator acting on states in a Hilbert space.
"""
struct Operator{HSType <: HilbertSpace, ScalarType <: Number}
  terms::Vector{OperatorTerm{ScalarType}}
end

"""
```julia
Operator{HSType, S}(op_expr::OperatorExpr{S}, soi::SetOfIndices) where {HSType, S}
```

Make a linear operator acting on a Hilbert space/subspace out of a [`polynomial expression`]
(@ref Main.KeldyshED.Operators.OperatorExpr) `op_expr`. The set `soi` is used to establish a
correspondence between compound indices of creation/annihilation operators met in
`op_expr` and the single-particle states -- bits of [`FockState`](@ref).
"""
function Operator{HSType, S}(op_expr::OperatorExpr{S},
                             soi::SetOfIndices) where {HSType, S}
  compute_count_mask = (d::Vector{Int}) -> begin
    mask::FockState = 0
    is_on = (length(d) % 2) == 1
    for i = 1:64
      if i in d
        is_on = !is_on
      else
        if is_on
          mask |= (one(FockState) << (i-1))
        end
      end
    end
    return mask
  end

  creation_ind = Int[]   # Linear indices of creation operators in a monomial
  annihilation_ind = Int[]  # Linear indices of annihilation operators in a monomial
  terms = OperatorTerm{S}[]
  for (monomial, coeff) in op_expr
    empty!(creation_ind)
    empty!(annihilation_ind)
    annihilation_mask::FockState = 0
    creation_mask::FockState = 0
    for c_op in monomial.ops
      if c_op.dagger
        push!(creation_ind, soi[c_op.indices])
        creation_mask |= (one(FockState) << (soi[c_op.indices]-1))
      else
        push!(annihilation_ind, soi[c_op.indices])
        annihilation_mask |= (one(FockState) << (soi[c_op.indices]-1))
      end
    end
    push!(terms, OperatorTerm(coeff,
                              annihilation_mask,
                              creation_mask,
                              compute_count_mask(annihilation_ind),
                              compute_count_mask(creation_ind)))
  end
  return Operator{HSType, S}(terms)
end

_parity_number_of_bits(v::FockState) = isodd(count_ones(v))

"""
$(TYPEDSIGNATURES)

Act with an operator `op` on a state `st` and return the resulting state.
"""
function Base.:*(op::Operator, st::StateType) where {StateType <: State}
  target_st = similar(st)
  for term in op.terms
    for (i, a) in pairs(st)
      f2 = st.hs[i]
      (f2 & term.annihilation_mask) != term.annihilation_mask && continue
      f2 &= ~term.annihilation_mask
      ((f2 ⊻ term.creation_mask) & term.creation_mask) !=
        term.creation_mask && continue
      f3 = ~(~f2 & ~term.creation_mask)
      sign = _parity_number_of_bits((f2 & term.annihilation_count_mask) ⊻
                                    (f3 & term.creation_count_mask)) == 0 ? 1 : -1
      ind = getstateindex(target_st.hs, f3)
      target_st[ind] += a * term.coeff * sign
    end
  end
  return target_st
end

##################
# SpacePartition #
##################

"""
Partition of a Hilbert space into a set of disjoint subspaces invariant under
action of a given Hermitian operator (Hamiltonian).

A detailed description of the algorithm can be found in Section 4.2 of
[Computer Physics Communications 200, March 2016, 274-284]
(https://doi.org/10.1016/j.cpc.2015.10.023).

[`SpacePartition`](@ref) supports iteration over pairs (index of a basis state of the
Hilbert space, index of the invariant subspace that state belongs to).

Fields
------
$(TYPEDFIELDS)
"""
struct SpacePartition{HSType <: HilbertSpace, ScalarType <: Number}
  "Full Hilbert space subject to partitioning"
  hs::HSType
  "Disjoint set of subspaces"
  subspaces::IntDisjointSets
  "Map root index to subspace index"
  root_to_index::Dict{Int, Int}
  "Matrix elements of the Hamiltonian"
  matrix_elements::SparseMatrixCSC{ScalarType, Int}
end

"""
```julia
SpacePartition{HSType, S}(hs::HSType,
                          H::OperatorType,
                          store_matrix_elements::Bool = true) where {
    HSType <: HilbertSpace, S <: Number, OperatorType <: Operator}
```

Create a [`SpacePartition`](@ref) structure by performing Phase I of the automatic
partitioning algorithm, i.e. discovering the invariant subspaces of the Hermitian operator
`H` acting on the Hilbert space `hs`.

If `store_matrix_elements = true`, the field `matrix_elements` of the created structure
contains a sparse-matrix representation of `H`.
"""
function SpacePartition{HSType, S}(hs::HSType,
                        H::OperatorType,
                        store_matrix_elements::Bool = true
                        ) where {HSType <: HilbertSpace,
                                 S <: Number,
                                 OperatorType <: Operator}
  subspaces = IntDisjointSets(length(hs))
  matrix_elements = spzeros(S, length(hs), length(hs))
  _merge_subspaces!(S, hs, H, subspaces, matrix_elements, store_matrix_elements)

  root_to_index = Dict{Int,Int}()
  _rebuild_root_to_index!(root_to_index, subspaces)

  return SpacePartition{HSType, S}(hs, subspaces, root_to_index, matrix_elements)
end

"""
$(TYPEDSIGNATURES)

Return the number of subspaces in a space partition `sp`.
"""
numsubspaces(sp::SpacePartition) = length(sp.root_to_index)

"""
$(TYPEDSIGNATURES)

Find the invariant subspace within a partition `sp`, the state with a given `index`
belongs to.
"""
function Base.getindex(sp::SpacePartition, index)
  return sp.root_to_index[find_root!(sp.subspaces, index)]
end

function _merge_subspaces!(S::Type,
                           hs::HSType,
                           op::OperatorType,
                           subspaces::IntDisjointSets,
                           matrix_elements,
                           store_matrix_elements::Bool) where {
                            HSType <: HilbertSpace,
                            OperatorType <: Operator}
  init_state = StateDict{HSType, S}(hs)
  for i=1:length(hs)
    init_state[i] = one(S)

    final_state = op * init_state

    for (f, a) in pairs(final_state)
      isapprox(a, 0, atol = 1e-10) && continue
      i_subspace = find_root!(subspaces, i)
      f_subspace = find_root!(subspaces, f)
      i_subspace != f_subspace && root_union!(subspaces, i_subspace, f_subspace)
      if store_matrix_elements
        matrix_elements[i, f] = a
      end
    end

    init_state[i] = zero(S)
  end
end

"""
$(SIGNATURES)

Merge some of the invariant subspaces in `sp` to ensure that the resulting subspaces are
also invariant w.r.t. a given operator `op`.

If `store_matrix_elements = true`, a sparse-matrix representation of `op` is returned.
"""
function merge_subspaces!(sp::SpacePartition{HSType, S},
                          op::OperatorType,
                          store_matrix_elements::Bool = true) where {
                           HSType <: HilbertSpace,
                           S <: Number,
                           OperatorType <: Operator}
  matrix_elements = spzeros(S, length(sp), length(sp))
  _merge_subspaces!(S, sp.hs, op, sp.subspaces, matrix_elements, store_matrix_elements)

  # Rebuild sp.root_to_index
  _rebuild_root_to_index!(sp.root_to_index, sp.subspaces)

  return matrix_elements
end

"""
$(SIGNATURES)

Perform Phase II of the automatic partition algorithm.

Merge some of the invariant subspaces in `sp` to ensure that a given operator `Cd`
and its Hermitian conjugate `C` generate only one-to-one connections between
the subspaces.

If `store_matrix_elements = true`, a tuple of two sparse matrices representing `Cd` and `C`
is returned.
"""
function merge_subspaces!(sp::SpacePartition{HSType, S},
                          Cd::OperatorType,
                          C::OperatorType,
                          store_matrix_elements::Bool = true) where {
                           HSType <: HilbertSpace,
                           S <: Number,
                           OperatorType <: Operator}
  Cd_elements = spzeros(S, length(sp), length(sp))
  C_elements = spzeros(S, length(sp), length(sp))

  Cd_connections = SortedMultiDict{Int,Int}()
  C_connections = SortedMultiDict{Int,Int}()

  # Fill connection multidicts
  init_state = StateDict{HSType, S}(sp.hs)
  for i=1:length(sp)
    i_subspace = find_root!(sp.subspaces, i)
    init_state[i] = one(S)

    fill_conn = (op, conn, matrix_elements) -> begin
      final_state = op * init_state
      for (f, a) in pairs(final_state)
        isapprox(a, 0, atol = 1e-10) && continue
        insert!(conn, i_subspace, find_root!(sp.subspaces, f))
        if store_matrix_elements
          matrix_elements[i, f] = a
        end
      end
    end

    fill_conn(Cd, Cd_connections, Cd_elements)
    fill_conn(C, C_connections, C_elements)

    init_state[i] = zero(S)
  end

  # 'Zigzag' traversal algorithm
  while !isempty(Cd_connections)
    # Take one C^† - connection
    # C^†|lower_subspace> = |upper_subspace>
    lower_subspace, upper_subspace = first(Cd_connections)

    # The following lambda-function
    #
    # - reveals all subspaces reachable from lower_subspace by application of
    #   a 'zigzag' product C† C C† C C† ... of any length;
    # - removes all visited connections from Cd_connections/C_connections;
    # - merges lower_subspace with all subspaces generated from lower_subspace
    #   by application of (C C†)^(2*n);
    # - merges upper_subspace with all subspaces generated from upper_subspace
    #   by application of (C† C)^(2*n).
    zigzag_traversal = (i_subspace, upwards) -> begin
      conn = upwards ? Cd_connections : C_connections
      while (tok = searchequalrange(conn, i_subspace)[1]) != pastendsemitoken(conn)
        f_subspace = deref_value((conn, tok))
        delete!((conn, tok))

        if (upwards)
          union!(sp.subspaces, f_subspace, upper_subspace)
        else
          union!(sp.subspaces, f_subspace, lower_subspace)
        end

        # Recursively apply to all found f_subspace's with the 'flipped' direction
        zigzag_traversal(f_subspace, !upwards)
      end
    end

    # Apply to all C† connections starting from lower_subspace
    zigzag_traversal(lower_subspace, true)
  end

  # Rebuild sp.root_to_index
  _rebuild_root_to_index!(sp.root_to_index, sp.subspaces)

  return (Cd_elements, C_elements)
end

# Rebuild a root_to_index dictionary from a space partition
function _rebuild_root_to_index!(root_to_index::Dict{Int, Int}, subspaces::IntDisjointSets)
  empty!(root_to_index)
  for i=1:length(subspaces)
    root = find_root!(subspaces, i)
    if !(root in keys(root_to_index))
      root_to_index[root] = length(root_to_index) + 1
    end
  end
end

#######################################
# SpacePartition: Iteration interface #
#######################################
Base.eltype(sd::SpacePartition) = Pair{Int, Int}

Base.length(sp::SpacePartition) = length(sp.hs)
Base.isempty(sp::SpacePartition) = isempty(sp.hs)

Base.size(sp::SpacePartition) = (length(sp),)
Base.size(sp::SpacePartition, dim) = dim == 1 ? length(sp) : 1

Base.iterate(sp::SpacePartition) = (1 => sp[1], 1)
function Base.iterate(sp::SpacePartition, it)
  if it < length(sp)
    return (it+1 => sp[it+1], it+1)
  else
    return nothing
  end
end

Base.keys(sp::SpacePartition) = LinearIndices(1:length(sp))
Base.values(sp::SpacePartition) = [sp[i] for i=1:length(sp)]
Base.pairs(sp::SpacePartition) = collect(Iterators.Pairs(values(sp), keys(sp)))

end # module Hilbert
