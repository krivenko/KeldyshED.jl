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
# Author: Igor Krivenko

using LinearAlgebra
using SparseArrays
using DocStringExtensions

using ..Operators: IndicesType, OperatorExpr, c, c_dag
using ..Hilbert: SetOfIndices,
                 FullHilbertSpace,
                 HilbertSubspace,
                 Operator,
                 StateVector,
                 StateDict,
                 SpacePartition,
                 merge_subspaces!,
                 numsubspaces,
                 getstateindex,
                 project

"""
Partial eigensystem within one invariant subspace of the Hamiltonian.

Fields
------
$(TYPEDFIELDS)
"""
struct EigenSystem{ScalarType <: Number}
  """
  Eigenvalues ``E_n`` of the Hamiltonian ``\\hat H``, in ascending order.
  The energy reference point is shifted so that the global energy minimum is zero.
  """
  eigenvalues::Vector{Float64}
  """
  Unitary transformation matrix ``\\hat U`` from the Fock basis to the eigenbasis,
  ``\\hat H = \\hat U \\mathrm{diag}(E_n) \\hat U^\\dagger``.
  """
  unitary_matrix::Matrix{ScalarType}
end

"""
Lightweight Exact Diagonalization solver for finite systems of fermions.

Fields
------
$(TYPEDFIELDS)
"""
struct EDCore{ScalarType <: Number}
  "Full Hilbert space of the system."
  full_hs::FullHilbertSpace
  "List of the invariant subspaces of the Hamiltonian."
  subspaces::Vector{HilbertSubspace}
  "Eigensystems of the Hamiltonian within the invariant subspaces."
  eigensystems::Vector{EigenSystem}
  """
  Subspace-to-subspace connections generated by the creation operators ``c^\\dagger_i``.
  If a creation operator, whose compound index ``i`` translates into a linear index `l` by
  `full_hs.soi`, acts between subspaces ``s`` and ``s'``, then
  `creation_connection[l][s] = s'`.
  """
  creation_connection::Vector{Dict{Int,Int}}
  """
  Subspace-to-subspace connections generated by the annihilation operators ``c_i``.
  If an annihilation operator, whose compound index ``i`` translates into a linear index `l`
  by `full_hs.soi`, acts between subspaces ``s`` and ``s'``, then
  `annihilation_connection[l][s] = s'`.
  """
  annihilation_connection::Vector{Dict{Int,Int}}
  """
  Matrices of the creation operators ``c^\\dagger_i`` in the eigenbasis of the Hamiltonian.
  If a creation operator, whose compound index ``i`` translates into a linear index `l` by
  `full_hs.soi`, acts between subspaces ``s`` and ``s'``, then the corresponding block of
  its matrix form is available as `cdag_matrices[l][s]`.
  """
  cdag_matrices::Vector{Dict{Int,Matrix{ScalarType}}}
  """
  Matrices of the annihilation operators ``c_i``  in the eigenbasis of the Hamiltonian.
  If an annihilation operator, whose compound index ``i`` translates into a linear index `l`
  by `full_hs.soi`, acts between subspaces ``s`` and ``s'``, then the corresponding block of
  its matrix form is available as `c_matrices[l][s]`.
  """
  c_matrices::Vector{Dict{Int,Matrix{ScalarType}}}
  "Ground state energy of the system."
  gs_energy::Float64
end

"""
$(TYPEDSIGNATURES)

Reduce a given Hamiltonian to a block-diagonal form and diagonalize it.

This constructor uses the [`autopartition procedure`](@ref autopartition), and the QR
algorithm to diagonalize the blocks. The invariant subspaces of the Hamiltonian are
chosen such that all creation and annihilation operators carrying compound indices
from the provided set `soi` map one subspace to one subspace.

It is possible to pass an optional list of operator expressions (`symmetry_breakers`) that
are required to share invariant subspaces with the Hamiltonian. As those operators can break
some symmetries of the Hamiltonian, taking them into account can result in a less refined
subspace partition and block structure.
"""
function EDCore(hamiltonian::OperatorExpr{S},
                soi::SetOfIndices;
                symmetry_breakers::Vector{OperatorExpr{S}} = OperatorExpr{S}[]
                ) where {S <: Number}
  full_hs = FullHilbertSpace(soi)

  h = Operator{FullHilbertSpace, S}(hamiltonian, soi)
  SP = SpacePartition{FullHilbertSpace, S}(full_hs, h, false)

  for op in symmetry_breakers
    merge_subspaces!(SP, Operator{FullHilbertSpace, S}(op, soi), false)
  end

  # Merge subspaces
  Cd_elements = Vector{SparseMatrixCSC{S, Int}}(undef, length(soi))
  C_elements = Vector{SparseMatrixCSC{S, Int}}(undef, length(soi))
  for (indices, n) in soi
    op_c_dag = Operator{FullHilbertSpace, S}(c_dag(indices...), soi)
    op_c = Operator{FullHilbertSpace, S}(c(indices...), soi)

    Cd_elements[n], C_elements[n] = merge_subspaces!(SP, op_c_dag, op_c, true)
  end

  # Fill subspaces
  subspaces = [HilbertSubspace() for sp=1:numsubspaces(SP)]
  for (n, fs) in pairs(full_hs)
    insert!(subspaces[SP[n]], fs)
  end

  # Fill connections
  creation_connection = [Dict{Int,Int}() for n=1:length(soi)]
  annihilation_connection = [Dict{Int,Int}() for n=1:length(soi)]
  for (indices, n) in soi
    rows = rowvals(Cd_elements[n])
    for f = 1:length(full_hs)
      for i in nzrange(Cd_elements[n], f)
        creation_connection[n][SP[rows[i]]] = SP[f]
      end
    end
    rows = rowvals(C_elements[n])
    for f = 1:length(full_hs)
      for i in nzrange(C_elements[n], f)
        annihilation_connection[n][SP[rows[i]]] = SP[f]
      end
    end
  end

  # Compute energy levels and eigenvectors of the Hamiltonian
  eigensystems = Vector{EigenSystem}(undef, length(subspaces))
  gs_energy::Float64 = Inf

  for (spn, subspace) in enumerate(subspaces)
    h_matrix = Matrix{S}(undef, length(subspace), length(subspace))
    i_state = StateVector{HilbertSubspace, S}(subspace)
    dim = length(subspace)
    for i=1:dim
      i_state[i] = one(S)
      f_state = h * i_state
      h_matrix[:, i] = f_state.amplitudes
      i_state[i] = zero(S)
    end

    eig = eigen(S <: Complex ? Hermitian(h_matrix) : Symmetric(h_matrix), 1:dim)
    eigensystems[spn] = EigenSystem(eig.values, eig.vectors)
    gs_energy = min(gs_energy, eigensystems[spn].eigenvalues[1])
  end

  # Shift energy reference point
  for es in eigensystems; es.eigenvalues .-= gs_energy end

  # Reorder eigensystems and subspaces along their minimal energy
  eigensystem_lt = (es1::EigenSystem, es2::EigenSystem) ->
    es1.eigenvalues[1] < es2.eigenvalues[1]
  perm = sortperm(eigensystems; lt = eigensystem_lt)
  inv_perm = invperm(perm)

  permute!(eigensystems, perm)
  permute!(subspaces, perm)

  # Update connections
  for (indices, n) in soi
    creation_connection[n] =
      Dict{Int,Int}(inv_perm[i] => inv_perm[f] for (i, f) in creation_connection[n])
    annihilation_connection[n] =
      Dict{Int,Int}(inv_perm[i] => inv_perm[f] for (i, f) in annihilation_connection[n])
  end

  # Fill cdag_matrices and c_matrices
  make_c_matrix = (connections, op_expr) -> begin
    op = Operator{FullHilbertSpace, S}(op_expr, soi)

    result = Dict{Int,Matrix{S}}()
    for (from_spn, to_spn) in connections
      from_sp = subspaces[from_spn]
      to_sp = subspaces[to_spn]

      mat = zeros(S, length(to_sp), length(from_sp))

      from_s = StateDict{FullHilbertSpace, S}(full_hs)
      for i in eachindex(from_sp)
        full_hs_i = getstateindex(full_hs, from_sp[i])
        from_s[full_hs_i] = one(S)
        to_s = op * from_s
        proj_s = project(to_s, to_sp)
        for (j, a) in pairs(proj_s); mat[j, i] = a end
        from_s[full_hs_i] = zero(S)
      end

      result[from_spn] = (eigensystems[to_spn].unitary_matrix)' * mat *
                          eigensystems[from_spn].unitary_matrix
    end
    result
  end

  cdag_matrices = [Dict{Int,Matrix{S}}() for n=1:length(soi)]
  c_matrices = [Dict{Int,Matrix{S}}() for n=1:length(soi)]
  for (indices, n) in soi
    cdag_matrices[n] = make_c_matrix(creation_connection[n], c_dag(indices...))
    c_matrices[n] = make_c_matrix(annihilation_connection[n], c(indices...))
  end

  EDCore{S}(full_hs,
            subspaces,
            eigensystems,
            creation_connection,
            annihilation_connection,
            cdag_matrices,
            c_matrices,
            gs_energy)
end

"""
$(TYPEDSIGNATURES)

Print various information about solution of an Exact Diagonalization problem.
"""
function Base.show(io::IO, ed::EDCore)
  println(io, "Dimension of full Hilbert space: ", length(ed.full_hs))
  println(io, "Number of invariant subspaces: ", length(ed.subspaces))
  for (n, sp) in enumerate(ed.subspaces)
    println(io, "  Subspace $n, dim = $(length(sp))")
    println(io, "    Basis Fock states: $(sp.fock_states)")
    println(io, "    Energy levels: $(ed.eigensystems[n].eigenvalues)")
  end
  println(io, "Ground state energy: ", ed.gs_energy)
end

##################
# c_connection() #
##################

"""
$(TYPEDSIGNATURES)

Extract a subspace-to-subspace connection generated by an annihilation operator from
an Exact Diagonalization solver. Returns `nothing` if no such connection exists.

Arguments
---------
- `ed`:              The Exact Diagonalization solver object.
- `op_linear_index`: Linear index of the annihilation operator as defined by
                     `ed.full_hs.soi`.
- `sp_index`:        Initial subspace index.
"""
function c_connection(ed::EDCore, op_linear_index::Int, sp_index::Int)
  get(ed.annihilation_connection[op_linear_index], sp_index, nothing)
end
c_connection(ed::EDCore, op_linear_index::Int, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Extract a subspace-to-subspace connection generated by an annihilation operator from
an Exact Diagonalization solver. Returns `nothing` if no such connection exists.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `indices`:  Compound index of the annihilation operator (must be part of
              `ed.full_hs.soi`).
- `sp_index`: Initial subspace index.
"""
function c_connection(ed::EDCore, indices::IndicesType, sp_index::Int)
  c_connection(ed, ed.full_hs.soi[indices], sp_index)
end
c_connection(ed::EDCore, indices::IndicesType, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Extract a matrix of subspace-to-subspace connections generated by an annihilation operator
from an Exact Diagonalization solver. This method returns a square boolean matrix of
size `length(ed.subspaces)` with `true` elements corresponding to the existing connections
between the subspaces.

Arguments
---------
- `ed`:              The Exact Diagonalization solver object.
- `op_linear_index`: Linear index of the annihilation operator as defined by
                     `ed.full_hs.soi`.
"""
function c_connection(ed::EDCore, op_linear_index::Int)
  conn = falses(length(ed.subspaces), length(ed.subspaces))
  for (j, i) in ed.annihilation_connection[op_linear_index]
    conn[i, j] = true
  end
  conn
end

"""
$(TYPEDSIGNATURES)

Extract a matrix of subspace-to-subspace connections generated by an annihilation operator
from an Exact Diagonalization solver. This method returns a square boolean matrix of
size `length(ed.subspaces)` with `true` elements corresponding to the existing connections
between the subspaces.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `indices`:  Compound index of the annihilation operator (must be part of
              `ed.full_hs.soi`).
"""
function c_connection(ed::EDCore, indices::IndicesType)
  c_connection(ed, ed.full_hs.soi[indices])
end

#####################
# cdag_connection() #
#####################

"""
$(TYPEDSIGNATURES)

Extract a subspace-to-subspace connection generated by a creation operator from
an Exact Diagonalization solver. Returns `nothing` if no such connection exists.

Arguments
---------
- `ed`:              The Exact Diagonalization solver object.
- `op_linear_index`: Linear index of the creation operator as defined by `ed.full_hs.soi`.
- `sp_index`:        Initial subspace index.
"""
function cdag_connection(ed::EDCore, op_linear_index::Int, sp_index::Int)
  get(ed.creation_connection[op_linear_index], sp_index, nothing)
end
cdag_connection(ed::EDCore, op_linear_index::Int, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Extract a subspace-to-subspace connection generated by a creation operator from
an Exact Diagonalization solver. Returns `nothing` if no such connection exists.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `indices`:  Compound index of the creation operator (must be part of `ed.full_hs.soi`).
- `sp_index`: Initial subspace index.
"""
function cdag_connection(ed::EDCore, indices::IndicesType, sp_index::Int)
  cdag_connection(ed, ed.full_hs.soi[indices], sp_index)
end
cdag_connection(ed::EDCore, indices::IndicesType, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Extract a matrix of subspace-to-subspace connections generated by a creation operator
from an Exact Diagonalization solver. This method returns a square boolean matrix of
size `length(ed.subspaces)` with `true` elements corresponding to the existing connections
between the subspaces.

Arguments
---------
- `ed`:              The Exact Diagonalization solver object.
- `op_linear_index`: Linear index of the creation operator as defined by `ed.full_hs.soi`.
"""
function cdag_connection(ed::EDCore, op_linear_index::Int)
  conn = falses(length(ed.subspaces), length(ed.subspaces))
  for (j, i) in ed.creation_connection[op_linear_index]
    conn[i, j] = true
  end
  conn
end

"""
$(TYPEDSIGNATURES)

Extract a matrix of subspace-to-subspace connections generated by a creation operator
from an Exact Diagonalization solver. This method returns a square boolean matrix of
size `length(ed.subspaces)` with `true` elements corresponding to the existing connections
between the subspaces.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `indices`:  Compound index of the creation operator (must be part of `ed.full_hs.soi`).
"""
function cdag_connection(ed::EDCore, indices::IndicesType)
  cdag_connection(ed, ed.full_hs.soi[indices])
end

#######################
# monomial_connection #
#######################

"""
$(TYPEDSIGNATURES)

Extract a subspace-to-subspace connection generated by a monomial operator (a product of
canonical operators ``c``/``c^\\dagger``) from an Exact Diagonalization solver.
Returns `nothing` if no such connection exists.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `mon`:      Monomial in question.
- `sp_index`: Initial subspace index.
"""
function monomial_connection(ed::EDCore, mon::Operators.Monomial, sp_index::Int)
  sp = sp_index
  for op in Iterators.reverse(mon.ops)
    sp = op.dagger ? cdag_connection(ed, op.indices, sp) :
                     c_connection(ed, op.indices, sp)
    sp === nothing && return nothing
  end
  sp
end
monomial_connection(ed::EDCore, mon::Operators.Monomial, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Extract a matrix of subspace-to-subspace connections generated by a monomial operator
(a product of canonical operators ``c``/``c^\\dagger``) from an Exact Diagonalization
solver. This method returns a square boolean matrix of size `length(ed.subspaces)` with
`true` elements corresponding to the existing connections between the subspaces.

Arguments
---------
- `ed`:  The Exact Diagonalization solver object.
- `mon`: Monomial in question.
"""
function monomial_connection(ed::EDCore, mon::Operators.Monomial)
  conn = falses(length(ed.subspaces), length(ed.subspaces))
  for j = 1:length(ed.subspaces)
    i = monomial_connection(ed, mon, j)
    if i !== nothing
      conn[i, j] = true
    end
  end
  conn
end

##############
# c_matrix() #
##############

"""
$(TYPEDSIGNATURES)

Extract a non-vanishing matrix block of an annihilation operator from an Exact
Diagonalization solver. The block is written in the eigenbasis of the Hamiltonian.

Arguments
---------
- `ed`:              The Exact Diagonalization solver object.
- `op_linear_index`: Linear index of the annihilation operator as defined by
                     `ed.full_hs.soi`.
- `sp_index`:        Initial subspace index.

The final subspace index is uniquely determined by the pair (`op_linear_index`, `sp_index`)
as there is at most one non-vanishing matrix per such a pair.
"""
function c_matrix(ed::EDCore, op_linear_index::Int, sp_index::Int)
  ed.c_matrices[op_linear_index][sp_index]
end

"""
$(TYPEDSIGNATURES)

Extract a non-vanishing matrix block of an annihilation operator from an Exact
Diagonalization solver. The block is written in the eigenbasis of the Hamiltonian.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `indices`:  Compound index of the annihilation operator (must be part of
              `ed.full_hs.soi`).
- `sp_index`: Initial subspace index.

The final subspace index is uniquely determined by the pair (`indices`, `sp_index`)
as there is at most one non-vanishing matrix per such a pair.
"""
function c_matrix(ed::EDCore, indices::IndicesType, sp_index::Int)
  c_matrix(ed, ed.full_hs.soi[indices], sp_index)
end

#################
# cdag_matrix() #
#################

"""
$(TYPEDSIGNATURES)

Extract a non-vanishing matrix block of a creation operator from an Exact
Diagonalization solver. The block is written in the eigenbasis of the Hamiltonian.

Arguments
---------
- `ed`:              The Exact Diagonalization solver object.
- `op_linear_index`: Linear index of the creation operator as defined by
                     `ed.full_hs.soi`.
- `sp_index`:        Initial subspace index.

The final subspace index is uniquely determined by the pair (`op_linear_index`, `sp_index`)
as there is at most one non-vanishing matrix per such a pair.
"""
function cdag_matrix(ed::EDCore, op_linear_index::Int, sp_index::Int)
  ed.cdag_matrices[op_linear_index][sp_index]
end

"""
$(TYPEDSIGNATURES)

Extract a non-vanishing matrix block of a creation operator from an Exact
Diagonalization solver. The block is written in the eigenbasis of the Hamiltonian.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `indices`:  Compound index of the creation operator (must be part of
              `ed.full_hs.soi`).
- `sp_index`: Initial subspace index.

The final subspace index is uniquely determined by the pair (`indices`, `sp_index`)
as there is at most one non-vanishing matrix per such a pair.
"""
function cdag_matrix(ed::EDCore, indices::IndicesType, sp_index::Int)
  cdag_matrix(ed, ed.full_hs.soi[indices], sp_index)
end

#####################
# monomial_matrix() #
#####################

"""
$(TYPEDSIGNATURES)

Extract a non-vanishing matrix block a monomial operator (a product of canonical operators
``c``/``c^\\dagger``) from an Exact Diagonalization solver. The block is written in the
eigenbasis of the Hamiltonian.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `mon`:      Monomial in question.
- `sp_index`: Initial subspace index.

The final subspace index is uniquely determined by the pair (`mon`, `sp_index`) as there is
at most one non-vanishing matrix per such a pair.
"""
function monomial_matrix(ed::EDCore{ScalarType},
                         mon::Operators.Monomial,
                         sp_index::Int) where {ScalarType <: Number}
  sp = sp_index
  sp_dim = length(ed.subspaces[sp])
  mat = Matrix{ScalarType}(LinearAlgebra.I, sp_dim, sp_dim)
  for op in Iterators.reverse(mon.ops)
    new_sp = op.dagger ? cdag_connection(ed, op.indices, sp) :
                         c_connection(ed, op.indices, sp)
    if new_sp === nothing
      throw(DomainError("Monomial $mon acts trivially in subspace $sp_index"))
    end
    mat = (op.dagger ? cdag_matrix(ed, op.indices, sp) :
                       c_matrix(ed, op.indices, sp)) * mat
    sp = new_sp
  end
  mat
end

#####################
# operator_blocks() #
#####################

"""
$(TYPEDSIGNATURES)

Compute blocks of the matrix representation of an operator acting on states in a given
(initial) subspace. The computed blocks are returned as a dictionary
`final subspace index => matrix`, and the matrices are written in the eigenbasis of the
Hamiltonian.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `op`:       Operator expression.
- `sp_index`: Initial subspace index.
"""
function operator_blocks(ed::EDCore{EDScalarType},
                         op::OperatorExpr{OPScalarType},
                         sp_index::Int) where {EDScalarType <: Number,
                                               OPScalarType <: Number}
  ScalarType = promote_type(EDScalarType, OPScalarType)
  d = Dict{Int64,Matrix{ScalarType}}()
  for (mon, coeff) in op
    sp = monomial_connection(ed, mon, sp_index)
    if sp !== nothing
      mat = coeff * monomial_matrix(ed, mon, sp_index)
      if sp in keys(d)
        d[sp] += mat
        isapprox(LinearAlgebra.norm(d[sp]), 0) && delete!(d, sp)
      else
        d[sp] = mat
      end
    end
  end
  d
end

"""
$(TYPEDSIGNATURES)
Compute blocks of the matrix representation of an operator.

The computed blocks are returned as a dictionary
`(initial subspace index, final subspace index) => matrix`, and the matrices are written in
the eigenbasis of the Hamiltonian.

Arguments
---------
- `ed`:       The Exact Diagonalization solver object.
- `op`:       Operator expression.
"""
function operator_blocks(ed::EDCore{EDScalarType},
                         op::OperatorExpr{OPScalarType}
                         ) where {EDScalarType <: Number,
                                  OPScalarType <: Number}
  ScalarType = promote_type(EDScalarType, OPScalarType)
  d = Dict{Tuple{Int64,Int64},Matrix{ScalarType}}()
  for (mon, coeff) in op
    for j = 1:length(ed.subspaces)
      i = monomial_connection(ed, mon, j)
      if i !== nothing
        mat = coeff * monomial_matrix(ed, mon, j)
        key = (i, j)
        if key in keys(d)
          d[key] += mat
          isapprox(LinearAlgebra.norm(d[key]), 0) && delete!(d, key)
        else
          d[key] = mat
        end
      end
    end
  end
  d
end

###################
# Other functions #
###################

"""
$(TYPEDSIGNATURES)

Collect lists of basis Fock states spanning invariant subspaces stored in a given
Exact Diagonalization object `ed`.
"""
fock_states(ed::EDCore) = [sp.fock_states for sp in ed.subspaces]

"""
$(TYPEDSIGNATURES)

Collect lists of energy levels from all invariant subspaces stored in a given
Exact Diagonalization object `ed`.
"""
energies(ed::EDCore) = [es.eigenvalues for es in ed.eigensystems]

"""
$(TYPEDSIGNATURES)

Collect unitary transformation matrices ``\\hat U`` from all invariant subspaces stored
in a given Exact Diagonalization object `ed`.
"""
unitary_matrices(ed::EDCore) = [es.unitary_matrix for es in ed.eigensystems]

"""
$(TYPEDSIGNATURES)

Transform a block-diagonal matrix written in the eigenbasis into the Fock state basis.

Arguments
---------
- `M`:  List of matrices' diagonal blocks.
- `ed`: An Exact Diagonalization object defining the invariant subspace structure and
        partial eigenbases within the subspaces.
"""
function tofockbasis(M::Vector{Matrix{T}}, ed::EDCore) where T <: Number
  [es.unitary_matrix * m * adjoint(es.unitary_matrix)
   for (m, es) in zip(M, ed.eigensystems)]
end

"""
$(TYPEDSIGNATURES)

Transform a block-diagonal matrix written in the Fock state basis into the eigenbasis.

Arguments
---------
- `M`:  List of matrices' diagonal blocks.
- `ed`: An Exact Diagonalization object defining the invariant subspace structure and
        partial eigenbases within the subspaces.
"""
function toeigenbasis(M::Vector{Matrix{T}}, ed::EDCore) where T <: Number
  [adjoint(es.unitary_matrix) * m * es.unitary_matrix
   for (m, es) in zip(M, ed.eigensystems)]
end

"""
$(TYPEDSIGNATURES)

Flatten a block-diagonal matrix and return a matrix acting in the full Hilbert space.

Arguments
---------
- `M`:  List of matrices' diagonal blocks.
- `ed`: An Exact Diagonalization object defining the invariant subspace structure.
"""
function full_hs_matrix(M::Vector{Matrix{T}}, ed::EDCore) where T <: Number
  n = length(ed.full_hs)
  M_full = zeros(T, n, n)
  for (Mss, hss) in zip(M, ed.subspaces)
      for (i1, fs1) in pairs(hss), (i2, fs2) in pairs(hss)
          M_full[getstateindex(ed.full_hs, fs1),
                 getstateindex(ed.full_hs, fs2)] = Mss[i1, i2]
      end
  end
  return M_full
end