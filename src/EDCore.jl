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

using KeldyshED.Operators
using KeldyshED.Hilbert
using LinearAlgebra
using SparseArrays

export EDCore
export fock_states, energies, unitary_matrices
export c_connection, cdag_connection, c_matrix, cdag_matrix
export monomial_connection, monomial_matrix
export operator_blocks

"""Eigensystem within one invariant subspace of the Hamiltonian"""
struct EigenSystem{ScalarType <: Number}
  # Eigenvalues, in ascending order
  # The energy reference point is shifted so that the global energy minimum is zero
  eigenvalues::Vector{Float64}
  # Unitary transformation matrix \hat U from the Fock basis to the eigenbasis.
  # Defined so that \hat H = \hat U \mathrm{diag}(E) * \hat U^\dagger.
  unitary_matrix::Matrix{ScalarType}
end

"""Lightweight exact diagonalization solver"""
struct EDCore{ScalarType <: Number}
  # Full Hilbert space of the problem
  full_hs::FullHilbertSpace
  # Invariant subspaces
  subspaces::Vector{HilbertSubspace}
  # Eigensystems in all subspaces
  eigensystems::Vector{EigenSystem}
  # Connections between subspaces generated by C†
  # creation_connection[cdag_linear_index][sp] -> sp'
  creation_connection::Vector{Dict{Int,Int}}
  # Connections between subspaces generated by C
  # creation_connection[c_linear_index][sp] -> sp'
  annihilation_connection::Vector{Dict{Int,Int}}
  # cdag_matrices[cdag_linear_index][sp] = matrix from subspace sp to subspace sp'
  cdag_matrices::Vector{Dict{Int,Matrix{ScalarType}}}
  # c_matrices[c_linear_index][sp] = matrix from subspace sp to subspace sp'
  c_matrices::Vector{Dict{Int,Matrix{ScalarType}}}
  # Energy of the ground state (before shift)
  gs_energy::Float64
end

"""
  Reduce a given Hamiltonian to a block-diagonal form and diagonalize it

  This constructor calls the auto-partition procedure, and the QR algorithm
  to diagonalize the blocks. The invariant subspaces of the Hamiltonian are
  chosen such that all creation and annihilation operators from the provided
  fundamental operator set map one subspace to one subspace.
"""
function EDCore(hamiltonian::OperatorExpr{S}, soi::SetOfIndices) where {S <: Number}
  full_hs = FullHilbertSpace(soi)

  h = Operator{FullHilbertSpace, S}(hamiltonian, soi)
  SP = SpacePartition{FullHilbertSpace, S}(full_hs, h, false)

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

"""Subspace-to-subspace connection generated by operator C"""
function c_connection(ed::EDCore, op_linear_index::Int, sp_index::Int)
  get(ed.annihilation_connection[op_linear_index], sp_index, nothing)
end
c_connection(ed::EDCore, op_linear_index::Int, ::Nothing) = nothing

"""Subspace-to-subspace connection generated by operator C"""
function c_connection(ed::EDCore, indices::IndicesType, sp_index::Int)
  c_connection(ed, ed.full_hs.soi[indices], sp_index)
end
c_connection(ed::EDCore, indices::IndicesType, ::Nothing) = nothing

"""Matrix of subspace-to-subspace connections generated by operator C"""
function c_connection(ed::EDCore, op_linear_index::Int)
  conn = falses(length(ed.subspaces), length(ed.subspaces))
  for (j, i) in ed.annihilation_connection[op_linear_index]
    conn[i, j] = true
  end
  conn
end

"""Matrix of subspace-to-subspace connections generated by operator C"""
function c_connection(ed::EDCore, indices::IndicesType)
  c_connection(ed, ed.full_hs.soi[indices])
end

#####################
# cdag_connection() #
#####################

"""Subspace-to-subspace connection generated by operator C†"""
function cdag_connection(ed::EDCore, op_linear_index::Int, sp_index::Int)
  get(ed.creation_connection[op_linear_index], sp_index, nothing)
end
cdag_connection(ed::EDCore, op_linear_index::Int, ::Nothing) = nothing

"""Subspace-to-subspace connection generated by operator C†"""
function cdag_connection(ed::EDCore, indices::IndicesType, sp_index::Int)
  cdag_connection(ed, ed.full_hs.soi[indices], sp_index)
end
cdag_connection(ed::EDCore, indices::IndicesType, ::Nothing) = nothing

"""Matrix of subspace-to-subspace connections generated by operator C†"""
function cdag_connection(ed::EDCore, op_linear_index::Int)
  conn = falses(length(ed.subspaces), length(ed.subspaces))
  for (j, i) in ed.creation_connection[op_linear_index]
    conn[i, j] = true
  end
  conn
end

"""Matrix of subspace-to-subspace connections generated by operator C†"""
function cdag_connection(ed::EDCore, indices::IndicesType)
  cdag_connection(ed, ed.full_hs.soi[indices])
end

#######################
# monomial_connection #
#######################

"""Subspace-to-subspace connection generated by a monomial"""
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

"""Matrix of subspace-to-subspace connections generated by a monomial"""
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

"""Matrix block of fundamental operator C"""
function c_matrix(ed::EDCore, op_linear_index::Int, sp_index::Int)
  ed.c_matrices[op_linear_index][sp_index]
end

"""Matrix block of fundamental operator C"""
function c_matrix(ed::EDCore, indices::IndicesType, sp_index::Int)
  c_matrix(ed, ed.full_hs.soi[indices], sp_index)
end

#################
# cdag_matrix() #
#################

"""Matrix block of fundamental operator C†"""
function cdag_matrix(ed::EDCore, op_linear_index::Int, sp_index::Int)
  ed.cdag_matrices[op_linear_index][sp_index]
end

"""Matrix block of fundamental operator C†"""
function cdag_matrix(ed::EDCore, indices::IndicesType, sp_index::Int)
  cdag_matrix(ed, ed.full_hs.soi[indices], sp_index)
end

#####################
# monomial_matrix() #
#####################

"""Matrix block of a monomial"""
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
  Compute blocks of the matrix representation of an operator acting on states
  in a given (initial) subspace

  The computed blocks are returned as a dictionary `final subspace index =>
  matrix`.
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
  Compute blocks of the matrix representation of an operator

  The computed blocks are returned as a dictionary
  `(initial subspace index, final subspace index) => matrix`.
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

"""List of Fock states for each subspace"""
fock_states(ed::EDCore) = [sp.fock_states for sp in ed.subspaces]

"""List of energy levels for each subspace"""
energies(ed::EDCore) = [es.eigenvalues for es in ed.eigensystems]

"""List of unitary matrices for each subspace"""
unitary_matrices(ed::EDCore) = [es.unitary_matrix for es in ed.eigensystems]
