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
# Authors: Hugo U.R. Strand, Igor Krivenko

using KeldyshED: EDCore, Hilbert
using Keldysh

export tofockbasis,
       toeigenbasis,
       partial_trace,
       partition_function,
       density_matrix,
       reduced_density_matrix,
       evolution_operator,
       reduced_evolution_operator

"""Compute the partition function at an inverse temperature β"""
function partition_function(ed::EDCore, β::Real)
  return sum(sum(exp.(-β * es.eigenvalues)) for es in ed.eigensystems)
end

"""
  Compute the equilibrium density matrix at an inverse temperature β.

  The density matrix is returned as a vector of diagonal blocks with each
  block represented in the eigenbasis of the system.
"""
function density_matrix(ed::EDCore, β::Real)
  z = partition_function(ed, β)
  return [diagm(exp.(-β * es.eigenvalues) / z) for es in ed.eigensystems]
end

const EvolutionOperator = Vector{GenericTimeGF{ComplexF64, false}}

raw"""
  Compute the evolution operator

    S(t, t') = exp(-i \int_{t'}^{t} H d\bar t)

  on a given time grid.

  The operator is returned as a vector of diagonal blocks (matrix-valued
  Green's function containers) with each block represented in the eigenbasis
  of the system.
"""
function evolution_operator(ed::EDCore,
                            grid::AbstractTimeGrid)::EvolutionOperator
  [GenericTimeGF(grid, length(es.eigenvalues)) do t1, t2
      diagm(exp.(-im * es.eigenvalues * (t1.bpoint.val - t2.bpoint.val)))
   end
   for es in ed.eigensystems]
end

#########################
# Basis transformations #
#########################

"""
  Transform a block-diagonal evolution operator written in the eigenbasis
  of the system into the Fock state basis.
"""
function tofockbasis(S::Vector{GF}, ed::EDCore) where {GF <: AbstractTimeGF}
  S_fock = GF[]
  for (s, es) in zip(S, ed.eigensystems)
    U = es.unitary_matrix
    push!(S_fock, similar(s))
    for t1 in S_fock[end].grid, t2 in S_fock[end].grid
      S_fock[end][t1, t2] = U * s[t1, t2] * adjoint(U)
    end
  end
  return S_fock
end

"""
  Transform a block-diagonal evolution operator written in the Fock state
  basis into the eigenbasis of the system.
"""
function toeigenbasis(S::Vector{GF}, ed::EDCore) where {GF <: AbstractTimeGF}
  S_eigen = GF[]
  for (s, es) in zip(S, ed.eigensystems)
    U = es.unitary_matrix
    push!(S_eigen, similar(s))
    for t1 in S_eigen[end].grid, t2 in S_eigen[end].grid
      S_eigen[end][t1, t2] = adjoint(U) * s[t1, t2] * U
    end
  end
  return S_eigen
end

#################
# Partial trace #
#################

"""
  Compute a partial trace of a block-diagonal matrix.

  The matrix is expected to be written in the Fock state basis.
  The resulting reduced matrix acts in a Hilbert space spanned by all
  fermionic Fock states generated by the set of indices `target_soi`.
"""
function partial_trace(M::Vector{Matrix{T}},
                       ed::EDCore,
                       target_soi::SetOfIndices) where T

  target_hs = FullHilbertSpace(target_soi)
  fmap = factorized_basis_map(ed.full_hs, target_hs)

  n = length(target_hs)
  M_target = zeros(T, n, n)

  for (Mss, hss) in zip(M, ed.subspaces)
    for (i1, fs1) in pairs(hss), (i2, fs2) in pairs(hss)
      i_target1, i_env1 = fmap[getstateindex(ed.full_hs, fs1)]
      i_target2, i_env2 = fmap[getstateindex(ed.full_hs, fs2)]

      i_env1 != i_env2 && continue
      M_target[i_target1, i_target2] += Mss[i1, i2]
    end
  end

  M_target
end

#
# Construct a similar GF object with a different number of orbitals
#

function Base.similar(G::T, norb::Int) where T <: GenericTimeGF
  GenericTimeGF(eltype(T), G.grid, norb, is_scalar(T))
end
function Base.similar(G::T, norb::Int) where T <: FullTimeGF
  FullTimeGF(eltype(T), G.grid, norb, G.ξ, is_scalar(T))
end
function Base.similar(G::T, norb::Int) where T <: TimeInvariantFullTimeGF
  TimeInvariantFullTimeGF(eltype(T), G.grid, norb, G.ξ, is_scalar(T))
end
function Base.similar(G::T, norb::Int) where T <: KeldyshTimeGF
  KeldyshTimeGF(eltype(T), G.grid, norb, G.ξ, is_scalar(T))
end
function Base.similar(G::T, norb::Int) where T <: TimeInvariantKeldyshTimeGF
  TimeInvariantKeldyshTimeGF(eltype(T), G.grid, norb, G.ξ, is_scalar(T))
end
function Base.similar(G::T, norb::Int) where T <: ImaginaryTimeGF
  ImaginaryTimeGF(eltype(T), G.grid, norb, G.ξ, is_scalar(T))
end

"""
  Compute a partial trace of a block-diagonal evolution operator written in
  the Fock state basis.

  The resulting reduced evolution operator acts in a Hilbert space spanned
  by all fermionic Fock states generated by the set of indices `target_soi`.
"""
function partial_trace(S::Vector{GF},
                       ed::EDCore,
                       target_soi::SetOfIndices) where {GF <: AbstractTimeGF}
  target_hs = FullHilbertSpace(target_soi)
  fmap = factorized_basis_map(ed.full_hs, target_hs)

  S_target = similar(S[1], length(target_hs))

  for (Sss, hss) in zip(S, ed.subspaces)
    for (i1, fs1) in pairs(hss), (i2, fs2) in pairs(hss)
      i_target1, i_env1 = fmap[getstateindex(ed.full_hs, fs1)]
      i_target2, i_env2 = fmap[getstateindex(ed.full_hs, fs2)]

      i_env1 != i_env2 && continue

      for t1 in S_target.grid, t2 in S_target.grid
        S_target[i_target1, i_target2, t1, t2] += Sss[i1, i2, t1, t2]
      end
    end
  end

  S_target
end

"""
  Compute a reduced equilibrium density matrix at an inverse temperature β.

  The resulting reduced matrix acts in a Hilbert space spanned by all
  fermionic Fock states generated by the set of indices `target_soi`.
"""
function reduced_density_matrix(ed::EDCore,
                                target_soi::SetOfIndices,
                                β::Real)
  ρ = tofockbasis(density_matrix(ed, β), ed)
  return partial_trace(ρ, ed, target_soi)
end

raw"""
  Compute a reduced evolution operator

    S_{target}(t, t') = Tr_{env}[exp(-i \int_{t'}^{t} H d\bar t)]

  on a given time grid.

  The Hamiltonian H acts in a direct product of Hilbert spaces
  H_{target} ⊗ H_{env}, while the resulting reduced propagator acts
  in H_{target} spanned by all fermionic Fock states generated by
  the set of indices `target_soi`.
"""
function reduced_evolution_operator(ed::EDCore,
                                    target_soi::SetOfIndices,
                                    grid::AbstractTimeGrid)
  S = tofockbasis(evolution_operator(ed, grid), ed)
  return partial_trace(S, ed, target_soi)
end
