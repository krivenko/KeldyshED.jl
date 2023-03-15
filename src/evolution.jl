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

export partition_function,
       density_matrix,
       evolution_operator,
       tofockbasis,
       toeigenbasis

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

"""
Transform a block-diagonal matrix written in the eigenbasis of the system
into the Fock state basis.
"""
function tofockbasis(A::Vector{Matrix{T}}, ed::EDCore) where T
  [es.unitary_matrix * a * adjoint(es.unitary_matrix)
   for (a, es) in zip(A, ed.eigensystems)]
end

"""
Transform a block-diagonal evolution operator written in the eigenbasis
of the system into the Fock state basis.
"""
function tofockbasis(S::EvolutionOperator, ed::EDCore)
  S_fock = EvolutionOperator()
  for (s, es) in zip(S, ed.eigensystems)
    U = es.unitary_matrix
    push!(S_fock,
          GenericTimeGF(s.grid, length(es.eigenvalues)) do t1, t2
            U * s[t1, t2] * adjoint(U)
          end
    )
  end
  S_fock
end

"""
Transform a block-diagonal matrix written in the Fock state basis into
the eigenbasis of the system.
"""
function toeigenbasis(A::Vector{Matrix{T}}, ed::EDCore) where T
  [adjoint(es.unitary_matrix) * a * es.unitary_matrix
   for (a, es) in zip(A, ed.eigensystems)]
end

"""
Transform a block-diagonal evolution operator written in the Fock state basis
into the eigenbasis of the system.
"""
function toeigenbasis(S::EvolutionOperator, ed::EDCore)
  S_eigen = EvolutionOperator()
  for (s, es) in zip(S, ed.eigensystems)
    U = es.unitary_matrix
    push!(S_eigen,
          GenericTimeGF(s.grid, length(es.eigenvalues)) do t1, t2
            adjoint(U) * s[t1, t2] * U
          end
    )
  end
  S_eigen
end