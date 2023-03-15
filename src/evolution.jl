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

export partition_function, density_matrix, evolution_operator

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
  return [Diagonal(exp.(-β * es.eigenvalues) / z) for es in ed.eigensystems]
end

raw"""
Compute the evolution operator

    S(t, t') = exp(-i \int_{t'}^{t} H d\bar t)

on a given time grid.

The operator is returned as a vector of diagonal blocks (matrix-valued
Green's function containers) with each block represented in the eigenbasis
of the system.
"""
function evolution_operator(ed::EDCore,
                            grid::AbstractTimeGrid)::
                            Vector{GenericTimeGF{ComplexF64, false}}
  [GenericTimeGF(grid, length(es.eigenvalues)) do t1, t2
      Diagonal(exp.(-im * es.eigenvalues * (t1.bpoint.val - t2.bpoint.val)))
   end
   for es in ed.eigensystems]
end
