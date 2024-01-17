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
# Author: Igor Krivenko

"""
Equilibrium Exact Diagonalization library that can also compute Green's functions
on the Keldysh contour.
"""
module KeldyshED

# EDCore
export EDCore
export fock_states, energies, unitary_matrices
export c_connection, cdag_connection, c_matrix, cdag_matrix
export monomial_connection, monomial_matrix
export operator_blocks
export tofockbasis, toeigenbasis, full_hs_matrix

# Green's functions
export computegf, SerialGFFiller, DistributedGFFiller

# Evolution operators
export tofockbasis, toeigenbasis
export partial_trace
export partition_function, density_matrix, reduced_density_matrix
export evolution_operator, reduced_evolution_operator

include("operators.jl")
include("hilbert.jl")
include("ed_core.jl")
include("gf.jl")
include("evolution.jl")

end # module KeldyshED
