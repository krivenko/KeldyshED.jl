# KeldyshED.jl
#
# Copyright (C) 2019 Igor Krivenko <igor.s.krivenko@gmail.com>
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

using KeldyshED: EDCore
using Keldysh
using LinearAlgebra: Diagonal, tr
using Distributed
@everywhere using SharedArrays

export computegf

"""Compute Green's function on a given Keldysh contour at inverse temperature β"""
function computegf(ed::EDCore, grid::TimeGrid, indices::IndicesType, β)
  gf = TimeGF(grid)

  en = energies(ed)
  ρ = density_matrix(ed, β)

  data = SharedArray{ComplexF64, 2}(length(grid), length(grid), pids = workers())
  all_jobs = [((n1, t1), (n2, t2)) for (n1, t1) in enumerate(grid)
                                   for (n2, t2) in enumerate(grid)]
  njobs = length(all_jobs)

  @sync @distributed for job = 1:length(all_jobs)
    (n1, t1), (n2, t2) = all_jobs[job]

    greater = θ(t1.val, t2.val)
    Δt = greater ? (t1.val.val - t2.val.val) : (t2.val.val - t1.val.val)

    conn_f = greater ? cdag_connection : c_connection
    left_mat_f, right_mat_f = greater ? (c_matrix, cdag_matrix) :
                                        (cdag_matrix, c_matrix)

    val::ComplexF64 = 0
    for outer_sp = 1:length(ed.subspaces)
      inner_sp = conn_f(ed, indices, outer_sp)
      isnothing(inner_sp) && continue

      # Eigenvalues in outer and inner subspaces
      outer_energies = en[outer_sp]
      inner_energies = en[inner_sp]

      right_mat = right_mat_f(ed, indices, outer_sp)
      outer_exp = Diagonal(exp.(1im * outer_energies * Δt))
      right_mat = right_mat * outer_exp

      left_mat = left_mat_f(ed, indices, inner_sp)
      inner_exp = Diagonal(exp.(-1im * inner_energies * Δt))
      left_mat = left_mat * inner_exp

      val += tr(ρ[outer_sp] * left_mat * right_mat)
    end
    data[n1, n2] = (greater ? -1im : 1im) * val
  end
  gf.data[:] = data
  gf
end
