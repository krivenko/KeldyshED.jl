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

"""
  Compute Green's function on a given Keldysh contour at inverse temperature β

  If 'β' is omitted, 'grid' must be defined on a time contour containing the
  imaginary branch.

  This method returns a vector of TimeGF objects, one element per a pair of
  indices in `c_cdag_index_pairs`.
"""
function computegf(ed::EDCore,
                   grid::TimeGrid,
                   c_cdag_index_pairs::Vector{Tuple{IndicesType, IndicesType}},
                   β = nothing)

  if isnothing(β)
    im_b = get_branch(grid.contour, imaginary_branch)
    if isnothing(im_b)
      throw(DomainError("Cannot extract inverse temperature " *
                        "-- no imaginary branch on the supplied contour"))
    end
    β = length(im_b)
  end

  en = energies(ed)
  ρ = density_matrix(ed, β)

  data = SharedArray{ComplexF64, 3}(length(c_cdag_index_pairs),
                                    length(grid),
                                    length(grid),
                                    pids = workers())
  all_jobs = [((ind_n, indices), (n1, t1), (n2, t2))
              for (ind_n, indices) in enumerate(c_cdag_index_pairs)
              for (n1, t1) in enumerate(grid)
              for (n2, t2) in enumerate(grid)]
  njobs = length(all_jobs)

  @sync @distributed for job = 1:length(all_jobs)
    (ind_n, (c_index, cdag_index)), (n1, t1), (n2, t2) = all_jobs[job]

    greater = θ(t1.val, t2.val)
    Δt = greater ? (t1.val.val - t2.val.val) : (t2.val.val - t1.val.val)

    left_conn_f = greater ? (sp) -> c_connection(ed, c_index, sp) :
                            (sp) -> cdag_connection(ed, cdag_index, sp)
    right_conn_f = greater ? (sp) -> cdag_connection(ed, cdag_index, sp) :
                             (sp) -> c_connection(ed, c_index, sp)

    left_mat_f = greater ? (sp) -> c_matrix(ed, c_index, sp) :
                           (sp) -> cdag_matrix(ed, cdag_index, sp)
    right_mat_f = greater ? (sp) -> cdag_matrix(ed, cdag_index, sp) :
                            (sp) -> c_matrix(ed, c_index, sp)

    val::ComplexF64 = 0
    for outer_sp = 1:length(ed.subspaces)
      inner_sp = right_conn_f(outer_sp)
      isnothing(inner_sp) && continue
      left_conn_f(inner_sp) != outer_sp && continue

      # Eigenvalues in outer and inner subspaces
      outer_energies = en[outer_sp]
      inner_energies = en[inner_sp]

      right_mat = right_mat_f(outer_sp)
      outer_exp = Diagonal(exp.(1im * outer_energies * Δt))
      right_mat = right_mat * outer_exp

      left_mat = left_mat_f(inner_sp)
      inner_exp = Diagonal(exp.(-1im * inner_energies * Δt))
      left_mat = left_mat * inner_exp

      val += tr(ρ[outer_sp] * left_mat * right_mat)
    end
    data[ind_n, n1, n2] = (greater ? -1im : 1im) * val
  end
  [TimeGF(data[i,:,:], grid) for i=1:length(c_cdag_index_pairs)]
end

"""
  Compute Green's function on a given Keldysh contour at inverse temperature β

  If 'β' is omitted, 'grid' must be defined on a time contour containing the
  imaginary branch.

  This method returns one TimeGF object corresponding to one diagonal matrix
  element of Keldysh GF.
"""
function computegf(ed::EDCore,
                   grid::TimeGrid,
                   c_cdag_index::IndicesType,
                   β = nothing)
  computegf(ed, grid, [(c_cdag_index, c_cdag_index)], β)[1]
end

"""
  Compute Green's function on a given Keldysh contour at inverse temperature β

  If 'β' is omitted, 'grid' must be defined on a time contour containing the
  imaginary branch.

  This method returns a matrix of TimeGF objects constructed from the direct
  product of `c_indices` and `cdag_indices`.
"""
function computegf(ed::EDCore,
                   grid::TimeGrid,
                   c_indices::Vector{IndicesType},
                   cdag_indices::Vector{IndicesType},
                   β = nothing)
  c_cdag_index_pairs = [(i, j) for i in c_indices for j in cdag_indices]
  reshape(computegf(ed, grid, c_cdag_index_pairs, β),
          (length(c_indices), length(cdag_indices)))
end
