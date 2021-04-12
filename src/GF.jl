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

export computegf

function _computegf!(ed::EDCore,
                     gf::AbstractTimeGF{T, scalar},
                     c_indices::Vector{IndicesType},
                     cdag_indices::Vector{IndicesType},
                     β::Float64) where {T <: Number, scalar}

  en = energies(ed)
  ρ = density_matrix(ed, β)

  norb = norbitals(gf)
  D = TimeDomain(gf)

  all_jobs = [((c_n, c_index), (cdag_n, cdag_index), (t1, t2))
              for (c_n, c_index) in enumerate(c_indices)
              for (cdag_n, cdag_index) in enumerate(cdag_indices)
              for (t1, t2) in D.points]

  @sync @distributed for job = 1:length(all_jobs)
    (c_n, c_index), (cdag_n, cdag_index), (t1, t2) = all_jobs[job]

    greater = heaviside(t1.val, t2.val)
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
    if scalar
      gf[t1, t2] = (greater ? -1im : 1im) * val
    else
      # A bit hacky ...
      val_mat = zeros(T, norb, norb)
      val_mat[c_n, cdag_n] = (greater ? -1im : 1im) * val
      gf[t1, t2] += val_mat
    end
    # TODO: Reduction over workers
  end
end

#
# Compute scalar-valued Green's functions
#

"""
  Compute Green's function on a full 3-branch contour

  This method returns a scalar-valued `TimeInvariantFullTimeGF` object.
"""
function computegf(ed::EDCore,
                   grid::FullTimeGrid,
                   c_index::IndicesType,
                   cdag_index::IndicesType)::
                   TimeInvariantFullTimeGF{ComplexF64, true}
  gf = TimeInvariantFullTimeGF(grid, 1, fermionic, true)
  _computegf!(ed, gf, [c_index], [cdag_index], grid.contour.β)
  gf
end

"""
  Compute Green's function on a Keldysh contour with a decoupled initial
  thermal state at inverse temperature β

  This method returns a scalar-valued `TimeInvariantKeldyshTimeGF` object.
"""
function computegf(ed::EDCore,
                   grid::KeldyshTimeGrid,
                   c_index::IndicesType,
                   cdag_index::IndicesType,
                   β::Float64)::
                   TimeInvariantKeldyshTimeGF{ComplexF64, true}
  gf = TimeInvariantKeldyshTimeGF(grid, 1, fermionic, true)
  _computegf!(ed, gf, [c_index], [cdag_index], β)
  gf
end

"""
  Compute imaginary time Green's function

  This method returns a scalar-valued `ImaginaryTimeGF` object.
"""
function computegf(ed::EDCore,
                   grid::ImaginaryTimeGrid,
                   c_index::IndicesType,
                   cdag_index::IndicesType)::
                   ImaginaryTimeGF{ComplexF64, true}
  gf = ImaginaryTimeGF(grid, 1, fermionic, true)
  _computegf!(ed, gf, [c_index], [cdag_index], grid.contour.β)
  gf
end

#
# Compute lists of scalar-valued Green's functions
#

"""
  Compute Green's function on a full 3-branch contour

  This method returns a vector of scalar-valued `TimeInvariantFullTimeGF`
  objects, one element per a pair of indices in `c_cdag_index_pairs`.
"""
function computegf(ed::EDCore,
                   grid::FullTimeGrid,
                   c_cdag_index_pairs::Vector{Tuple{IndicesType, IndicesType}})::
                   Vector{TimeInvariantFullTimeGF{ComplexF64, true}}
  map(c_cdag_index_pairs) do (c_index, cdag_index)
    gf = TimeInvariantFullTimeGF(grid, 1, fermionic, true)
    _computegf!(ed, gf, [c_index], [cdag_index], grid.contour.β)
    gf
  end
end

"""
  Compute Green's function on a Keldysh contour with a decoupled initial thermal
  state at inverse temperature β

  This method returns a vector of scalar-valued `TimeInvariantKeldyshTimeGF`
  objects, one element per a pair of indices in `c_cdag_index_pairs`.
"""
function computegf(ed::EDCore,
                   grid::KeldyshTimeGrid,
                   c_cdag_index_pairs::Vector{Tuple{IndicesType, IndicesType}},
                   β::Float64)::
                   Vector{TimeInvariantKeldyshTimeGF{ComplexF64, true}}
  map(c_cdag_index_pairs) do (c_index, cdag_index)
    gf = TimeInvariantKeldyshTimeGF(grid, 1, fermionic, true)
    _computegf!(ed, gf, [c_index], [cdag_index], β)
    gf
  end
end

"""
  Compute imaginary time Green's function

  This method returns a vector of scalar-valued `ImaginaryTimeGF` objects,
  one element per a pair of indices in `c_cdag_index_pairs`.
"""
function computegf(ed::EDCore,
                   grid::ImaginaryTimeGrid,
                   c_cdag_index_pairs::Vector{Tuple{IndicesType, IndicesType}})::
                   Vector{ImaginaryTimeGF{ComplexF64, true}}
  map(c_cdag_index_pairs) do (c_index, cdag_index)
    gf = ImaginaryTimeGF(grid, 1, fermionic, true)
    _computegf!(ed, gf, [c_index], [cdag_index], grid.contour.β)
    gf
  end
end

#
# Compute matrix-valued Green's functions
#

"""
  Compute Green's function on a full 3-branch contour

  This method returns a matrix-valued `TimeInvariantFullTimeGF` objects
  constructed from a direct product of `c_indices` and `cdag_indices`.
"""
function computegf(ed::EDCore,
                   grid::FullTimeGrid,
                   c_indices::Vector{IndicesType},
                   cdag_indices::Vector{IndicesType})::
                   TimeInvariantFullTimeGF{ComplexF64, false}
  norb = length(c_indices)
  @assert norb == length(cdag_indices)
  gf = TimeInvariantFullTimeGF(grid, norb)
  _computegf!(ed, gf, c_indices, cdag_indices, grid.contour.β)
  gf
end

"""
  Compute Green's function on a Keldysh contour with a decoupled initial thermal
  state at inverse temperature β

  This method returns a matrix-valued `TimeInvariantKeldyshTimeGF` objects
  constructed from a direct product of `c_indices` and `cdag_indices`.
"""
function computegf(ed::EDCore,
                   grid::KeldyshTimeGrid,
                   c_indices::Vector{IndicesType},
                   cdag_indices::Vector{IndicesType},
                   β::Float64)::
                   TimeInvariantKeldyshTimeGF{ComplexF64, false}
  norb = length(c_indices)
  @assert norb == length(cdag_indices)
  gf = TimeInvariantKeldyshTimeGF(grid, norb)
  _computegf!(ed, gf, c_indices, cdag_indices, β)
  gf
end

"""
  Compute imaginary time Green's function

  This method returns a matrix-valued `ImaginaryTimeGF` objects
  constructed from a direct product of `c_indices` and `cdag_indices`.
"""
function computegf(ed::EDCore,
                   grid::ImaginaryTimeGrid,
                   c_indices::Vector{IndicesType},
                   cdag_indices::Vector{IndicesType})::
                   ImaginaryTimeGF{ComplexF64, false}
  norb = length(c_indices)
  @assert norb == length(cdag_indices)
  gf = ImaginaryTimeGF(grid, norb)
  _computegf!(ed, gf, c_indices, cdag_indices, grid.contour.β)
  gf
end
