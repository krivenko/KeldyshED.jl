# KeldyshED.jl
#
# Copyright (C) 2019-2024 Igor Krivenko
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

using LinearAlgebra: Diagonal, tr
using Distributed
using DocStringExtensions

using Keldysh

"""
$(TYPEDSIGNATURES)

Compute single-particle Keldysh Green's function

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)],
\\quad \\hat\\rho = \\frac{e^{-\\beta\\hat H}}{\\mathrm{Tr}[e^{-\\beta\\hat H}]}
```

at given contour times ``t_1``, ``t_2`` for given compound indices of creation/annihilation
operators ``i`` and ``j``.

Arguments
---------
- `ed`:         Exact Diagonalization object.
- `t1`:         Contour time ``t_1``.
- `t2`:         Contour time ``t_2``.
- `c_index`:    Compound index ``i``.
- `cdag_index`: Compound index ``j``.
- `β`:          Inverse temperature ``\\beta``.
"""
function computegf(ed::EDCore,
                   t1::BranchPoint,
                   t2::BranchPoint,
                   c_index::IndicesType,
                   cdag_index::IndicesType,
                   β::Float64)::ComplexF64
  return computegf(ed,
                   t1,
                   t2,
                   c_index,
                   cdag_index,
                   en = energies(ed),
                   ρ = density_matrix(ed, β)
         )
end

"""
$(TYPEDSIGNATURES)

Compute single-particle Keldysh Green's function

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)]
```

at given contour times ``t_1``, ``t_2`` for given compound indices of creation/annihilation
operators ``i`` and ``j``. This method is useful if the density matrix ``\\hat\\rho`` has
already been computed.

Arguments
---------
- `ed`:         Exact Diagonalization object.
- `t1`:         Contour time ``t_1``.
- `t2`:         Contour time ``t_2``.
- `c_index`:    Compound index ``i``.
- `cdag_index`: Compound index ``j``.
- `en`:         Energy levels of the system as returned by
                [`energies()`](@ref KeldyshED.energies).
- `ρ`           Density matrix ``\\hat\\rho`` as returned by
                [`density_matrix()`](@ref KeldyshED.density_matrix).
"""
function computegf(ed::EDCore,
                   t1::BranchPoint,
                   t2::BranchPoint,
                   c_index::IndicesType,
                   cdag_index::IndicesType;
                   en,
                   ρ)::ComplexF64

  greater = heaviside(t1, t2)
  Δt = greater ? (t1.val - t2.val) : (t2.val - t1.val)

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
  return (greater ? -1im : 1im) * val
end

"Abstract supertype for GF filler types."
abstract type AbstractGFFiller end

"""
An argument of this type instructs Green's function computation routines to use a serial
(non-parallelized) algorithm.
"""
struct SerialGFFiller <: AbstractGFFiller end

function (::SerialGFFiller)(f::Function,
                            gf::AbstractTimeGF{T, scalar},
                            element_list) where {T <: Number, scalar}
  for ((c_n, c_index), (cdag_n, cdag_index), (t1, t2)) in element_list
    gf[c_n, cdag_n, t1, t2] = f(t1.bpoint, t2.bpoint, c_index, cdag_index)
  end
end

"""
An argument of this type instructs Green's function computation routines to use
`Distributed.@distributed` to speed up calculation.
"""
struct DistributedGFFiller <: AbstractGFFiller end

function (::DistributedGFFiller)(f::Function,
                                 gf::AbstractTimeGF{T, scalar},
                                 element_list) where {T <: Number, scalar}
    all_jobs = collect(element_list)
    njobs = length(all_jobs)

    # Transfer computed matrix elements to process id 1
    # as (t1, t2, c_n, cdag_n, value) tuples
    gf_element_channel = RemoteChannel(
      ()->Channel{Tuple{TimeGridPoint,TimeGridPoint,Int,Int,ComplexF64}}(njobs),
      1
    )

    @distributed for job = 1:njobs
      (c_n, c_index), (cdag_n, cdag_index), (t1, t2) = all_jobs[job]
      val = f(t1.bpoint, t2.bpoint, c_index, cdag_index)
      put!(gf_element_channel, (t1, t2, c_n, cdag_n, val))
    end

    # Extract all computed elements from the channel
    for i = 1:njobs
      t1, t2, c_n, cdag_n, val = take!(gf_element_channel)
      gf[c_n, cdag_n, t1, t2] = val
    end

    close(gf_element_channel)
end

function _computegf!(ed::EDCore,
                     gf::AbstractTimeGF{T, scalar},
                     c_indices::Vector{IndicesType},
                     cdag_indices::Vector{IndicesType},
                     β::Float64,
                     gf_filler::AbstractGFFiller) where {T <: Number, scalar}

  en = energies(ed)
  ρ = density_matrix(ed, β)

  element_list = Iterators.product(
    enumerate(c_indices),
    enumerate(cdag_indices),
    TimeDomain(gf).points
  )

  gf_filler(gf, element_list) do t1, t2, c_index, cdag_index
    computegf(ed, t1, t2, c_index, cdag_index, en = en, ρ = ρ)
  end
end

#
# Compute scalar-valued Green's functions
#

"""
$(TYPEDSIGNATURES)

Compute single-particle Keldysh Green's function

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)]
```

on a 3-branch Konstantinov-Perel' contour for given compound indices of
creation/annihilation operators ``i`` and ``j``.

Arguments
---------
- `ed`:         Exact Diagonalization object.
- `grid`:       Time grid on the 3-branch contour.
- `c_index`:    Compound index ``i``.
- `cdag_index`: Compound index ``j``.
- `gf_filler`:  [`Algorithm selector`](@ref AbstractGFFiller) for GF computation.
"""
function computegf(ed::EDCore,
                   grid::FullTimeGrid,
                   c_index::IndicesType,
                   cdag_index::IndicesType;
                   gf_filler::AbstractGFFiller = SerialGFFiller())::
                   TimeInvariantFullTimeGF{ComplexF64, true}
  gf = TimeInvariantFullTimeGF(grid, 1, fermionic, true)
  _computegf!(ed, gf, [c_index], [cdag_index], grid.contour.β, gf_filler)
  return gf
end

"""
$(TYPEDSIGNATURES)

Compute single-particle Keldysh Green's function

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)],
\\quad \\hat\\rho = \\frac{e^{-\\beta\\hat H}}{\\mathrm{Tr}[e^{-\\beta\\hat H}]}
```

on a 2-branch Keldysh contour for given compound indices of creation/annihilation
operators ``i`` and ``j`` with a decoupled initial thermal state at inverse temperature
``\\beta``.

Arguments
---------
- `ed`:         Exact Diagonalization object.
- `grid`:       Time grid on the 2-branch contour.
- `c_index`:    Compound index ``i``.
- `cdag_index`: Compound index ``j``.
- `β`:          Inverse temperature ``\\beta``.
- `gf_filler`:  [`Algorithm selector`](@ref AbstractGFFiller) for GF computation.
"""
function computegf(ed::EDCore,
                   grid::KeldyshTimeGrid,
                   c_index::IndicesType,
                   cdag_index::IndicesType,
                   β::Float64;
                   gf_filler::AbstractGFFiller = SerialGFFiller())::
                   TimeInvariantKeldyshTimeGF{ComplexF64, true}
  gf = TimeInvariantKeldyshTimeGF(grid, 1, fermionic, true)
  _computegf!(ed, gf, [c_index], [cdag_index], β, gf_filler)
  return gf
end

"""
$(TYPEDSIGNATURES)

Compute single-particle Keldysh Green's function

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)]
```

on the imaginary time segment ``[0;-i\\beta]`` for given compound indices of
creation/annihilation operators ``i`` and ``j``.

Arguments
---------
- `ed`:         Exact Diagonalization object.
- `grid`:       Time grid on the imaginary time segment.
- `c_index`:    Compound index ``i``.
- `cdag_index`: Compound index ``j``.
- `gf_filler`:  [`Algorithm selector`](@ref AbstractGFFiller) for GF computation.
"""
function computegf(ed::EDCore,
                   grid::ImaginaryTimeGrid,
                   c_index::IndicesType,
                   cdag_index::IndicesType;
                   gf_filler::AbstractGFFiller = SerialGFFiller())::
                   ImaginaryTimeGF{ComplexF64, true}
  gf = ImaginaryTimeGF(grid, 1, fermionic, true)
  _computegf!(ed, gf, [c_index], [cdag_index], grid.contour.β, gf_filler)
  return gf
end

#
# Compute lists of scalar-valued Green's functions
#

"""
$(TYPEDSIGNATURES)

Compute multiple single-particle Keldysh Green's functions

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)]
```

on a 3-branch Konstantinov-Perel' contour for given compound indices of
creation/annihilation operators ``i`` and ``j``.

Arguments
---------
- `ed`:                 Exact Diagonalization object.
- `grid`:               Time grid on the 3-branch contour.
- `c_cdag_index_pairs`: List of compound index pairs ``(i, j)``.
- `gf_filler`:          [`Algorithm selector`](@ref AbstractGFFiller) for GF computation.
"""
function computegf(ed::EDCore,
                   grid::FullTimeGrid,
                   c_cdag_index_pairs::Vector{Tuple{IndicesType, IndicesType}};
                   gf_filler::AbstractGFFiller = SerialGFFiller())::
                   Vector{TimeInvariantFullTimeGF{ComplexF64, true}}
  return map(c_cdag_index_pairs) do (c_index, cdag_index)
    gf = TimeInvariantFullTimeGF(grid, 1, fermionic, true)
    _computegf!(ed, gf, [c_index], [cdag_index], grid.contour.β, gf_filler)
    return gf
  end
end

"""
$(TYPEDSIGNATURES)

Compute multiple single-particle Keldysh Green's functions

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)],
\\quad \\hat\\rho = \\frac{e^{-\\beta\\hat H}}{\\mathrm{Tr}[e^{-\\beta\\hat H}]}
```

on a 2-branch Keldysh contour for given compound indices of creation/annihilation
operators ``i`` and ``j`` with a decoupled initial thermal state at inverse temperature
``\\beta``.

Arguments
---------
- `ed`:                 Exact Diagonalization object.
- `grid`:               Time grid on the 2-branch contour.
- `c_cdag_index_pairs`: List of compound index pairs ``(i, j)``.
- `β`:                  Inverse temperature ``\\beta``.
- `gf_filler`:          [`Algorithm selector`](@ref AbstractGFFiller) for GF computation.
"""
function computegf(ed::EDCore,
                   grid::KeldyshTimeGrid,
                   c_cdag_index_pairs::Vector{Tuple{IndicesType, IndicesType}},
                   β::Float64;
                   gf_filler::AbstractGFFiller = SerialGFFiller())::
                   Vector{TimeInvariantKeldyshTimeGF{ComplexF64, true}}
  return map(c_cdag_index_pairs) do (c_index, cdag_index)
    gf = TimeInvariantKeldyshTimeGF(grid, 1, fermionic, true)
    _computegf!(ed, gf, [c_index], [cdag_index], β, gf_filler)
    return gf
  end
end

"""
$(TYPEDSIGNATURES)

Compute multiple single-particle Keldysh Green's functions

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)]
```

on the imaginary time segment ``[0;-i\\beta]`` for given compound indices of
creation/annihilation operators ``i`` and ``j``.

Arguments
---------
- `ed`:                 Exact Diagonalization object.
- `grid`:               Time grid on the imaginary time segment.
- `c_cdag_index_pairs`: List of compound index pairs ``(i, j)``.
- `gf_filler`:          [`Algorithm selector`](@ref AbstractGFFiller) for GF computation.
"""
function computegf(ed::EDCore,
                   grid::ImaginaryTimeGrid,
                   c_cdag_index_pairs::Vector{Tuple{IndicesType, IndicesType}};
                   gf_filler::AbstractGFFiller = SerialGFFiller())::
                   Vector{ImaginaryTimeGF{ComplexF64, true}}
  return map(c_cdag_index_pairs) do (c_index, cdag_index)
    gf = ImaginaryTimeGF(grid, 1, fermionic, true)
    _computegf!(ed, gf, [c_index], [cdag_index], grid.contour.β, gf_filler)
    return gf
  end
end

#
# Compute matrix-valued Green's functions
#

"""
$(TYPEDSIGNATURES)

Compute a matrix-valued single-particle Keldysh Green's function

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)]
```

on a 3-branch Konstantinov-Perel' contour for all possible pairs of compound indices
``(i, j)`` given a list of ``i`` and a list of ``j`` (the lists must have equal length).

Arguments
---------
- `ed`:           Exact Diagonalization object.
- `grid`:         Time grid on the 3-branch contour.
- `c_indices`:    List of compound indices ``i``.
- `cdag_indices`: List of compound indices ``j``.
- `gf_filler`:    [`Algorithm selector`](@ref AbstractGFFiller) for GF computation.
"""
function computegf(ed::EDCore,
                   grid::FullTimeGrid,
                   c_indices::Vector{IndicesType},
                   cdag_indices::Vector{IndicesType};
                   gf_filler::AbstractGFFiller = SerialGFFiller())::
                   TimeInvariantFullTimeGF{ComplexF64, false}
  norb = length(c_indices)
  @assert norb == length(cdag_indices)
  gf = TimeInvariantFullTimeGF(grid, norb)
  _computegf!(ed, gf, c_indices, cdag_indices, grid.contour.β, gf_filler)
  return gf
end

"""
$(TYPEDSIGNATURES)

Compute a matrix-valued single-particle Keldysh Green's function

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)],
\\quad \\hat\\rho = \\frac{e^{-\\beta\\hat H}}{\\mathrm{Tr}[e^{-\\beta\\hat H}]}
```

on a 2-branch Keldysh contour for all possible pairs of compound indices
``(i, j)`` given a list of ``i`` and a list of ``j`` (the lists must have equal length).

Arguments
---------
- `ed`:           Exact Diagonalization object.
- `grid`:         Time grid on the 2-branch contour.
- `c_indices`:    List of compound indices ``i``.
- `cdag_indices`: List of compound indices ``j``.
- `β`:            Inverse temperature ``\\beta``.
- `gf_filler`:    [`Algorithm selector`](@ref AbstractGFFiller) for GF computation.
"""
function computegf(ed::EDCore,
                   grid::KeldyshTimeGrid,
                   c_indices::Vector{IndicesType},
                   cdag_indices::Vector{IndicesType},
                   β::Float64;
                   gf_filler::AbstractGFFiller = SerialGFFiller())::
                   TimeInvariantKeldyshTimeGF{ComplexF64, false}
  norb = length(c_indices)
  @assert norb == length(cdag_indices)
  gf = TimeInvariantKeldyshTimeGF(grid, norb)
  _computegf!(ed, gf, c_indices, cdag_indices, β, gf_filler)
  return gf
end

"""
$(TYPEDSIGNATURES)

Compute a matrix-valued single-particle Keldysh Green's function

```math
G_{ij}(t_1, t_2) =
-i \\mathrm{Tr}[\\hat\\rho \\mathbb{T}_\\mathcal{C} c_i(t_1) c^\\dagger_j(t_2)]
```

on the imaginary time segment ``[0;-i\\beta]`` for all possible pairs of compound indices
``(i, j)`` given a list of ``i`` and a list of ``j`` (the lists must have equal length).

Arguments
---------
- `ed`:           Exact Diagonalization object.
- `grid`:         Time grid on the imaginary time segment.
- `c_indices`:    List of compound indices ``i``.
- `cdag_indices`: List of compound indices ``j``.
- `gf_filler`:    [`Algorithm selector`](@ref AbstractGFFiller) for GF computation.
"""
function computegf(ed::EDCore,
                   grid::ImaginaryTimeGrid,
                   c_indices::Vector{IndicesType},
                   cdag_indices::Vector{IndicesType};
                   gf_filler::AbstractGFFiller = SerialGFFiller())::
                   ImaginaryTimeGF{ComplexF64, false}
  norb = length(c_indices)
  @assert norb == length(cdag_indices)
  gf = ImaginaryTimeGF(grid, norb)
  _computegf!(ed, gf, c_indices, cdag_indices, grid.contour.β, gf_filler)
  return gf
end
