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

using Test
using LinearAlgebra: diagm, tr

using Keldysh

using KeldyshED.Operators
using KeldyshED.Hilbert
using KeldyshED

function make_hamiltonian(atoms, μ, U, t, λ)
  soi = SetOfIndices([[s, a] for s in ("up","dn") for a in atoms])
  H = OperatorExpr{Float64}()

  H += -μ * sum(n("up", a) + n("dn", a) for a in atoms)
  H += U * sum(n("up", a) * n("dn", a) for a in atoms)

  for a1 in atoms, a2 in atoms
      a1 == a2 && continue
      H += -t * c_dag("up", a1) * c("up", a2)
      H += -t * c_dag("dn", a1) * c("dn", a2)
  end

  H += λ * sum(c_dag("up", a) * c("dn", a) for a in atoms)
  H += λ * sum(c_dag("dn", a) * c("up", a) for a in atoms)

  (soi, H)
end

@testset "Partition function and density matrix" begin

  # Hubbard trimer
  atoms = 1:3
  U = 1.0
  t = 0.2
  μ = 0.5 * U
  λ = 0.1

  soi, H = make_hamiltonian(atoms, μ, U, t, λ)
  ed = EDCore(H, soi)

  β = 2.0

  en = reduce(vcat, energies(ed))
  ρ_ref = exp(-β * diagm(en))
  z_ref = tr(ρ_ref)
  ρ_ref /= z_ref

  @test isapprox(partition_function(ed, β), z_ref, atol=1e-10)
  ρ = density_matrix(ed, β)
  @test isapprox(cat(ρ..., dims=(1,2)), ρ_ref, atol=1e-10)

  @test isapprox(toeigenbasis(tofockbasis(ρ, ed), ed), ρ, atol=1e-10)

end

@testset "Reduced density matrix" begin

  # Hubbard trimer
  atoms = 1:3
  U = 1.0
  t = 0.2
  μ = 0.5 * U
  λ = 0.1
  β = 2.0

  soi, H = make_hamiltonian(atoms, μ, U, t, λ)
  ed = EDCore(H, soi)
  fhs = ed.full_hs

  ρ_fock = tofockbasis(density_matrix(ed, β), ed)

  #
  # Unreduced density matrix
  #

  ρ_red = reduced_density_matrix(ed, soi, β)
  for (ρ_ss, hss) in zip(ρ_fock, ed.subspaces)
    for (i1, fs1) in pairs(hss), (i2, fs2) in pairs(hss)
      @test ρ_red[getstateindex(fhs, fs1), getstateindex(fhs, fs2)] ==
            ρ_ss[i1, i2]
    end
  end

  # Fully reduced density matrix
  @test reduced_density_matrix(ed, SetOfIndices(), β) ≈ ones(1, 1)

  #
  # Partially reduced density matrix
  #

  ρ_fock_flat = zeros(64, 64)
  for (ρ_ss, hss) in zip(ρ_fock, ed.subspaces)
    for (i1, fs1) in pairs(hss), (i2, fs2) in pairs(hss)
      ρ_fock_flat[getstateindex(fhs, fs1), getstateindex(fhs, fs2)] =
                 ρ_ss[i1, i2]
    end
  end

  soi1 = SetOfIndices([[s, 1] for s in ("up","dn")])
  fhs1 = FullHilbertSpace(soi1)
  bmap = product_basis_map(fhs1, fhs / fhs1, fhs)
  ρ_red_ref = zeros(4, 4)
  for j in 1:16
    for i1 in 1:4, i2 in 1:4
      ρ_red_ref[i1, i2] += ρ_fock_flat[bmap[i1, j], bmap[i2, j]]
    end
  end
  @test reduced_density_matrix(ed, soi1, β) ≈ ρ_red_ref

end

@testset "Evolution operator" begin

  # Hubbard trimer
  atoms = 1:3
  U = 1.0
  t = 0.2
  μ = 0.5 * U
  λ = 0.1

  soi, H = make_hamiltonian(atoms, μ, U, t, λ)
  ed = EDCore(H, soi)

  tmax = 2.0
  β = 2.0

  nt = 11
  nτ = 21

  function ref(sp, t1, t2)
    exp(-im * diagm(ed.eigensystems[sp].eigenvalues) *
                    (t1.bpoint.val - t2.bpoint.val))
  end

  grid_full = FullTimeGrid(twist(FullContour(tmax=tmax, β=β)), nt, nτ)
  grid_keld = KeldyshTimeGrid(twist(KeldyshContour(tmax=tmax)), nt)
  grid_imag = ImaginaryTimeGrid(ImaginaryContour(β=β), nτ)

  for grid in (grid_full, grid_keld, grid_imag)
    S = evolution_operator(ed, grid)
    @test all(isapprox(S[sp][t1, t2], ref(sp, t1, t2), atol=1e-10)
      for (t1, t2, sp) in Iterators.product(grid, grid, eachindex(S))
    )

    S_rec = toeigenbasis(tofockbasis(S, ed), ed)
    @test all(isapprox(S_rec[sp][t1, t2], S[sp][t1, t2], atol=1e-10)
      for (t1, t2, sp) in Iterators.product(grid, grid, eachindex(S))
    )
  end

end

@testset "Reduced evolution operator" begin

  # Hubbard trimer
  atoms = 1:3
  U = 1.0
  t = 0.2
  μ = 0.5 * U
  λ = 0.1

  tmax = 1
  β = 2.0

  nt = 11
  nτ = 21

  soi, H = make_hamiltonian(atoms, μ, U, t, λ)
  ed = EDCore(H, soi)
  fhs = ed.full_hs

  grid = FullTimeGrid(twist(FullContour(tmax=tmax, β=β)), nt, nτ)
  S_fock = tofockbasis(evolution_operator(ed, grid), ed)

  #
  # Unreduced evolution operator
  #

  S_red = reduced_evolution_operator(ed, soi, grid)
  for (S_ss, hss) in zip(S_fock, ed.subspaces)
    for (i1, fs1) in pairs(hss), (i2, fs2) in pairs(hss)
      @test all(S_red[getstateindex(fhs, fs1), getstateindex(fhs, fs2), t1, t2]
                == S_ss[i1, i2, t1, t2] for t1 in grid, t2 in grid)
    end
  end

  #
  # Partially reduced evolution operator
  #

  S_fock_flat = GenericTimeGF(ComplexF64, grid, 64)
  for (S_ss, hss) in zip(S_fock, ed.subspaces)
    for (i1, fs1) in pairs(hss), (i2, fs2) in pairs(hss)
      for t1 in grid, t2 in grid
        S_fock_flat[getstateindex(fhs, fs1), getstateindex(fhs, fs2), t1, t2] =
                    S_ss[i1, i2, t1, t2]
      end
    end
  end

  soi1 = SetOfIndices([[s, 1] for s in ("up","dn")])
  fhs1 = FullHilbertSpace(soi1)
  bmap = product_basis_map(fhs1, fhs / fhs1, fhs)
  S_red_ref = GenericTimeGF(ComplexF64, grid, 4)
  for j in 1:16
    for i1 in 1:4, i2 in 1:4
      for t1 in grid, t2 in grid
        S_red_ref[i1, i2, t1, t2] +=
          S_fock_flat[bmap[i1, j], bmap[i2, j], t1, t2]
      end
    end
  end
  S_red = reduced_evolution_operator(ed, soi1, grid)
  @test all(S_red[t1, t2] ≈ S_red_ref[t1, t2] for t1 in grid, t2 in grid)

end
