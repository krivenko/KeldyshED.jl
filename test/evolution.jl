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

using Keldysh
using KeldyshED.Operators
using KeldyshED.Hilbert
using KeldyshED
using Test
using LinearAlgebra: diagm, tr

function make_hamiltonian()
  # Hubbard trimer
  n_atom = 3
  U = 1.0
  t = 0.2
  mu = 0.5 * U

  soi = SetOfIndices([[s, a] for s in ("up","dn") for a=1:n_atom])
  H = OperatorExpr{Float64}()

  H += sum(-mu * (n("up", a) + n("dn", a)) for a=1:n_atom)
  H += sum(U * n("up", a) * n("dn", a) for a=1:n_atom)

  for a1=1:n_atom, a2=1:n_atom
      a1 == a2 && continue
      H += -t * c_dag("up", a1) * c("up", a2)
      H += -t * c_dag("dn", a1) * c("dn", a2)
  end
  (soi, H)
end

@testset "Partition function and density matrix" begin
  soi, H = make_hamiltonian()
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

@testset "Evolution operator" begin
  soi, H = make_hamiltonian()
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