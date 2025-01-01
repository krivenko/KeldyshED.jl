# KeldyshED.jl
#
# Copyright (C) 2019-2025 Igor Krivenko
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

using Test
using HDF5
using Distributed: nprocs

using Keldysh

using KeldyshED.Operators
using KeldyshED.Hilbert
using KeldyshED: EDCore, computegf

@testset "computegf(): 3 bath sites" begin

tmax = 2.0
β = 5.0
e_d = 1.0
U = 2.0
ε = [-0.5, 0, 0.5]
V = [0.3, 0.3, 0.3]
nt = 21
ntau = 51

spins = ("down", "up")
soi = SetOfIndices([[s, a] for s in spins for a in 0:length(ε)])

# Local Hamiltonian
H_loc = e_d * (n("up", 0) + n("down", 0)) + U * n("up", 0) * n("down", 0)

# Bath Hamiltonian
H_bath = sum(e * (n("up", a) + n("down", a)) for (a, e) in enumerate(ε))

# Hybridization Hamiltonian
H_hyb = OperatorExpr{Float64}()
for s in spins
  H_hyb += sum(v * c_dag(s, a) * c(s, 0) for (a, v) in enumerate(V))
  H_hyb += sum(v * c_dag(s, 0) * c(s, a) for (a, v) in enumerate(V))
end

H = H_loc + H_bath + H_hyb

# Diagonalization
ed = EDCore(H, soi)

# Grids
grid_full = FullTimeGrid(twist(FullContour(tmax = tmax, β=β)), nt, ntau)
grid_keld = KeldyshTimeGrid(twist(KeldyshContour(tmax = tmax)), nt)
grid_imag = ImaginaryTimeGrid(ImaginaryContour(β=β), ntau)

d = IndicesType(["down", 0])
u = IndicesType(["up", 0])

function gf_is_approx(f1, f2, grid)
  all(map(((t1, t2),) -> isapprox(f1(t1, t2), f2(t1, t2), atol=1e-10),
          Iterators.product(grid, grid)))
end

gf_filler = nprocs() > 1 ? DistributedGFFiller() : SerialGFFiller()

#
# Scalar GF
#

g_full_s = [computegf(ed, grid_full, d, d, gf_filler = gf_filler),
            computegf(ed, grid_full, u, u, gf_filler = gf_filler)]
g_keld_s = [computegf(ed, grid_keld, d, d, β, gf_filler = gf_filler),
            computegf(ed, grid_keld, u, u, β, gf_filler = gf_filler)]
g_imag_s = [computegf(ed, grid_imag, d, d, gf_filler = gf_filler),
            computegf(ed, grid_imag, u, u, gf_filler = gf_filler)]

test_dir = @__DIR__
h5open(test_dir * "/gf.ref.h5", "r") do ref_file
  @test isapprox(ed.gs_energy, read(ref_file["gs_energy"]), atol = 1.e-8)
  for s = 1:2
    g_ref = read(ref_file["/gf/$(s-1)"], Keldysh.ALPSTimeGF).G

    @test gf_is_approx((t1, t2) -> g_full_s[s][t1, t2],
                       (t1, t2) -> g_ref[t1, t2],
                       g_full_s[s].grid)
    @test gf_is_approx((t1, t2) -> g_keld_s[s][t1, t2],
                       (t1, t2) -> g_ref(t1.bpoint, t2.bpoint),
                       g_keld_s[s].grid)
    @test gf_is_approx((t1, t2) -> g_imag_s[s][t1, t2],
                       (t1, t2) -> g_ref(t1.bpoint, t2.bpoint),
                       g_imag_s[s].grid)
  end
end

#
# Matrix GF
#

function test_gf_matrix_isapprox(G_matrix, G_scalar)
  @test length(G_scalar) == norbitals(G_matrix)
  grid = G_matrix.grid
  for s in eachindex(G_scalar)
    @test G_scalar[s].grid == grid
    @test gf_is_approx((t1, t2) -> G_scalar[s][t1, t2],
                       (t1, t2) -> G_matrix[t1, t2][s, s],
                       grid)
  end
end

test_gf_matrix_isapprox(
  computegf(ed, grid_full, [d, u], [d, u], gf_filler = gf_filler),
  g_full_s
)
test_gf_matrix_isapprox(
  computegf(ed, grid_keld, [d, u], [d, u], β, gf_filler = gf_filler),
  g_keld_s
)
test_gf_matrix_isapprox(
  computegf(ed, grid_imag, [d, u], [d, u], gf_filler = gf_filler),
  g_imag_s
)

#
# Lists of scalar GFs
#

function test_gf_list_isapprox(G1, G2)
  @test length(G1) == length(G2)
  grid = G1[1].grid
  for s in eachindex(G1)
    @test G1[s].grid == G2[s].grid
    @test gf_is_approx((t1, t2) -> G1[s][t1, t2],
                       (t1, t2) -> G2[s][t1, t2],
                       grid)
  end
end

test_gf_list_isapprox(
  computegf(ed, grid_full, [(d, d), (u, u)], gf_filler = gf_filler),
  g_full_s
)
test_gf_list_isapprox(
  computegf(ed, grid_keld, [(d, d), (u, u)], β, gf_filler = gf_filler),
  g_keld_s
)
test_gf_list_isapprox(
  computegf(ed, grid_imag, [(d, d), (u, u)], gf_filler = gf_filler),
  g_imag_s
)

end
