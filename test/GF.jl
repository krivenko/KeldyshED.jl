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

using Keldysh
using KeldyshED.Operators
using KeldyshED.Hilbert
using KeldyshED: EDCore, computegf
using HDF5
using Test

@testset "computegf(): 3 bath sites" begin

tmax = 2.0
β = 5.0
e_d = 1.0
U = 2.0
ε = [-0.5, 0, 0.5]
V = [0.3, 0.3, 0.3]
npts_real = 21
npts_imag = 51

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

# Keldysh contour and grid
contour = twist(Contour(full_contour, tmax = tmax, β=β))
grid = TimeGrid(contour, npts_real = npts_real, npts_imag = npts_imag)

gf = [computegf(ed, grid, IndicesType([s, 0]), β) for s in spins]

h5open("test/GF.ref.h5", "r") do ref_file
  @test isapprox(ed.gs_energy, read(ref_file["gs_energy"]), atol = 1.e-8)
  for s = 1:2
    gf_ref = read(ref_file["/gf/$(s-1)"], TimeGF)
    @test gf[s].grid == gf_ref.grid
    @test isapprox(gf[s].data, gf_ref.data, atol = 1e-8)
  end
end

# Test other methods of computegf()
d = IndicesType(["down", 0])
u = IndicesType(["up", 0])

@test computegf(ed, grid, [(d, d), (u, u)], β) == gf

gf_matrix = computegf(ed, grid, [d,u], [d,u], β)
@test gf_matrix[1,1] == gf[1]
@test gf_matrix[1,2] == TimeGF((t1,t2) -> 0, grid)
@test gf_matrix[2,1] == TimeGF((t1,t2) -> 0, grid)
@test gf_matrix[2,2] == gf[2]

end
