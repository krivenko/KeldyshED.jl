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

using Test
using LinearAlgebra: I, Diagonal

using KeldyshED.Operators
using KeldyshED.Hilbert
using KeldyshED

function make_hamiltonian(n_orb, mu, U, J)
  soi = SetOfIndices([[s,o] for s in ("up","dn") for o = 1:n_orb])

  H = OperatorExpr{Float64}()

  for o=1:n_orb; H += -mu * (n("up", o) + n("dn", o)) end
  for o=1:n_orb; H += U * n("up", o) * n("dn", o) end

  for o1=1:n_orb, o2=1:n_orb
    o1 == o2 && continue
    H += (U - 2 * J) * n("up", o1) * n("dn", o2)
  end
  for o1=1:n_orb, o2=1:n_orb
    o2 >= o1 && continue
    H += (U - 3 * J) * n("up", o1) * n("up", o2)
    H += (U - 3 * J) * n("dn", o1) * n("dn", o2)
  end
  for o1=1:n_orb, o2=1:n_orb
    o1 == o2 && continue
    H += -J * c_dag("up", o1) * c_dag("dn", o1) * c("up", o2) * c("dn", o2)
    H += -J * c_dag("up", o1) * c_dag("dn", o2) * c("up", o2) * c("dn", o1)
  end

  (soi, H)
end

@testset "EDCore: 3-orbital Hubbard-Kanamori atom" begin

n_orb = 3
U = 1.0
J = 0.2
mu = 0.5*U
beta = 5

soi, H = make_hamiltonian(n_orb, mu, U, J)

ed = EDCore(H, soi)

n_subspaces = 44
@test length(ed.subspaces) == n_subspaces
@test isapprox(ed.gs_energy, -0.6, atol=1e-10)

# Thorough testing

basis_ref = Vector{FockState}[
[3],[5],[6],[10,17],[12,33],[20,34],[24],[40],[48],[1],[2],[4],[8],[16],[32],[7],
[14, 21, 35],[28, 42, 49],[56],[9, 18, 36],[0],[11, 38],[13, 22],[19, 37],
[25, 52],[26, 44],[41, 50],[15],[23],[29, 43],[30, 51],[39],[46, 53],[57],[58],
[60],[27, 45, 54],[31],[47],[55],[59],[61],[62],[63]]

# fock_states()
basis = fock_states(ed)
@test sort(basis) == sort(basis_ref) # Equal up to a permutation of subspaces

# energies()
en_ref = Vector{Float64}[[0],[0],[0],[0,0.4],[0,0.4],[0,0.4],[0],[0],[0],
[0.1],[0.1],[0.1],[0.1],[0.1],[0.1],[0.3],[0.3,0.9,0.9],[0.3,0.9,0.9],[0.3],
[0.4,0.4,1],[0.6],[0.9,1.3],[0.9,1.3],[0.9,1.3],[0.9,1.3],[0.9,1.3],[0.9,1.3],
[2],[2],[2,2.4],[2,2.4],[2],[2,2.4],[2],[2],[2],[2.4,2.4,3],[4.1],[4.1],[4.1],
[4.1],[4.1],[4.1],[6.6]]

en = energies(ed)
for i=1:n_subspaces
  i_ref = findfirst(x->x==basis[i], basis_ref)
  @test isapprox(en[i], en_ref[i_ref], atol = 1e-10)
end

# unitary_matrices()
Id = hcat(1) # identity matrix 1x1
u_mat_ref = Matrix{Float64}[Id, Id, Id,
[0.7071067812 0.7071067812; -0.7071067812 0.7071067812],
[0.7071067812 0.7071067812; -0.7071067812 0.7071067812],
[0.7071067812 0.7071067812; -0.7071067812 0.7071067812],
Id, Id, Id, Id, Id, Id, Id, Id, Id, Id,
[-0.5773502692 0.7071067812 0.4082482905;
  0.5773502692 0.7071067812 -0.4082482905;
 -0.5773502692 0 -0.8164965809],
[-0.5773502692 0.7071067812 0.4082482905;
  0.5773502692 0.7071067812 -0.4082482905;
 -0.5773502692 0 -0.8164965809],
Id,
[-0.1685467429 -0.7989109225 -0.5773502692;
 -0.6076037828 0.5454212223 -0.5773502692;
  0.7761505257 0.2534897002 -0.5773502692],
Id,
[-0.7071067812 -0.7071067812; -0.7071067812 0.7071067812],
[-0.7071067812 0.7071067812; 0.7071067812 0.7071067812],
[-0.7071067812 0.7071067812; 0.7071067812 0.7071067812],
[-0.7071067812 -0.7071067812; -0.7071067812 0.7071067812],
[-0.7071067812 0.7071067812; 0.7071067812 0.7071067812],
[-0.7071067812 0.7071067812; 0.7071067812 0.7071067812],
Id, Id,
[-0.7071067812 0.7071067812; 0.7071067812 0.7071067812],
[-0.7071067812 0.7071067812; 0.7071067812 0.7071067812],
Id,
[-0.7071067812 0.7071067812; 0.7071067812 0.7071067812],
Id, Id, Id,
[ 0.4082482905 0.7071067812 0.5773502692;
  0.4082482905 -0.7071067812 0.5773502692;
 -0.8164965809 0 0.5773502692],
Id, Id, Id, Id, Id, Id, Id]

u_mat = unitary_matrices(ed)
for i=1:n_subspaces
  i_ref = findfirst(x->x==basis[i], basis_ref)
  h = u_mat[i] * Diagonal(en[i]) * (u_mat[i]')
  h_ref = u_mat_ref[i_ref] * Diagonal(en_ref[i_ref]) * (u_mat_ref[i_ref]')
  @test isapprox(h, h_ref, atol = 1e-8)
end

# cdag_connection()
cdag_conn_ref = [Dict([3=>16,4=>22,5=>23,6=>17,7=>25,8=>27,9=>18,11=>1,
12=>2,13=>20,14=>4,15=>5,17=>28,18=>30,19=>34,20=>24,21=>10,22=>32,23=>29,
25=>33,26=>37,27=>31,31=>38,33=>39,35=>41,36=>42,37=>40,43=>44]),
Dict([2=>16,4=>24,5=>17,6=>23,7=>26,8=>18,9=>27,10=>1,12=>3,13=>4,14=>20,15=>6,
17=>29,18=>31,19=>35,20=>22,21=>11,23=>28,24=>32,25=>37,26=>33,27=>30,30=>38,
33=>40,34=>41,36=>43,37=>39,42=>44]),
Dict([1=>16,4=>17,5=>24,6=>22,7=>18,8=>26,9=>25,10=>2,11=>3,13=>5,14=>6,15=>20,
17=>32,18=>33,19=>36,20=>23,21=>12,22=>28,24=>29,25=>30,26=>31,27=>37,30=>39,
31=>40,34=>42,35=>43,37=>38,41=>44]),
Dict([1=>22,2=>23,3=>17,4=>25,5=>27,6=>18,9=>19,10=>20,11=>4,12=>5,14=>7,15=>8,
16=>28,17=>30,18=>34,20=>26,21=>13,22=>33,23=>31,24=>37,25=>36,27=>35,29=>38,
31=>41,32=>39,33=>42,37=>43,40=>44]),
Dict([1=>24,2=>17,3=>23,4=>26,5=>18,6=>27,8=>19,10=>4,11=>20,12=>6,13=>7,15=>9,
16=>29,17=>31,18=>35,20=>25,21=>14,22=>37,23=>30,24=>33,26=>36,27=>34,28=>38,
30=>41,32=>40,33=>43,37=>42,39=>44]),
Dict([1=>17,2=>24,3=>22,4=>18,5=>26,6=>25,7=>19,10=>5,11=>6,12=>20,13=>8,14=>9,
16=>32,17=>33,18=>36,20=>27,21=>15,22=>30,23=>37,24=>31,25=>34,26=>35,28=>39,
29=>40,30=>42,31=>43,37=>41,38=>44])]

function check_connection(conn_ref, i, j)
  # Account for a possible difference in subspace order in basis and basis_ref
  i_ref = findfirst(x->x==basis[i], basis_ref)
  j_ref = get(conn_ref, i_ref, nothing)
  (isnothing(j) && isnothing(j_ref)) || (basis[j] == basis_ref[j_ref])
end

for (indices, n) in soi
  for i=1:n_subspaces
    @test check_connection(cdag_conn_ref[n], i, cdag_connection(ed, n, i))
    @test check_connection(cdag_conn_ref[n], i, cdag_connection(ed, indices, i))
  end

  cdag_conn_mat = cdag_connection(ed, n)
  @test cdag_conn_mat == cdag_connection(ed, indices)

  @test cdag_conn_mat ==
        [(i == cdag_connection(ed, n, j)) for i=1:n_subspaces, j=1:n_subspaces]
end

# c_connection()

# 'Transpose' cdag_conn_ref
c_conn_ref = map(d -> Dict(j => i for (i, j) in d), cdag_conn_ref)

for (indices, n) in soi
  for i=1:n_subspaces
    @test check_connection(c_conn_ref[n], i, c_connection(ed, n, i))
    @test check_connection(c_conn_ref[n], i, c_connection(ed, indices, i))
  end

  c_conn_mat = c_connection(ed, n)
  @test c_conn_mat == c_connection(ed, indices)

  @test c_conn_mat ==
        [(i == c_connection(ed, n, j)) for i=1:n_subspaces, j=1:n_subspaces]
end

# monomial_connection()

# Constant monomial
let m = Operators.Monomial()
  for i=1:n_subspaces
    @test monomial_connection(ed, m, i) == i
  end
  @test monomial_connection(ed, m) ==
        BitArray([i == j for i in 1:n_subspaces, j in 1:n_subspaces])
end
# Quadratic monomial
for (indices1, n1) in soi
  for (indices2, n2) in soi
    let m = Operators.Monomial([Operators.CanonicalOperator(true, indices1),
                                Operators.CanonicalOperator(false, indices2)])
      for i=1:n_subspaces
        conn_ref = cdag_connection(ed, indices1, c_connection(ed, indices2, i))
        @test monomial_connection(ed, m, i) == conn_ref
      end
      conn_mat_ref = cdag_connection(ed, indices1) * c_connection(ed, indices2)
      @test monomial_connection(ed, m) == conn_mat_ref
    end
  end
end

# cdag_matrix() and c_matrix()
# Check that C† * C is the number of particles
for (indices, n) in soi
  for i=1:n_subspaces
    j = c_connection(ed, n, i)
    isnothing(j) && continue

    cdag_mat = cdag_matrix(ed, n, j)
    c_mat = c_matrix(ed, n, i)

    @test cdag_mat == cdag_matrix(ed, indices, j)
    @test c_mat == c_matrix(ed, indices, i)

    n_mat = cdag_mat * c_mat
    n_mat = u_mat[i] * n_mat * (u_mat[i]')

    n_mat_ref = Diagonal([digits(fs,base=2,pad=64)[n] for fs in basis[i]])
    @test isapprox(n_mat, n_mat_ref, atol = 1e-8)
  end
end

# monomial_matrix()

# Constant monomial
let m = Operators.Monomial()
  for i=1:n_subspaces
    sp_dim = length(ed.subspaces[i])
    @test monomial_matrix(ed, m, i) == Matrix{Float64}(I, sp_dim, sp_dim)
  end
end
# Quadratic monomial
for (indices1, n1) in soi
  for (indices2, n2) in soi
    let m = Operators.Monomial([Operators.CanonicalOperator(true, indices1),
                                Operators.CanonicalOperator(false, indices2)])
      for i=1:n_subspaces
        j = c_connection(ed, indices2, i)
        k = cdag_connection(ed, indices1, j)
        if j === nothing || k === nothing
          @test_throws DomainError monomial_matrix(ed, m, i)
        else
          mat_ref = cdag_matrix(ed, indices1, j) * c_matrix(ed, indices2, i)
          mat = monomial_matrix(ed, m, i)
          @test isapprox(mat, mat_ref, atol = 1e-8)
        end
      end
    end
  end
end

# operator_blocks()

# Check the case of one-monomial operators
let op_one_term = RealOperatorExpr()
  for o1=1:n_orb, o2=1:n_orb
    op_one_term = (o1 + o2) * c_dag("up", o1) * c("dn", o2)
    mon = Operators.Monomial([
      Operators.CanonicalOperator(true, IndicesType(["up", o1])),
      Operators.CanonicalOperator(false, IndicesType(["dn", o2]))
    ])

    for j=1:n_subspaces
      blocks = operator_blocks(ed, op_one_term, j)
      i = monomial_connection(ed, mon, j)
      if i === nothing
        @test isempty(blocks)
      else
        @test length(blocks) == 1
        @test blocks[i] ≈ (o1 + o2) * monomial_matrix(ed, mon, j)
      end
    end
  end
end

# Check that both methods of operator_blocks() give equivalent results
let op = ComplexOperatorExpr()
  for o1=1:n_orb, o2=1:n_orb
    op += 1im * (o1 + o2) * c_dag("up", o1; scalar_type = ComplexF64) *
                            c("dn", o2; scalar_type = ComplexF64)
  end

  blocks_ref2 = Dict{Tuple{Int64,Int64},Matrix{ComplexF64}}()

  for j=1:n_subspaces
    blocks = operator_blocks(ed, op, j)

    # Fill blocks_ref2
    for (i, mat) in blocks
      blocks_ref2[(i, j)] = mat
    end
  end

  @test operator_blocks(ed, op) == blocks_ref2
end

end

@testset "EDCore: 7-orbital Hubbard-Kanamori atom" begin

n_orb = 7
U = 1.0
J = 0.2
mu = 0.5*U
beta = 5

soi, H = make_hamiltonian(n_orb, mu, U, J)

ed = EDCore(H, soi)

# Only basic tests
@test length(ed.subspaces) == 2368
@test isapprox(ed.gs_energy, -0.6, atol=1e-10)

end
