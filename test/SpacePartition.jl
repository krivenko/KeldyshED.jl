using KeldyshED.Operators
using KeldyshED.Hilbert
using SparseArrays
using Test

@testset "SpacePartition" begin

# 3 orbital Hubbard-Kanamori atom
mu = 0.7
U  = 3.0
J  = 0.3

soi = SetOfIndices()
for o = 1:3
  insert!(soi, "up", o)
  insert!(soi, "dn", o)
end

H = OperatorExpr{Float64}()

for o=1:3; H += -mu * (n("up", o) + n("dn", o)) end
for o=1:3; H += U * n("up", o) * n("dn", o) end

for o1=1:3, o2=1:3
  o1 == o2 && continue
  H += (U - 2 * J) * n("up", o1) * n("dn", o2)
end
for o1=1:3, o2=1:3
  o2 >= o1 && continue
  H += (U - 3 * J) * n("up", o1) * n("up", o2)
  H += (U - 3 * J) * n("dn", o1) * n("dn", o2)
end
for o1=1:3, o2=1:3
  o1 == o2 && continue
  H += -J * c_dag("up", o1) * c_dag("dn", o1) * c("up", o2) * c("dn", o2)
  H += -J * c_dag("up", o1) * c_dag("dn", o2) * c("up", o2) * c("dn", o1)
end

fhs = FullHilbertSpace(soi)
u1 = fhs[Set{IndicesType}([["up", 1]])]
u2 = fhs[Set{IndicesType}([["up", 2]])]
u3 = fhs[Set{IndicesType}([["up", 3]])]
d1 = fhs[Set{IndicesType}([["dn", 1]])]
d2 = fhs[Set{IndicesType}([["dn", 2]])]
d3 = fhs[Set{IndicesType}([["dn", 3]])]

Hop = Operator{FullHilbertSpace, Float64}(H, soi)

# Phase I: construct the finest possible partition

SP1 = SpacePartition{FullHilbertSpace, Float64}(fhs, Hop, false)

@test length(SP1) == 64
@test numsubspaces(SP1) == 44

# Calculated classification of states.
# Sets are used to neglect order of subspaces and of states within a subspace.
cl = [Int[] for i=1:44]
for (i, sp) in SP1
  push!(cl[sp], fhs[i])
end
cl = Set([Set(s) for s in cl])

# Expected classification of states.
ref_cl = [# N=0
          [0],
          # N=1
          [d1],
          [d2],
          [d3],
          [u1],
          [u2],
          [u3],
          # N=2, same spin
          [d1 + d2],
          [d1 + d3],
          [d2 + d3],
          [u1 + u2],
          [u1 + u3],
          [u2 + u3],
          # N=2, pair hopping
          [d1 + u1, d2 + u2, d3 + u3],
          # N=2, spin flip
          [d1 + u2, d2 + u1],
          [d1 + u3, d3 + u1],
          [d2 + u3, d3 + u2],
          # N=3
          [d1 + d2 + d3],
          [u1 + u2 + u3],
          [d1 + d2 + u1, d2 + d3 + u3],
          [d1 + d3 + u1, d2 + d3 + u2],
          [d1 + d2 + u2, d1 + d3 + u3],
          [d1 + u1 + u2, d3 + u2 + u3],
          [d2 + u1 + u2, d3 + u1 + u3],
          [d1 + u1 + u3, d2 + u2 + u3],
          [d2 + d3 + u1, d1 + d3 + u2, d1 + d2 + u3],
          [d3 + u1 + u2, d1 + u2 + u3, d2 + u1 + u3],
          # N=4, 2 holes with the same spin
          [d3 + u1 + u2 + u3],
          [d2 + u1 + u2 + u3],
          [d1 + u1 + u2 + u3],
          [d1 + d2 + d3 + u3],
          [d1 + d2 + d3 + u2],
          [d1 + d2 + d3 + u1],
          # N=4, pair hopping
          [d2 + d3 + u2 + u3, d1 + d3 + u1 + u3, d1 + d2 + u1 + u2],
          # N=4, spin flip
          [d2 + d3 + u1 + u3, d1 + d3 + u2 + u3],
          [d2 + d3 + u1 + u2, d1 + d2 + u2 + u3],
          [d1 + d3 + u1 + u2, d1 + d2 + u1 + u3],
          # N=5
          [d2 + d3 + u1 + u2 + u3],
          [d1 + d3 + u1 + u2 + u3],
          [d1 + d2 + u1 + u2 + u3],
          [d1 + d2 + d3 + u2 + u3],
          [d1 + d2 + d3 + u1 + u3],
          [d1 + d2 + d3 + u1 + u2],
          # N=6
          [d1 + d2 + d3 + u1 + u2 + u3]]

ref_cl = Set([Set(s) for s in ref_cl])

@test cl == ref_cl

@test keys(SP1) == collect(1:64)
@test values(SP1) == [sp for (i, sp) in SP1]
@test pairs(SP1) == [p for p in SP1]

# Check matrix elements

SP2 = SpacePartition{FullHilbertSpace, Float64}(fhs, Hop, true)

@test length(SP2) == 64
@test numsubspaces(SP2) == 44

ref_matrix_elements_list = [
  # N=1
  (d1, d1, -mu), (d2, d2, -mu), (d3, d3, -mu),
  (u1, u1, -mu), (u2, u2, -mu), (u3, u3, -mu),
  # N=2, same spin
  (d1 + d2, d1 + d2, -2 * mu + U - 3 * J),
  (d1 + d3, d1 + d3, -2 * mu + U - 3 * J),
  (d2 + d3, d2 + d3, -2 * mu + U - 3 * J),
  (u1 + u2, u1 + u2, -2 * mu + U - 3 * J),
  (u1 + u3, u1 + u3, -2 * mu + U - 3 * J),
  (u2 + u3, u2 + u3, -2 * mu + U - 3 * J),
  # N=2, pair hopping
  (d1 + u1, d1 + u1, -2 * mu + U),
  (d2 + u2, d2 + u2, -2 * mu + U),
  (d3 + u3, d3 + u3, -2 * mu + U),
  (d1 + u1, d2 + u2, J), (d1 + u1, d3 + u3, J), (d2 + u2, d3 + u3, J),
  (d2 + u2, d1 + u1, J), (d3 + u3, d1 + u1, J), (d3 + u3, d2 + u2, J),
  # N=2, spin flip
  (d1 + u2, d1 + u2, -2 * mu + U - 2 * J),
  (d2 + u1, d2 + u1, -2 * mu + U - 2 * J),
  (d1 + u2, d2 + u1, J),
  (d2 + u1, d1 + u2, J),
  (d1 + u3, d1 + u3, -2 * mu + U - 2 * J),
  (d3 + u1, d3 + u1, -2 * mu + U - 2 * J),
  (d1 + u3, d3 + u1, J), (d3 + u1, d1 + u3, J),
  (d2 + u3, d2 + u3, -2 * mu + U - 2 * J),
  (d3 + u2, d3 + u2, -2 * mu + U - 2 * J),
  (d2 + u3, d3 + u2, J), (d3 + u2, d2 + u3, J),
  # N=3
  (d1 + d2 + d3, d1 + d2 + d3, -3 * mu + 3 * U - 9 * J),
  (u1 + u2 + u3, u1 + u2 + u3, -3 * mu + 3 * U - 9 * J),
  (d1 + d2 + u1, d1 + d2 + u1, -3 * mu + 3 * U - 5 * J),
  (d2 + d3 + u3, d2 + d3 + u3, -3 * mu + 3 * U - 5 * J),
  (d1 + d2 + u1, d2 + d3 + u3, -J),
  (d2 + d3 + u3, d1 + d2 + u1, -J),
  (d1 + d3 + u1, d1 + d3 + u1, -3 * mu + 3 * U - 5 * J),
  (d2 + d3 + u2, d2 + d3 + u2, -3 * mu + 3 * U - 5 * J),
  (d1 + d3 + u1, d2 + d3 + u2, J),
  (d2 + d3 + u2, d1 + d3 + u1, J),
  (d1 + d2 + u2, d1 + d2 + u2, -3 * mu + 3 * U - 5 * J),
  (d1 + d3 + u3, d1 + d3 + u3, -3 * mu + 3 * U - 5 * J),
  (d1 + d2 + u2, d1 + d3 + u3, J),
  (d1 + d3 + u3, d1 + d2 + u2, J),
  (d1 + u1 + u2, d1 + u1 + u2, -3 * mu + 3 * U - 5 * J),
  (d3 + u2 + u3, d3 + u2 + u3, -3 * mu + 3 * U - 5 * J),
  (d1 + u1 + u2, d3 + u2 + u3, -J),
  (d3 + u2 + u3, d1 + u1 + u2, -J),
  (d2 + u1 + u2, d2 + u1 + u2, -3 * mu + 3 * U - 5 * J),
  (d3 + u1 + u3, d3 + u1 + u3, -3 * mu + 3 * U - 5 * J),
  (d2 + u1 + u2, d3 + u1 + u3, J),
  (d3 + u1 + u3, d2 + u1 + u2, J),
  (d1 + u1 + u3, d1 + u1 + u3, -3 * mu + 3 * U - 5 * J),
  (d2 + u2 + u3, d2 + u2 + u3, -3 * mu + 3 * U - 5 * J),
  (d1 + u1 + u3, d2 + u2 + u3, J),
  (d2 + u2 + u3, d1 + u1 + u3, J),
  (d2 + d3 + u1, d2 + d3 + u1, -3 * mu + 3 * U - 7 * J),
  (d1 + d3 + u2, d1 + d3 + u2, -3 * mu + 3 * U - 7 * J),
  (d1 + d2 + u3, d1 + d2 + u3, -3 * mu + 3 * U - 7 * J),
  (d2 + d3 + u1, d1 + d3 + u2, J),
  (d1 + d3 + u2, d2 + d3 + u1, J),
  (d2 + d3 + u1, d1 + d2 + u3, -J),
  (d1 + d2 + u3, d2 + d3 + u1, -J),
  (d1 + d3 + u2, d1 + d2 + u3, J),
  (d1 + d2 + u3, d1 + d3 + u2, J),
  (d3 + u1 + u2, d3 + u1 + u2, -3 * mu + 3 * U - 7 * J),
  (d1 + u2 + u3, d1 + u2 + u3, -3 * mu + 3 * U - 7 * J),
  (d2 + u1 + u3, d2 + u1 + u3, -3 * mu + 3 * U - 7 * J),
  (d3 + u1 + u2, d1 + u2 + u3, -J),
  (d1 + u2 + u3, d3 + u1 + u2, -J),
  (d3 + u1 + u2, d2 + u1 + u3, J),
  (d2 + u1 + u3, d3 + u1 + u2, J),
  (d1 + u2 + u3, d2 + u1 + u3, J),
  (d2 + u1 + u3, d1 + u2 + u3, J),
  # N=4, 2 holes with the same spin
  (d3 + u1 + u2 + u3, d3 + u1 + u2 + u3, -4 * mu + 6 * U - 13 * J),
  (d2 + u1 + u2 + u3, d2 + u1 + u2 + u3, -4 * mu + 6 * U - 13 * J),
  (d1 + u1 + u2 + u3, d1 + u1 + u2 + u3, -4 * mu + 6 * U - 13 * J),
  (d1 + d2 + d3 + u1, d1 + d2 + d3 + u1, -4 * mu + 6 * U - 13 * J),
  (d1 + d2 + d3 + u2, d1 + d2 + d3 + u2, -4 * mu + 6 * U - 13 * J),
  (d1 + d2 + d3 + u3, d1 + d2 + d3 + u3, -4 * mu + 6 * U - 13 * J),
  # N=4, pair hopping
  (d2 + d3 + u2 + u3, d2 + d3 + u2 + u3, -4 * mu + 6 * U - 10 * J),
  (d1 + d3 + u1 + u3, d1 + d3 + u1 + u3, -4 * mu + 6 * U - 10 * J),
  (d1 + d2 + u1 + u2, d1 + d2 + u1 + u2, -4 * mu + 6 * U - 10 * J),
  (d2 + d3 + u2 + u3, d1 + d3 + u1 + u3, J),
  (d1 + d3 + u1 + u3, d2 + d3 + u2 + u3, J),
  (d2 + d3 + u2 + u3, d1 + d2 + u1 + u2, J),
  (d1 + d2 + u1 + u2, d2 + d3 + u2 + u3, J),
  (d1 + d3 + u1 + u3, d1 + d2 + u1 + u2, J),
  (d1 + d2 + u1 + u2, d1 + d3 + u1 + u3, J),
  # N=4, spin flip
  (d2 + d3 + u1 + u3, d2 + d3 + u1 + u3, -4 * mu + 6 * U - 12 * J),
  (d1 + d3 + u2 + u3, d1 + d3 + u2 + u3, -4 * mu + 6 * U - 12 * J),
  (d2 + d3 + u1 + u3, d1 + d3 + u2 + u3, J),
  (d1 + d3 + u2 + u3, d2 + d3 + u1 + u3, J),
  (d2 + d3 + u1 + u2, d2 + d3 + u1 + u2, -4 * mu + 6 * U - 12 * J),
  (d1 + d2 + u2 + u3, d1 + d2 + u2 + u3, -4 * mu + 6 * U - 12 * J),
  (d2 + d3 + u1 + u2, d1 + d2 + u2 + u3, J),
  (d1 + d2 + u2 + u3, d2 + d3 + u1 + u2, J),
  (d1 + d3 + u1 + u2, d1 + d3 + u1 + u2, -4 * mu + 6 * U - 12 * J),
  (d1 + d2 + u1 + u3, d1 + d2 + u1 + u3, -4 * mu + 6 * U - 12 * J),
  (d1 + d3 + u1 + u2, d1 + d2 + u1 + u3, J),
  (d1 + d2 + u1 + u3, d1 + d3 + u1 + u2, J),
  # N=5
  (d2 + d3 + u1 + u2 + u3, d2 + d3 + u1 + u2 + u3, -5 * mu + 10 * U - 20 * J),
  (d1 + d3 + u1 + u2 + u3, d1 + d3 + u1 + u2 + u3, -5 * mu + 10 * U - 20 * J),
  (d1 + d2 + u1 + u2 + u3, d1 + d2 + u1 + u2 + u3, -5 * mu + 10 * U - 20 * J),
  (d1 + d2 + d3 + u2 + u3, d1 + d2 + d3 + u2 + u3, -5 * mu + 10 * U - 20 * J),
  (d1 + d2 + d3 + u1 + u3, d1 + d2 + d3 + u1 + u3, -5 * mu + 10 * U - 20 * J),
  (d1 + d2 + d3 + u1 + u2, d1 + d2 + d3 + u1 + u2, -5 * mu + 10 * U - 20 * J),
  # N=6
  (d1 + d2 + d3 + u1 + u2 + u3, d1 + d2 + d3 + u1 + u2 + u3, -6*mu+15*U-30*J)]

ref_matrix_elements = spzeros(64, 64)
for (i, j, v) in ref_matrix_elements_list
  ref_matrix_elements[i + 1, j + 1] = v
end

@test isapprox(SP2.matrix_elements, ref_matrix_elements, atol = 1e-10)

# Phase II: Check merged subspaces

SP3 = SpacePartition{FullHilbertSpace, Float64}(fhs, Hop, false)

@test length(SP3) == 64
@test numsubspaces(SP3) == 44

Cd = Operator{FullHilbertSpace, Float64}[]
C = Operator{FullHilbertSpace, Float64}[]
for o = 1:3, spin in ["up", "dn"]
  push!(Cd, Operator{FullHilbertSpace, Float64}(c_dag(spin, o), soi))
  push!(C, Operator{FullHilbertSpace, Float64}(c(spin, o), soi))

  merge_subspaces!(SP3, Cd[end], C[end]);
end

@test numsubspaces(SP3) == 44

all_ops = cat(Cd, C, dims = 1)

# Are all operators C†/C subspace-to-subspace mappings?
for op in all_ops
  # Connections between subspaces generated by 'op'
  conn = Set{Tuple{Int,Int}}()
  for i=1:64
    init_state = StateDict{FullHilbertSpace, Float64}(fhs)
    init_state[i] = 1.0
    final_state = op * init_state
    for (f, a) in pairs(final_state)
      isapprox(a, 0, atol = 1e-10) && continue
      push!(conn, (SP3[i], SP3[f]))
    end
  end

  i_subspaces = [i for (i, f) in conn]
  @test unique(i_subspaces) == i_subspaces
  f_subspaces = [f for (i, f) in conn]
  @test unique(f_subspaces) == f_subspaces
end

end
