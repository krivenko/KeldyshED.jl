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

using KeldyshED.Operators
using KeldyshED.Hilbert

@testset "SetOfIndices" begin

soi1 = SetOfIndices()
@test length(soi1) == 0
@test isempty(soi1)

for i = 4:-1:1; insert!(soi1, i) end
@test length(soi1) == 4
@test soi1[[3]] == 3

soi2 = SetOfIndices([[1],[2],[3],[4]])
@test length(soi2) == 4
@test soi1 == soi2

soi3 = SetOfIndices()
for i = 1:2; insert!(soi3, "up", i) end
for i = 1:2; insert!(soi3, "down", i) end
@test length(soi3) == 4
@test soi3["down", 1] == 1
@test ["up", 1] in soi3
@test !(["down", 5] in soi3)

soi4 = SetOfIndices()
for k in ["d","a","c","b"]; insert!(soi4, k) end
soi5 = SetOfIndices([["d"],["a"],["c"],["b"]])

for soi in (soi4, soi5)
  @test [(k, v) for (k, v) in soi] == [(["a"],1),(["b"],2),(["c"],3),(["d"],4)]
  @test collect(keys(soi)) == [["a"],["b"],["c"],["d"]]
  @test collect(values(soi)) == [1,2,3,4]
  @test collect(pairs(soi)) == [["a"]=>1,["b"]=>2,["c"]=>3,["d"]=>4]
  @test reversemap(soi) == [["a"],["b"],["c"],["d"]]
end

soi_from = SetOfIndices([["a"],["c"]])
soi_to = SetOfIndices([["a"],["b"],["c"],["d"]])
@test matching_indices(soi_from, soi_to) == [1, 3]

end

@testset "FullHilbertSpace" begin

using Base.Iterators

@test length(FullHilbertSpace()) == 1

@test translate(FockState(0b0010110), [3, 2, 1, 4, 7, 6, 5]) ==
      FockState(0b1000011)
@test translate(FockState(0b1000011), [3, 2, 1, 4, 7, 6, 5], reverse=true) ==
      FockState(0b0010110)

soi = SetOfIndices([[i,j] for i=1:2 for j=1:4])

fhs1 = FullHilbertSpace(soi)
@test length(fhs1) == 256
@test FockState(130) in fhs1
@test !(FockState(256) in fhs1)
@test fhs1[121] == FockState(120)
@test getstateindex(fhs1, FockState(120)) == 121
@test fhs1[Set{IndicesType}([])] == FockState(0)
# Fock state for C†(1,2)c†(2,4)|vac>
@test fhs1[Set{IndicesType}([[1,2],[2,4]])] == FockState(130)

fhs2 = fhs1
@test length(fhs2) == 256
@test [fs for fs in fhs2] == [FockState(i) for i=0:255]
@test keys(fhs2) == LinearIndices(1:256)
@test values(fhs2) == [FockState(i) for i=0:255]
@test collect(pairs(fhs2)) == [i + 1 => FockState(i) for i=0:255]

soi_A = SetOfIndices([[1], [3], [4]])
soi_B = SetOfIndices([[2], [5]])
fhs_A = FullHilbertSpace(soi_A)
fhs_B = FullHilbertSpace(soi_B)
fhs_AB = fhs_A ⊗ fhs_B
@test length(fhs_AB) == 32
@test collect(keys(fhs_AB.soi)) == [[1], [2], [3], [4], [5]]
@test length(fhs_B ⊗ FullHilbertSpace()) == 4
@test collect(keys((fhs_B ⊗ FullHilbertSpace()).soi)) == [[2], [5]]

fhs_ABoverA = fhs_AB / fhs_A
@test length(fhs_ABoverA) == 4
@test collect(keys(fhs_ABoverA.soi)) == [[2], [5]]
@test length(fhs_B / FullHilbertSpace()) == 4
@test collect(keys((fhs_B / FullHilbertSpace()).soi)) == [[2], [5]]

@testset "product_basis_map()" begin
  fhs_A = FullHilbertSpace(SetOfIndices([[2], [5]]))
  fhs_B = FullHilbertSpace(SetOfIndices([[3], [4]]))
  fhs_big = FullHilbertSpace(SetOfIndices([[1], [2], [3], [4], [5]]))

  @test product_basis_map(fhs_A, fhs_B, fhs_big) ==
    [1 5 9 13; 3 7 11 15; 17 21 25 29; 19 23 27 31]
end

@testset "factorized_basis_map()" begin
  fhs_A = FullHilbertSpace(SetOfIndices([[2], [5]]))
  fhs_B = FullHilbertSpace(SetOfIndices([[3], [4]]))
  fhs_AB = FullHilbertSpace(SetOfIndices([[2], [3], [4], [5]]))

  bmap = product_basis_map(fhs_A, fhs_B, fhs_AB)
  fmap = factorized_basis_map(fhs_AB, fhs_A)
  @test length(fmap) == 16
  @test all(bmap[i, j] == k for (k, (i, j)) in enumerate(fmap))
end

end

# Check amplitudes of a given state
function check_state(st, ref::Dict{FockState, Float64})
  for (i, v) in pairs(st)
    @test isapprox(v, get(ref, st.hs[i], 0), atol = 1e-10)
    @test i in keys(st)
    @test v in values(st)
    @test (i => v) in pairs(st)
  end
end

@testset "HilbertSubspace" begin

Cdag = c_dag("up", 2);
@test repr(Cdag) == "1.0*c†(\"up\",2)"

soi1 = SetOfIndices([["up", i] for i=1:5])
soi2 = SetOfIndices()
for i=1:2
  insert!(soi2, "up", i)
  insert!(soi2, "down", i)
end

fhs2 = FullHilbertSpace(soi2)

hss1 = HilbertSubspace()
insert!(hss1, fhs2[1]) # 000
insert!(hss1, fhs2[2]) # 001
insert!(hss1, fhs2[3]) # 010
insert!(hss1, fhs2[4]) # 011

@test fhs2[3] in hss1
@test !(fhs2[7] in hss1)

hss2 = HilbertSubspace()
insert!(hss2, fhs2[5]) # 100
insert!(hss2, fhs2[6]) # 101
insert!(hss2, fhs2[7]) # 110
insert!(hss2, fhs2[8]) # 111

@test !(fhs2[3] in hss2)
@test fhs2[7] in hss2

@test length(hss2) == 4
@test getstateindex(hss2, FockState(0b101)) == 2
@test keys(hss2) == collect(1:4)
@test values(hss2) == map(FockState, [0b100, 0b101, 0b110, 0b111])
@test collect(pairs(hss2)) == [i => FockState(i+3) for i=1:4]

end

@testset "State" begin

soi = SetOfIndices([[i] for i=1:5])
fhs1 = FullHilbertSpace(soi)

hss1 = HilbertSubspace()
for i=1:10 insert!(hss1, FockState(i-1)) end

all_states = [(fhs1, StateVector{FullHilbertSpace, Float64}(fhs1)),
              (hss1, StateVector{HilbertSubspace, Float64}(hss1)),
              (fhs1, StateDict{FullHilbertSpace, Float64}(fhs1)),
              (hss1, StateDict{HilbertSubspace, Float64}(hss1))]

for (hs, st1) in all_states
  @test eltype(st1) == Pair{Int, Float64}

  st2 = deepcopy(st1)

  st1[1] = 3.0
  st1[4] = 5.0

  st2[1] = -3.0
  st2[4] = 5.0
  st2[10] = 2.0

  @test st1[1] == 3.0
  @test st1[4] == 5.0
  @test st2[1] == -3.0
  @test st2[4] == 5.0
  @test st2[10] == 2.0

  check_state(st1, Dict(hs[1] => 3.0, hs[4] => 5.0))
  check_state(st2, Dict(hs[1] => -3.0, hs[4] => 5.0, hs[10] => 2.0))

  @test repr(st1) == " +(3.0)|0> +(5.0)|3>"
  @test repr(st2) == " +(-3.0)|0> +(5.0)|3> +(2.0)|9>"

  check_state(2.0*st1, Dict(hs[1] => 6.0, hs[4] => 10.0))
  check_state(st1*2.0, Dict(hs[1] => 6.0, hs[4] => 10.0))
  check_state(st1/2.0, Dict(hs[1] => 1.5, hs[4] => 2.5))
  check_state(st1+st2, Dict(hs[4] => 10.0, hs[10] => 2.0))
  check_state(st1-st2, Dict(hs[1] => 6.0, hs[10] => -2.0))
  @test isapprox(dot(st1, st2), 16.0, atol = 1e-10)

  st3 = similar(st2)
  check_state(st3, Dict{FockState, Float64}())
end

end

@testset "StateProjection" begin

soi = SetOfIndices([["s", i] for i=1:3])
fhs = FullHilbertSpace(soi)

for st in [StateVector{FullHilbertSpace, Float64}(fhs),
           StateDict{FullHilbertSpace, Float64}(fhs)]
  st[1] = 0.1
  st[3] = 0.2
  st[5] = 0.3
  st[7] = 0.4

  check_state(st, Dict(fhs[1] => 0.1, fhs[3] => 0.2, fhs[5] => 0.3, fhs[7] => 0.4))

  # Project to a smaller FullHilbertSpace
  soi2 = SetOfIndices([["s", i] for i=1:2])
  fhs2 = FullHilbertSpace(soi2)
  proj_st = project(st, fhs2)
  check_state(proj_st, Dict(fhs[1] => 0.1, fhs[3] => 0.2))

  # Project to HilbertSubspace
  hss = HilbertSubspace()
  insert!(hss, fhs[5])
  insert!(hss, fhs[6])
  insert!(hss, fhs[7])
  insert!(hss, fhs[8])
  proj_st = project(st, hss)
  check_state(proj_st, Dict(fhs[5] => 0.3, fhs[7] => 0.4))
end
end

@testset "Operator" begin

soi = SetOfIndices([["up", i] for i=1:5])
fhs = FullHilbertSpace(soi)

X = 3*c_dag("up",2)*c("up",2) + 2*c_dag("up",3)*c("up",3) + c("up",2)*c("up",3)
@test repr(X) == "3.0*c†(\"up\",2)c(\"up\",2) + " *
                 "2.0*c†(\"up\",3)c(\"up\",3) + -1.0*c(\"up\",3)c(\"up\",2)"

opX = Operator{FullHilbertSpace, Float64}(X, soi)

for st in [StateVector{FullHilbertSpace, Float64}(fhs),
           StateDict{FullHilbertSpace, Float64}(fhs)]
  st[8] = 1.0
  check_state(st, Dict(fhs[8] => 1.0))

  new_state = opX * st

  check_state(new_state, Dict(fhs[2] => -1.0, fhs[8] => 5.0))
end

end

@testset "QuarticOperator" begin
  soi = SetOfIndices()
  insert!(soi, "up", 1)
  insert!(soi, "down", 1)
  insert!(soi, "up", 2)
  insert!(soi, "down", 2)

  fhs = FullHilbertSpace(soi)
  @test length(fhs) == 16

  X = -1.0 * c_dag("up", 1) * c_dag("down", 2) * c("up", 2) * c("down", 1)
  @test repr(X) == "1.0*c†(\"down\",2)c†(\"up\",1)c(\"up\",2)c(\"down\",1)"

for st in [StateVector{FullHilbertSpace, Float64}(fhs),
           StateDict{FullHilbertSpace, Float64}(fhs)]
  st[10] = 1.0                                                        # 0110
  check_state(st, Dict(fhs[10] => 1.0))                               # old state
  check_state(Operator{FullHilbertSpace, Float64}(X, soi) * st,
              Dict(fhs[7] => 1.0))                                    # new state
end

end
