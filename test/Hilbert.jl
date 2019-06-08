using KeldyshED.Operators
using KeldyshED.Hilbert
using Test

@testset "SetOfIndices" begin

soi1 = SetOfIndices()
@test length(soi1) == 0
@test isempty(soi1)

for i = 4:-1:1; insert!(soi1, IndicesType([i])) end
@test length(soi1) == 4
@test soi1[IndicesType([3])] == 3

soi2 = SetOfIndices([IndicesType([i]) for i in [1,2,3,4]])
@test length(soi2) == 4
@test soi1 == soi2

soi3 = SetOfIndices()
for i = 1:2; insert!(soi3, "up", i) end
for i = 1:2; insert!(soi3, "down", i) end
@test length(soi3) == 4
@test soi3["down", 1] == 1
@test IndicesType(["up", 1]) in soi3
@test !(IndicesType(["down", 5]) in soi3)

soi4 = SetOfIndices()
for k in ["d","a","c","b"]; insert!(soi4, k) end
@test [(k, v) for (k, v) in soi4] == [(["a"],1),(["b"],2),(["c"],3),(["d"],4)]
@test collect(keys(soi4)) == [["a"],["b"],["c"],["d"]]
@test collect(values(soi4)) == [1,2,3,4]
@test collect(pairs(soi4)) == [["a"]=>1,["b"]=>2,["c"]=>3,["d"]=>4]
@test reversemap(soi4) == [["a"],["b"],["c"],["d"]]

end

@testset "FullHilbertSpace" begin

using Base.Iterators

soi = SetOfIndices([IndicesType([i,j]) for i=1:2 for j=1:4])

fhs1 = FullHilbertSpace(soi)
@test length(fhs1) == 256
@test FockState(130) in fhs1
@test !(FockState(256) in fhs1)
@test fhs1[121] == FockState(120)
@test getstateindex(fhs1, FockState(120)) == 121
@test fhs1[soi => Set{IndicesType}([])] == FockState(0)
# Fock state for C†(1,2)c†(2,4)|vac>
@test fhs1[soi => Set{IndicesType}([[1,2],[2,4]])] == FockState(130)

fhs2 = fhs1
@test length(fhs2) == 256
@test [fs for fs in fhs2] == [FockState(i) for i=0:255]
@test keys(fhs2) == LinearIndices(1:256)
@test values(fhs2) == [FockState(i) for i=0:255]
@test collect(pairs(fhs2)) == [i + 1 => FockState(i) for i=0:255]

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

soi1 = SetOfIndices([IndicesType(["up",i]) for i=1:5])
soi2 = SetOfIndices()
for i=1:2
  insert!(soi2, "up", i)
  insert!(soi2, "down", i)
end

fhs2 = FullHilbertSpace(soi2)

hss1 = HilbertSubspace(1)
insert!(hss1, fhs2[1]) # 000
insert!(hss1, fhs2[2]) # 001
insert!(hss1, fhs2[3]) # 010
insert!(hss1, fhs2[4]) # 011

@test fhs2[3] in hss1
@test !(fhs2[7] in hss1)

hss2 = HilbertSubspace(2)
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

# TODO:
#=
std::vector<int> Cdagmap(2, -1);
Cdagmap[hss1.get_index()] = hss2.get_index();
std::vector<sub_hilbert_space> sub1{hss1, hss2};
auto opCdag = imperative_operator<sub_hilbert_space, double, true>(Cdag, fop1, Cdagmap, &sub1);

state<sub_hilbert_space, double, false> start(hss0);
start(0) = 1.0;
start(1) = 2.0;
start(2) = 3.0;
start(3) = 4.0;

check_state(start, {{0, 1.0}, {1, 2.0}, {2, 3.0}, {3, 4.0}});
check_state(opCdag(start), {{4, 1.0}, {5, -2.0}, {6, -3.0}, {7, 4.0}});
=#

end

@testset "State" begin

soi = SetOfIndices([IndicesType([i]) for i=1:5])
fhs1 = FullHilbertSpace(soi)

hss1 = HilbertSubspace()
for i=1:10 insert!(hss1, FockState(i-1)) end

all_states = [(fhs1, StateVector{FullHilbertSpace, Float64}(fhs1)),
              (hss1, StateVector{HilbertSubspace, Float64}(hss1)),
              (fhs1, StateDict{FullHilbertSpace, Float64}(fhs1)),
              (hss1, StateDict{HilbertSubspace, Float64}(hss1))]

for (hs, st1) in all_states
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

soi = SetOfIndices([IndicesType(["s", i]) for i=1:3])
fhs = FullHilbertSpace(soi)

for st in [StateVector{FullHilbertSpace, Float64}(fhs),
           StateDict{FullHilbertSpace, Float64}(fhs)]
  st[1] = 0.1
  st[3] = 0.2
  st[5] = 0.3
  st[7] = 0.4

  check_state(st, Dict(fhs[1] => 0.1, fhs[3] => 0.2, fhs[5] => 0.3, fhs[7] => 0.4))

  # Project to a smaller FullHilbertSpace
  soi2 = SetOfIndices([IndicesType(["s", i]) for i=1:2])
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
