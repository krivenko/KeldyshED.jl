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

@testset "HilbertSpace" begin

using Base.Iterators

soi = SetOfIndices([IndicesType([i,j]) for i=1:2 for j=1:4])

hs1 = HilbertSpace(soi)
@test length(hs1) == 256
@test FockState(130) in hs1
@test !(FockState(256) in hs1)
@test hs1[121] == FockState(120)
@test getstateindex(hs1, FockState(120)) == 121
@test hs1[soi => Set{IndicesType}([])] == FockState(0)
# Fock state for C†(1,2)c†(2,4)|vac>
@test hs1[soi => Set{IndicesType}([[1,2],[2,4]])] == FockState(130)

hs2 = hs1
@test length(hs2) == 256
@test [fs for fs in hs2] == [FockState(i) for i=0:255]
@test keys(hs2) == LinearIndices(1:256)
@test values(hs2) == [FockState(i) for i=0:255]
@test collect(pairs(hs2)) == [i + 1 => FockState(i) for i=0:255]

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

hs2 = HilbertSpace(soi2)

hss1 = HilbertSubspace(1)
insert!(hss1, hs2[1]) # 000
insert!(hss1, hs2[2]) # 001
insert!(hss1, hs2[3]) # 010
insert!(hss1, hs2[4]) # 011

@test hs2[3] in hss1
@test !(hs2[7] in hss1)

hss2 = HilbertSubspace(2)
insert!(hss2, hs2[5]) # 100
insert!(hss2, hs2[6]) # 101
insert!(hss2, hs2[7]) # 110
insert!(hss2, hs2[8]) # 111

@test !(hs2[3] in hss2)
@test hs2[7] in hss2

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
