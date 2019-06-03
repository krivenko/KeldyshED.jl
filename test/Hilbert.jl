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
