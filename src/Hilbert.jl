module Hilbert

using DataStructures
using KeldyshED.Operators

export SetOfIndices, reversemap

################
# SetOfIndices #
################

"""Mapping from Operators.IndicesType to a linear index"""
mutable struct SetOfIndices
  map_index_n::SortedDict{IndicesType, Int}
end

SetOfIndices() = SetOfIndices(SortedDict{IndicesType, Int}())
function SetOfIndices(v::Vector{IndicesType})
  SetOfIndices(SortedDict{IndicesType, Int}(i => n for (n, i) in enumerate(v)))
end

"""Insert a new index sequence"""
function Base.insert!(soi::SetOfIndices, indices::IndicesType)
  insert!(soi.map_index_n, indices, length(soi.map_index_n) + 1)
  # Reorder the linear indices
  soi.map_index_n = SortedDict{IndicesType, Int}(
    k => n for (n, (k, v)) in enumerate(soi.map_index_n)
  )
end

"""Insert a new index sequence"""
function Base.insert!(soi::SetOfIndices, indices...)
  insert!(soi, IndicesType([indices...]))
end

function Base.:(==)(soi1::SetOfIndices, soi2::SetOfIndices)
  soi1.map_index_n == soi2.map_index_n
end

Base.getindex(soi::SetOfIndices, indices) = soi.map_index_n[indices]
function Base.getindex(soi::SetOfIndices, indices...)
  soi.map_index_n[IndicesType([indices...])]
end

Base.in(indices, soi::SetOfIndices) = indices in keys(soi.map_index_n)

"""Build and return the reverse map: Int -> IndicesType"""
reversemap(soi::SetOfIndices) = collect(keys(soi.map_index_n))

#####################################
# SetOfIndices: Iteration interface #
#####################################
Base.eltype(soi::SetOfIndices) = Pair{IndicesType, Int}

Base.length(soi::SetOfIndices) = length(soi.map_index_n)
Base.isempty(soi::SetOfIndices) = isempty(soi.map_index_n)

function Base.size(soi::SetOfIndices, dim = 1)
  @assert dim == 1
  length(soi)
end

Base.iterate(soi::SetOfIndices) = iterate(soi.map_index_n)
Base.iterate(soi::SetOfIndices, it) = iterate(soi.map_index_n, it)

Base.keys(soi::SetOfIndices) = keys(soi.map_index_n)
Base.values(soi::SetOfIndices) = values(soi.map_index_n)
Base.pairs(soi::SetOfIndices) = pairs(soi.map_index_n)

end # module Hilbert
