"""
Equilibrium Exact Diagonalization solver to compute Green's functions
on the Keldysh contour.
"""
module KeldyshED

include("Operators.jl")
include("Hilbert.jl")
include("EDCore.jl")
include("GF.jl")

end # module KeldyshED
