# Usage example

This example shows how to use `KeldyshED` to diagonalize a
model Hamiltonian, a system of two coupled 2-orbital Hubbard-Kanamori
atoms.

```math
    \begin{align*}
    \hat H &= \sum_{i=1}^2 \hat H_{loc,i} + \hat H_{hop},\\
    \hat H_{loc,i} &=
    -\mu \sum_{\sigma} \sum_{m=1}^2 n_{i,\sigma,m} +\\
    &+ U \sum_{m=1}^2 n_{i,\uparrow,m} n_{i,\downarrow,m}
    + (U - 2J) \sum_{m\neq m'=1}^2
        n_{i,\uparrow,m} n_{i,\downarrow,m'}
    + (U - 3J) \sum_{m<m'}^2 \sum_\sigma
        n_{i,\sigma,m} n_{i,\sigma,m'} +\\
    &+ J \sum_{m\neq m'=1}^2
       (c^\dagger_{i,\uparrow,m} c^\dagger_{i,\downarrow,m'}
        c_{i,\downarrow,m} c_{i,\uparrow,m'}
      + c^\dagger_{i,\uparrow,m} c^\dagger_{i,\downarrow,m}
        c_{i,\downarrow,m'} c_{i,\uparrow,m'}),\\
    \hat H_{hop} &= t \sum_{\sigma}\sum_{m=1}^2
        (c^\dagger_{1,\sigma,m} c_{2,\sigma,m} + h.c.)
    \end{align*}
```

```@example 1
using KeldyshED.Operators: IndicesType, OperatorExpr, c, c_dag, n
using KeldyshED.Hilbert: SetOfIndices
using KeldyshED: EDCore

norb = 2 # Number of orbitals

# Construct a set of compound indices (atom, spin, orbital)
soi = SetOfIndices()
for atom in 1:2
    for orb in 1:norb
        insert!(soi, atom, "up", orb)
        insert!(soi, atom, "dn", orb)
    end
end

# Define system's Hamiltonian
function make_hamiltonian(μ, U, J, t)
    H = OperatorExpr{Float64}()

    # Local terms
    for atom in 1:2
        for orb in 1:norb
            H += -μ * (n(atom, "up", orb) + n(atom, "dn", orb))
            H += U * n(atom, "up", orb) * n(atom, "dn", orb)
        end

        for orb1 in 1:norb, orb2 in 1:norb
            orb1 == orb2 && continue
            H += (U - 2 * J) * n(atom, "up", orb1) * n(atom, "dn", orb2)
        end
        for orb1 in 1:norb, orb2 in 1:norb
            orb2 >= orb1 && continue
            H += (U - 3 * J) * n(atom, "up", orb1) * n(atom, "up", orb2)
            H += (U - 3 * J) * n(atom, "dn", orb1) * n(atom, "dn", orb2)
        end
        for orb1 in 1:norb, orb2 in 1:norb
            orb1 == orb2 && continue
            H += -J * c_dag(atom,"up", orb1) * c_dag(atom, "dn", orb1) *
                    c(atom, "up", orb2) * c(atom, "dn", orb2)
            H += -J * c_dag(atom,"up", orb1) * c_dag(atom, "dn", orb2) *
                    c(atom, "up", orb2) * c(atom, "dn", orb1)
        end
    end

    # Hopping terms between the two atoms
    for spin in ("up", "dn")
        for orb in 1:norb
            H += t * (c_dag(1, spin, orb) * c(2, spin, orb) +
                      c_dag(2, spin, orb) * c(1, spin, orb))
        end
    end

    return H
end

H = make_hamiltonian(
    1.0, # Chemical potential
    3.0, # Hubbard interaction
    0.3, # Hund coupling
    0.5  # Hopping constant
)

# Diagonalize the system
ed = EDCore(H, soi)

println("Hilbert space dimension: $(length(ed.full_hs))")
println("Dimensions of invariant subspaces (sectors) of the Hamiltonian:")
println(IOContext(stdout, :limit => true), length.(ed.subspaces))
println("Ground state energy: $(ed.gs_energy)")
```

Having diagonalized the Hamiltonian, we can calculate a single-particle
Keldysh Green's function. Here, it is computed for atom 1 and
for all combinations of spin and orbital indices. Time arguments of the
Green's function are defined on a 3-branch Konstantinov-Perel' contour.

```@example 1
using Keldysh: FullContour, FullTimeGrid
using KeldyshED: computegf, DistributedGFFiller

tmax = 10.0 # Maximum observation time on the real branches
β = 5.0     # Inverse temperature

# `FullContour` stands for the 3-branch contour
contour = FullContour(; tmax=tmax, β=β)

nt = 11 # Number of time points on each of the two real branches
nτ = 5  # Number of time points on the imaginary branch

# Discrete time grid on the contour
grid = FullTimeGrid(contour, nt, nτ)

# List of compound indices of the Green's function
op_indices = [IndicesType([1, spin, orb])
              for spin in ("up", "dn") for orb in 1:norb]

# `gf_filler = DistributedGFFiller()` instructs `computegf()` to use
# a parallelized algorithm based on `Distributed.@distributed`
G = computegf(ed,
              grid,
              op_indices, op_indices;
              gf_filler = DistributedGFFiller())
```

One can also compute the evolution operator of the system,
```math
    \hat S(t, t') = \exp\left(-i \int_{t'}^{t} \hat H d\bar t\right)
```
and its reduced version
```math
    \hat S_1(t, t') =
    \mathrm{Tr}_2\left[\exp\left(-i \int_{t'}^{t} \hat H d\bar t\right)\right],
```
where all degrees of freedom on the second atom are traced out.

```@example 1
using KeldyshED: evolution_operator, reduced_evolution_operator

S = evolution_operator(ed, grid)

# Compound indices on the first atom
soi1 = SetOfIndices()
for orb in 1:norb
    insert!(soi1, 1, "up", orb)
    insert!(soi1, 1, "dn", orb)
end
S_1 = reduced_evolution_operator(ed, soi1, grid)
```