# KeldyshED

The package [KeldyshED.jl](https://github.com/krivenko/KeldyshED.jl)
is a small scale Equilibrium Exact Diagonalization solver for
finite fermionic models. It can compute single-particle Green's functions
in the (Matsubara) imaginary time domain, on the 2-branch Keldysh contour,
and on the 3-branch Konstantinov-Perel' contour.

Its main intended use is solution of auxiliary problems required by
more advanced solvers such as the
[Hybridization Expansion Quantum Monte Carlo](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.155107) and
the [N-crossing approximations](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.155107).

The initial public version of this package was a simplified Julia port of
the [TRIQS/atom_diag](https://triqs.github.io/triqs/latest/documentation/manual/triqs/atom_diag/contents.html)
library, but has since acquired a number of unique features, e.g. calculation
of [Evolution operators](@ref).

## Installation

KeldyshED.jl is a registered Julia package that can be installed via the following
invocation

```julia
using Pkg
Pkg.add("KeldyshED")
```

## Public API

```@docs
KeldyshED
```

```@contents
Pages = ["operators.md",
         "hilbert.md",
         "ed_core.md",
         "gf.md",
         "evolution.md"]
```