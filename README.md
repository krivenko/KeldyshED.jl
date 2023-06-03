KeldyshED.jl
============

[![CI](https://github.com/krivenko/KeldyshED.jl/actions/workflows/CI.yml/badge.svg)](
https://github.com/krivenko/KeldyshED.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-red)](
https://krivenko.github.io/KeldyshED.jl)

Equilibrium Exact Diagonalization solver for finite fermionic models that can
also compute Green's functions on the Keldysh contour.

Copyright (C) 2019-2023 Igor Krivenko <igor.s.krivenko@gmail.com>

Copyright (C) 2015 P. Seth, I. Krivenko, M. Ferrero and O. Parcollet

This is my first attempt at writing Julia code and, to a large extent, a
simplified port of the `TRIQS/atom_diag` library.

Special thanks to Joseph Kleinhenz for reviewing my code, as well as for writing
the [Keldysh.jl](https://github.com/kleinhenz/Keldysh.jl) library, which `KeldyshED.jl`
depends on.

Usage
-----

KeldyshED.jl can be installed by typing the following command in Julia's `Pkg` REPL:

    add KeldyshED

Some usage examples can be found in the `test` subdirectory.

License
-------

KeldyshED.jl is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

KeldyshED.jl is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
KeldyshED.jl (in the file LICENSE in this directory). If not, see
<http://www.gnu.org/licenses/>.
