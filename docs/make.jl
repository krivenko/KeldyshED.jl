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

using KeldyshED
using Documenter

makedocs(;
    modules=[
      KeldyshED,
      ],
    authors="Igor Krivenko <igor.s.krivenko@gmail.com>",
    repo=Documenter.Remotes.GitHub("krivenko", "KeldyshED.jl"),
    sitename="KeldyshED.jl",
    format=Documenter.HTML(;
        canonical = "https://krivenko.github.io/KeldyshED.jl/",
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Public API" => [
            "Expressions with creation/annihilation operators of fermions" => "operators.md",
            "Hilbert spaces, states and linear operators" => "hilbert.md",
            "Exact Diagonalization solver" => "ed_core.md",
            "Single-particle Green's functions" => "gf.md",
            "Density matrix and evolution operators" => "evolution.md"
        ],
        "Usage example" => "example.md"
    ],
)

deploydocs(
    repo = "github.com/krivenko/KeldyshED.jl.git",
    versions = nothing
)
