using KeldyshED
using Documenter

makedocs(;
    modules=[
      KeldyshED,
      ],
    authors="Igor Krivenko <igor.s.krivenko@gmail.com>",
    repo="https://github.com/krivenko/KeldyshED.jl/blob/{commit}{path}#L{line}",
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
    repo = "github.com/krivenko/KeldyshED.jl.git"
)
