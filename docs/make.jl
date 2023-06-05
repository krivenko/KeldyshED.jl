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
        "Expressions with creation/annihilation operators of fermions" => "operators.md",
        "Hilbert spaces" => "hilbert.md",
        "Exact diagonalization core" => "ed_core.md",
        "Single-particle Green's functions" => "gf.md",
        "Evolution operators" => "evolution.md"
    ],
)

deploydocs(
    repo = "github.com/krivenko/KeldyshED.jl.git"
)