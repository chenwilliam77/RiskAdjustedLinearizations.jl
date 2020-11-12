using Documenter, RiskAdjustedLinearizations

makedocs(
    modules = [RiskAdjustedLinearizations],
    doctest = false,
    strict = false,
    clean = false,
    format = Documenter.HTML(; # prettyurls = get(ENV, "CI", "false") == "true",
                             canonical = "https://juliadocs.github.io/Documenter.jl/stable/",
                             assets=String[],
                             ),
    sitename = "RiskAdjustedLinearizations.jl",
    authors = "William Chen",
    linkcheck = false,
    pages = ["Home" => "index.md",
             "Risk-Adjusted Linearizations" => "risk_adjusted_linearization.md",
             "Numerical Algorithms" => "numerical_algorithms.md",
             "Example" => "example.md",
             "Caching" => "caching.md",
             "Diagnostics" => "diagnostics.md",
             "Tips" => "tips.md",
             ]
)

deploydocs(;
    repo = "github.com/chenwilliam77/RiskAdjustedLinearizations.jl.git",
    target = "build",
    versions = ["stable" => "v^", "v#.#"],
)
