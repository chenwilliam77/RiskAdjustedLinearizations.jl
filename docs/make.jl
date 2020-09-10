using Documenter, RiskAdjustedLinearizations

makedocs(
  modules = [RiskAdjustedLinearizations],
  doctest = true
  linkcheck = false,
  strict = true,
  sitename = "RiskAdjustedLinearizations.jl",
  pages = ["Home" => "index.md",
           "Risk-Adjusted Linearizations" => "risk_adjusted_linearization.md",
           "Numerical Algorithms" => "numerical_algorithms.md",
           "Example" => "example.md",
           "Tips" => "tips.md"
          ]
)

deploydocs(repo = "github.com/chenwilliam77/RiskAdjustedLinearizations.jl.git")
