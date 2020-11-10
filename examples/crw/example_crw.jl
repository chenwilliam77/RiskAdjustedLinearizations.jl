# This script actually solves the WachterDisasterRisk model with a risk-adjusted linearization
# and times the methods, if desired
using BenchmarkTools, RiskAdjustedLinearizations, Test, JLD2
include("crw.jl")

# Set up
m_crw = CoeurdacierReyWinant()
m = crw(m_crw)
z0 = copy(m.z)
y0 = copy(m.y)
Ψ0 = copy(m.Ψ)

sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../test/reference/crw_sss.jld2"), "r")

# Small perturbation b/c initialized at the stochastic steady state from a saved file
m.z .= 1.1 * m.z
m.y .= 1.1 * m.y
m.Ψ .= 1.1 * m.Ψ

# Solve!
solve!(m, m.z, m.y, m.Ψ; algorithm = :homotopy)

# Only homotopy seems to work for this model. The relaxation algorithm
# has trouble finding an answer with smaller error than 1e-3
# solve!(m, m.z, m.y, m.Ψ; algorithm = :relaxation, verbose = :high, ftol = 5e-5, damping = .9)

@test isapprox(sssout["z_rss"], m.z)
@test isapprox(sssout["y_rss"], m.y)
@test isapprox(sssout["Psi_rss"], m.Ψ)
