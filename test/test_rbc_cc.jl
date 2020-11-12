# This script actually solves the WachterDisasterRisk model with a risk-adjusted linearization
# and times the methods, if desired
using BenchmarkTools, RiskAdjustedLinearizations, Test
include(joinpath(dirname(@__FILE__), "..", "examples", "rbc_cc", "rbc_cc.jl"))

# Set up
m_rbc_cc = RBCCampbellCochraneHabits()
m = rbc_cc(m_rbc_cc, 0)

# Solve!
solve!(m; algorithm = :relaxation)

sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "reference", "rbccc_sss_iterative_output.jld2"), "r")

@test isapprox(sssout["z_rss"], m.z, atol=1e-4)
@test isapprox(sssout["y_rss"], m.y, atol=1e-4)
@test isapprox(sssout["Psi_rss"], m.Ψ, atol=1e-4)
