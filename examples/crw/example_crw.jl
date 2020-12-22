# This script actually solves the CoeurdacierReyWinant model with a risk-adjusted linearization
# and times the methods, if desired
using BenchmarkTools, RiskAdjustedLinearizations, Test, JLD2
include("crw.jl")

# Settings
diagnostics = true

# Set up
m_crw = CoeurdacierReyWinant()
m = crw(m_crw)
z0 = copy(m.z)
y0 = copy(m.y)
Ψ0 = copy(m.Ψ)

sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference/crw_sss.jld2"), "r")

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

if diagnostics
    # See crw.jl for the definition of the functions
    # crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, and crw_endo_states
    shocks = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "crw_shocks.jld2"), "r")["shocks"]
    @test abs(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, shocks, summary_statistic = x -> norm(x, Inf))) < 3e-5
    c_err, endo_states_err = dynamic_euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, crw_endo_states, 1, shocks)
    @test c_err < 2e-5
    @test endo_states_err < 1e-3
end
