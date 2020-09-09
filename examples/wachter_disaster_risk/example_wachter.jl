# This script actually solves the WachterDisasterRisk model with a risk-adjusted linearization
# and times the methods, if desired
using BenchmarkTools, RiskAdjustedLinearizations
include("wachter.jl")

# Settings: what do you want to do?
time_methods        = true
numerical_algorithm = :relaxation
autodiff            = false

# Set up
autodiff_method = autodiff ? :forwarddiff : :central
m_wachter = WachterDisasterRisk()
m = inplace_wachter_disaster_risk(m_wachter)
z0 = copy(m.z)
y0 = copy(m.y)
Ψ0 = copy(m.Ψ)

# Solve!
solve!(m; method = numerical_algorithm, autodiff = autodiff_method)

if time_methods
    @btime "Deterministic steady state" begin
        solve!(m, z0, y0; method = :deterministic, autodiff = autodiff_method, verbose = :none)
    end

    # Use deterministic steady state as guesses
    solve!(m, z0, y0; method = :deterministic, autodiff = autodiff_method, verbose = :none)
    zdet = copy(m.z)
    ydet = copy(m.y)
    Ψdet = copy(m.Ψ)

    @btime "Relaxation method" begin # called the "iterative" method in the original paper
        solve!(m, zdet, ydet, Ψdet; method = :relaxation, autodiff = autodiff_method, verbose = :none)
    end

    @btime "Homotopy method" begin # called the "continuation" method in the original paper, but is called homotopy in the original code
        # solve!(m, zdet, ydet, Ψdet; method = :homotopy, autodiff = autodiff_method, verbose = :none)
    end
end
