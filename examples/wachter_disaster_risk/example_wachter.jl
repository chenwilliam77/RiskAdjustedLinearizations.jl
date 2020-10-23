# This script actually solves the WachterDisasterRisk model with a risk-adjusted linearization
# and times the methods, if desired
using BenchmarkTools, RiskAdjustedLinearizations, Test
include("wachter.jl")

# Settings: what do you want to do?
time_methods        = false
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
solve!(m; algorithm = numerical_algorithm, autodiff = autodiff_method)

if numerical_algorithm == :relaxation
    sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../test/reference/iterative_sss_output.jld2"), "r")

elseif numerical_algorithm == :homotopy
    sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../test/reference/homotopy_sss_output.jld2"), "r")
end

@test isapprox(sssout["z_rss"], m.z, atol=1e-4)
@test isapprox(sssout["y_rss"], m.y, atol=1e-4)
@test isapprox(sssout["Psi_rss"], m.Ψ, atol=1e-4)

if time_methods
    println("Deterministic steady state")
    @btime begin
        solve!(m, z0, y0; algorithm = :deterministic, autodiff = autodiff_method, verbose = :none)
    end

    # Use deterministic steady state as guesses
    solve!(m, z0, y0; algorithm = :deterministic, autodiff = autodiff_method, verbose = :none)
    zdet = copy(m.z)
    ydet = copy(m.y)
    Ψdet = copy(m.Ψ)

    println("Relaxation method")
    @btime begin # called the "iterative" method in the original paper
        solve!(m, zdet, ydet, Ψdet; algorithm = :relaxation, autodiff = autodiff_method, verbose = :none)
    end

    println("Relaxation method with Anderson acceleration")
    @btime begin # called the "iterative" method in the original paper
        solve!(m, zdet, ydet, Ψdet; algorithm = :relaxation, use_anderson = true, m = 3, autodiff = autodiff_method, verbose = :none)
    end

    println("Homotopy method")
    @btime begin # called the "continuation" method in the original paper, but is called homotopy in the original code
        solve!(m, zdet, ydet, Ψdet; algorithm = :homotopy, autodiff = autodiff_method, verbose = :none)
    end
end
