# This script actually solves the WachterDisasterRisk model with a risk-adjusted linearization
# and times the methods, if desired
using BenchmarkTools, RiskAdjustedLinearizations, Test, JLD2
include("crw.jl")

# Settings: what do you want to do?
time_methods        = false
numerical_algorithm = :relaxation
autodiff            = false

# Set up
autodiff_method = autodiff ? :forward : :central
m_crw = CoeurdacierReyWinant()
m = crw(m_crw)
z0 = copy(m.z)
y0 = copy(m.y)
Ψ0 = copy(m.Ψ)

detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../test/reference/crw_dss.jld2"), "r")
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../test/reference/crw_sss.jld2"), "r")

m.z .= vec(detout["z_dss"])
m.y .= vec(detout["y_dss"])
m.Ψ .= detout["Psi_dss"]
# Solve!
# solve!(m, m.z, m.y, m.Ψ; algorithm = :homotopy, verbose = :high) # homotopy works!
# solve!(m, m.z, m.y, m.Ψ; algorithm = :relaxation, verbose = :high, ftol = 8e-4,
#        damping = .1) # bug!, need to double check the error or output when calling g0
# solve!(m; algorithm = numerical_algorithm, autodiff = autodiff_method)

#=
@test isapprox(sssout["z"], m.z, atol=1e-4)
@test isapprox(sssout["y"], m.y, atol=1e-4)
@test isapprox(sssout["Psi"], m.Ψ, atol=1e-4)
=#
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
