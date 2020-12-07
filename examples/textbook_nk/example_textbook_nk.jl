# This script actually solves the WachterDisasterRisk model with a risk-adjusted linearization
# and times the methods, if desired
using RiskAdjustedLinearizations, JLD2, Test
include("textbook_nk.jl")
out = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "textbook_nk_ss_output.jld2"), "r")

# Settings
autodiff              = false
algorithm             = :relaxation
test_price_dispersion = false # check if price dispersion in steady state is always bounded below by 1

# Set up
m_nk = TextbookNK() # create parameters
m = textbook_nk(m_nk) # instantiate risk-adjusted linearization
autodiff_method = autodiff ? :forward : :central

# Solve!
solve!(m; algorithm = :deterministic, autodiff = autodiff_method)
@test m.z ≈ out["z_det"]
@test m.y ≈ out["y_det"]
@test m.Ψ ≈ out["Psi_det"]
z_det = copy(m.z)
y_det = copy(m.y)
Ψ_det = copy(m.Ψ)

solve!(m; algorithm = algorithm, autodiff = autodiff_method)
@test m.z ≈ out["z"]
@test m.y ≈ out["y"]
@test m.Ψ ≈ out["Psi"]

if test_price_dispersion
    π̃_ss_vec = log.(range(1 - .005, stop = 1 + .005, length = 10)) # On annualized basis, range from -2% to 2% target inflation
    det_soln = Dict()
    sss_soln = Vector{RiskAdjustedLinearization}(undef, length(π̃_ss_vec))

    for (i, π̃_ss) in enumerate(π̃_ss_vec)
        m_nk = TextbookNK(; π̃_ss = π̃_ss)
        m = textbook_nk(m_nk)
        solve!(m; algorithm = :deterministic, verbose = :none)
        det_soln[i] = Dict()
        det_soln[i][:z] = copy(m.z)
        det_soln[i][:y] = copy(m.y)
        det_soln[i][:Ψ] = copy(m.Ψ)
        solve!(m; algorithm = algorithm, verbose = :none)
        sss_soln[i] = m
    end

    @test all(det_v .> 1.)
    @test all(sss_v .> 1.)
end

nothing
