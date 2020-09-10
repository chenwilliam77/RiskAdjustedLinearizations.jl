# TODO: add unit test for solve_steadystate! specifically
using RiskAdjustedLinearizations, Test, JLD2
include(joinpath(dirname(@__FILE__), "../../examples/wachter_disaster_risk/wachter.jl"))

# Load in guesses and true solutions
detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/det_ss_output.jld2"), "r")

# Set up RiskAdjustedLinearization
m = WachterDisasterRisk()
ral = inplace_wachter_disaster_risk(m) # has guesses for z and y already
z = copy(ral.z)
y = copy(ral.y)
Ψ = copy(ral.Ψ)

# Solve!
@test_throws AssertionError solve!(ral, ral.z, ral.y, ral.Ψ; method = :deterministic, verbose = :high, autodiff = :central) # first w/finite diff nlsolve Jacobian
@info "The following series of print statements are expected."
solve!(ral, ral.z, ral.y; method = :deterministic, verbose = :high, autodiff = :central) # first w/finite diff nlsolve Jacobian
@test ral.z ≈ detout["z"]
@test ral.y ≈ detout["y"]
RiskAdjustedLinearizations.deterministic_steadystate!(ral, vcat(ral.z, ral.y); method = :deterministic,
                                                      verbose = :none, autodiff = :central) # first w/finite diff nlsolve Jacobian
@test ral.z ≈ detout["z"]
@test ral.y ≈ detout["y"]


update!(ral, z, y, Ψ) # now autodiff Jacobian
@test_broken solve!(ral, ral.z, ral.y; method = :deterministic, verbose = :high, autodiff = :forward) # now autodiff nlsolve Jacobian
@test_broken RiskAdjustedLinearizations.deterministic_steadystate!(ral, vcat(ral.z, ral.y); method = :deterministic,
                                                                   verbose = :none, autodiff = :forward)
#=@test ral.z ≈ sssout["z"]
@test ral.y ≈ sssout["y"] atol=5e-7
@test ral.Ψ ≈ sssout["Psi"]=#
