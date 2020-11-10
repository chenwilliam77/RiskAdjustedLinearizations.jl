using RiskAdjustedLinearizations, Test, JLD2
include(joinpath(dirname(@__FILE__), "..", "..", "examples", "wachter_disaster_risk", "wachter.jl"))

# Load in guesses and true solutions
detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "reference", "det_ss_output.jld2"), "r")

# Set up RiskAdjustedLinearization
m = WachterDisasterRisk()
ral = inplace_wachter_disaster_risk(m) # has guesses for z and y already
z = copy(ral.z)
y = copy(ral.y)
Ψ = copy(ral.Ψ)

# Solve!
@test_throws AssertionError solve!(ral, ral.z, ral.y, ral.Ψ; algorithm = :deterministic, verbose = :high, autodiff = :central) # first w/finite diff nlsolve Jacobian
@info "The following series of print statements are expected."
ral.z .= z .* 1.001
ral.y .= y .* 1.001
solve!(ral, ral.z, ral.y; algorithm = :deterministic, verbose = :high, autodiff = :central) # first w/finite diff nlsolve Jacobian
@test maximum(abs.(ral.z - detout["z"])) < 1e-6
@test maximum(abs.(ral.y - detout["y"])) < 1e-6
ral.z .= z .* 1.001
ral.y .= y .* 1.001
RiskAdjustedLinearizations.deterministic_steadystate!(ral, vcat(ral.z, ral.y);
                                                      verbose = :none, autodiff = :central) # first w/finite diff nlsolve Jacobian
@test maximum(abs.(ral.z - detout["z"])) < 1e-6
@test maximum(abs.(ral.y - detout["y"])) < 1e-6

update!(ral, z, y, Ψ) # now autodiff Jacobian
ral.z .= vec(detout["z"]) * 1.001
ral.y .= vec(detout["y"]) * 1.001
solve!(ral, ral.z, ral.y; algorithm = :deterministic, verbose = :high, autodiff = :forward) # now autodiff nlsolve Jacobian
@test maximum(abs.(ral.z - detout["z"])) < 1e-6
@test maximum(abs.(ral.y - detout["y"])) < 1e-6
ral.z .= vec(detout["z"]) * 1.001
ral.y .= vec(detout["y"]) * 1.001
RiskAdjustedLinearizations.deterministic_steadystate!(ral, vcat(ral.z, ral.y);
                                                      verbose = :none, autodiff = :forward)
@test maximum(abs.(ral.z - detout["z"])) < 1e-6
@test maximum(abs.(ral.y - detout["y"])) < 1e-6
