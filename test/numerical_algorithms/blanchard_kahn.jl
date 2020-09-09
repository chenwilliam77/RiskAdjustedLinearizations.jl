using RiskAdjustedLinearizations, Test, JLD2
include(joinpath(dirname(@__FILE__), "../../examples/wachter_disaster_risk/wachter.jl"))

# Get stochastic steady state
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/iterative_sss_output.jld2"), "r")
z = vec(sssout["z"])
y = vec(sssout["y"])
Ψ = sssout["Psi"]

# Create model and update it
m = WachterDisasterRisk()
ral = inplace_wachter_disaster_risk(m)
RiskAdjustedLinearizations.update!(ral, z, y, Ψ)
@info "The following message about Blanchard-Kahn conditions is expected."
@test RiskAdjustedLinearizations.blanchard_kahn(ral)
@test RiskAdjustedLinearizations.blanchard_kahn(ral; verbose = :none)

nothing
