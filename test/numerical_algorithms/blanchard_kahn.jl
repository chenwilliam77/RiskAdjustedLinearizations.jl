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
update!(ral, z, y, Ψ)
@info "The following messages about Blanchard-Kahn conditions are expected."
@test RiskAdjustedLinearizations.blanchard_kahn(ral)
@test RiskAdjustedLinearizations.blanchard_kahn(ral; verbose = :low)
@test RiskAdjustedLinearizations.blanchard_kahn(ral; verbose = :none)

# Get deterministic steady state
detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/det_ss_output.jld2"), "r")
z = vec(detout["z"])
y = vec(detout["y"])

update!(ral, z, y, zeros(length(y), length(z)))
RiskAdjustedLinearizations.compute_Ψ(ral; zero_entropy_jacobian = true)
@test RiskAdjustedLinearizations.blanchard_kahn(ral; deterministic = true, verbose = :low)
@test RiskAdjustedLinearizations.blanchard_kahn(ral; deterministic = true, verbose = :high)

nothing
