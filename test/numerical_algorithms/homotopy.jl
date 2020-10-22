using RiskAdjustedLinearizations, Test, JLD2
include(joinpath(dirname(@__FILE__), "../../examples/wachter_disaster_risk/wachter.jl"))

# Load in guesses and true solutions
detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/det_ss_output.jld2"), "r")
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/homotopy_sss_output.jld2"), "r")
z = vec(detout["z"])
y = vec(detout["y"])
Ψ = zeros(length(y), length(z))

# Set up RiskAdjustedLinearization
m = WachterDisasterRisk()
ral = inplace_wachter_disaster_risk(m)
update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ)

# Solve!
@info "The following series of print statements are expected."
RiskAdjustedLinearizations.homotopy!(ral, vcat(ral.z, ral.y, vec(ral.Ψ)); verbose = :low, autodiff = :central,
                                     step = .1, ftol = 1e-8) # first with finite diff NLsolve Jacobian
@test ral.z ≈ sssout["z"] atol=1e-6
@test ral.y ≈ sssout["y"] atol=1e-4
@test ral.Ψ ≈ sssout["Psi"] atol=5e-3

update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ) # now autodiff Jacobian
@test_broken RiskAdjustedLinearizations.homotopy!(ral, vcat(ral.z, ral.y, vec(ral.Ψ)); verbose = :low, autodiff = :forward,
                                                  step = .1, ftol = 1e-8) # currently can't autodiff b/c problem with chunk size selection
#=@test ral.z ≈ sssout["z"] atol=1e-6
@test ral.y ≈ sssout["y"] atol=1e-4
@test ral.Ψ ≈ sssout["Psi"] atol=5e-3=#
