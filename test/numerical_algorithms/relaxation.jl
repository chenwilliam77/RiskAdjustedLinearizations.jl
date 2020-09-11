using RiskAdjustedLinearizations, Test, JLD2
include(joinpath(dirname(@__FILE__), "../../examples/wachter_disaster_risk/wachter.jl"))

# Load in guesses and true solutions
detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/det_ss_output.jld2"), "r")
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/iterative_sss_output.jld2"), "r")
z = vec(detout["z"])
y = vec(detout["y"])
Ψ = zeros(length(y), length(z))

# Set up RiskAdjustedLinearization
m = WachterDisasterRisk()
ral = inplace_wachter_disaster_risk(m)
update!(ral, z, y, Ψ)

# Solve!
@info "The following series of print statements are expected."
RiskAdjustedLinearizations.relaxation!(ral, vcat(ral.z, ral.y), ral.Ψ; verbose = :low, autodiff = :central,
                                             tol = 1e-10, max_iters = 1000, damping = .5, pnorm = Inf, ftol = 1e-8) # first with finite diff NLsolve Jacobian
@test ral.z ≈ sssout["z"]
@test ral.y ≈ sssout["y"] atol=5e-7
@test ral.Ψ ≈ sssout["Psi"]

update!(ral, z, y, Ψ) # now autodiff Jacobian
@test_broken RiskAdjustedLinearizations.relaxation!(ral, vcat(ral.z, ral.y), ral.Ψ; verbose = :low, autodiff = :forward,
                                                    tol = 1e-10, max_iters = 1000, damping = .5, pnorm = Inf, ftol = 1e-8) # currently can't autodiff b/c caching problem
#=@test ral.z ≈ sssout["z"]
@test ral.y ≈ sssout["y"] atol=5e-7
@test ral.Ψ ≈ sssout["Psi"]=#
