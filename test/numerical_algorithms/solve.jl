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
zguess = copy(ral.z)
yguess = copy(ral.y)

# Solve!
@info "The following series of print statements are expected."

# First w/finite diff Jacobian
solve!(ral, zguess, yguess; verbose = :high, autodiff = :central, ftol = 1e-8) # first w/ calculating the deterministic steady state
update!(ral, z, y, Ψ)                                             # and then proceeding to stochastic steady state
solve!(ral, z, y, Ψ; verbose = :none, autodiff = :central, ftol = 1e-8) # Now just go straight to solving stochastic steady state
@test ral.z ≈ sssout["z"]
@test ral.y ≈ sssout["y"] atol=5e-7
@test ral.Ψ ≈ sssout["Psi"]

solve!(ral, zguess, yguess; verbose = :high, algorithm = :homotopy, autodiff = :central, ftol = 1e-8) # first w/ calculating the deterministic steady state
update!(ral, z, y, Ψ)                                             # and then proceeding to stochastic steady state
solve!(ral, z, y, Ψ; verbose = :none, algorithm = :homotopy, autodiff = :central, ftol = 1e-8) # Now just go straight to solving stochastic steady state
@test ral.z ≈ sssout["z"]
@test ral.y ≈ sssout["y"] atol=5e-7
@test ral.Ψ ≈ sssout["Psi"]

# Now autodiff Jacobian
@test_broken solve!(ral, zguess, yguess; verbose = :high, autodiff = :forward, ftol = 1e-8)
update!(ral, z, y, Ψ)
@test_broken solve!(ral, z, y, Ψ; verbose = :none, autodiff = :forward, ftol = 1e-8) # currently can't autodiff b/c caching problem

@test_broken solve!(ral, zguess, yguess; verbose = :high, autodiff = :forward, ftol = 1e-8, algorithm = :homotopy)
update!(ral, z, y, Ψ)
@test_broken solve!(ral, z, y, Ψ; verbose = :none, autodiff = :forward, ftol = 1e-8, algorithm = :homotopy) # currently can't autodiff b/c caching problem

#=@test ral.z ≈ sssout["z"]
@test ral.y ≈ sssout["y"] atol=5e-7
@test ral.Ψ ≈ sssout["Psi"]=#
