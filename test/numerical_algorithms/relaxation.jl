using RiskAdjustedLinearizations, Test, JLD2
include(joinpath(dirname(@__FILE__), "..", "..", "examples", "wachter_disaster_risk", "wachter.jl"))

# Load in guesses and true solutions
detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "reference", "det_ss_output.jld2"), "r")
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "reference", "iterative_sss_output.jld2"), "r")
z = vec(detout["z"])
y = vec(detout["y"])
Ψ = zeros(length(y), length(z))

# Set up RiskAdjustedLinearization
m = WachterDisasterRisk()
ral = inplace_wachter_disaster_risk(m)
update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ)

# Solve!
@info "The following series of print statements are expected."
RiskAdjustedLinearizations.relaxation!(ral, vcat(ral.z, ral.y), ral.Ψ; verbose = :low, autodiff = :central,
                                       tol = 1e-10, max_iters = 1000, damping = .5, pnorm = Inf, ftol = 1e-8) # first with finite diff NLsolve Jacobian
@test ral.z ≈ sssout["z"] atol=1e-6
@test ral.y ≈ sssout["y"] atol=1e-6
@test ral.Ψ ≈ sssout["Psi"] atol=1e-6

update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ) # now autodiff Jacobian
RiskAdjustedLinearizations.relaxation!(ral, vcat(ral.z, ral.y), ral.Ψ; verbose = :low, autodiff = :forward,
                                       tol = 1e-10, max_iters = 1000, damping = .5, pnorm = Inf, ftol = 1e-8)
@test ral.z ≈ sssout["z"] atol=1e-6
@test ral.y ≈ sssout["y"] atol=1e-6
@test ral.Ψ ≈ sssout["Psi"] atol=1e-6

update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ) # now use Anderson acceleration
RiskAdjustedLinearizations.relaxation!(ral, vcat(ral.z, ral.y), ral.Ψ; verbose = :low, autodiff = :central, use_anderson = true,
                                       tol = 1e-10, max_iters = 1000, damping = .5, pnorm = Inf, ftol = 1e-8)
@test ral.z ≈ sssout["z"] atol=1e-6
@test ral.y ≈ sssout["y"] atol=1e-6
@test ral.Ψ ≈ sssout["Psi"] atol=1e-6

# Check sparse Jacobian methods work
sparsity, colorvec = compute_sparsity_pattern(ral, :relaxation; sparsity_detection = false)
jac_cache = preallocate_jac_cache(ral, :relaxation; sparsity_detection = false)
RiskAdjustedLinearizations.relaxation!(ral, 1.001 .* vcat(ral.z, ral.y), ral.Ψ;
                                     verbose = :none, sparse_jacobian = true,
                                     sparsity = sparsity, colorvec = colorvec)
@test maximum(abs.(ral.z - sssout["z"])) < 1e-6
@test maximum(abs.(ral.y - sssout["y"])) < 1e-6
RiskAdjustedLinearizations.relaxation!(ral, 1.001 .* vcat(ral.z, ral.y), ral.Ψ;
                                     verbose = :none, sparse_jacobian = true,
                                     autodiff = :forward, sparsity = sparsity,
                                     colorvec = colorvec)
@test maximum(abs.(ral.z - sssout["z"])) < 1e-6
@test maximum(abs.(ral.y - sssout["y"])) < 1e-6
RiskAdjustedLinearizations.relaxation!(ral, 1.001 .* vcat(ral.z, ral.y), ral.Ψ;
                                     verbose = :none, sparse_jacobian = true,
                                     jac_cache = jac_cache)
@test maximum(abs.(ral.z - sssout["z"])) < 1e-6
@test maximum(abs.(ral.y - sssout["y"])) < 1e-6
RiskAdjustedLinearizations.relaxation!(ral, 1.001 .* vcat(ral.z, ral.y), ral.Ψ;
                                       verbose = :none, sparse_jacobian = true,
                                       sparsity_detection = false)
@test maximum(abs.(ral.z - sssout["z"])) < 1e-6
@test maximum(abs.(ral.y - sssout["y"])) < 1e-6

# Using SparsityDetection.jl fails
@test_broken RiskAdjustedLinearizations.relaxation!(ral, 1.001 .* vcat(ral.z, ral.y), ral.Ψ;
                                                    verbose = :none, sparse_jacobian = true,
                                                    sparsity_detection = true)
