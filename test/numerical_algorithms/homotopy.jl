using RiskAdjustedLinearizations, Test, JLD2
include(joinpath(dirname(@__FILE__), "..", "..", "examples", "wachter_disaster_risk", "wachter.jl"))

# Load in guesses and true solutions
detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "reference", "det_ss_output.jld2"), "r")
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "reference", "homotopy_sss_output.jld2"), "r")
z = vec(detout["z"])
y = vec(detout["y"])
Ψ = zeros(length(y), length(z))

# Set up RiskAdjustedLinearization
m = WachterDisasterRisk()
ral = inplace_wachter_disaster_risk(m)
update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ)

# Solve!
@info "The following series of print statements are expected."
for i in 1:2
    try # To avoid issues arising sometimes where homotopy accidentally fails when it shouldn't
        RiskAdjustedLinearizations.homotopy!(ral, vcat(ral.z, ral.y, vec(ral.Ψ)); verbose = :high, autodiff = :central,
                                             step = .5, ftol = 1e-8) # first with finite diff NLsolve Jacobian
        break
    catch e
        update!(ral, .99 * vec(sssout["z"]), .99 * vec(sssout["y"]), .99 * sssout["Psi"])
    end
    if i == 2 # trigger error if there actually is one
        RiskAdjustedLinearizations.homotopy!(ral, vcat(ral.z, ral.y, vec(ral.Ψ)); verbose = :low, autodiff = :central,
                                             step = .5, ftol = 1e-8) # first with finite diff NLsolve Jacobian
    end
end
@test ral.z ≈ sssout["z"] atol=1e-6
@test ral.y ≈ sssout["y"] atol=1e-4
@test ral.Ψ ≈ sssout["Psi"] atol=5e-3

update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ) # now autodiff Jacobian
@test_broken RiskAdjustedLinearizations.homotopy!(ral, vcat(ral.z, ral.y, vec(ral.Ψ)); verbose = :low, autodiff = :forward,
                                                  step = .5, ftol = 1e-8) # currently can't autodiff b/c problem with chunk size selection
#=@test ral.z ≈ sssout["z"] atol=1e-6
@test ral.y ≈ sssout["y"] atol=1e-4
@test ral.Ψ ≈ sssout["Psi"] atol=5e-3=#

# Check sparse Jacobian methods work
sparsity, colorvec = compute_sparsity_pattern(ral, :homotopy; sparsity_detection = false)
jac_cache = preallocate_jac_cache(ral, :homotopy; sparsity_detection = false)
RiskAdjustedLinearizations.homotopy!(ral, 1.001 .* vcat(ral.z, ral.y, vec(ral.Ψ));
                                     verbose = :none, sparse_jacobian = true,
                                     sparsity = sparsity, colorvec = colorvec)
@test maximum(abs.(ral.z - sssout["z"])) < 1e-6
@test maximum(abs.(ral.y - sssout["y"])) < 1e-6
RiskAdjustedLinearizations.homotopy!(ral, 1.001 .* vcat(ral.z, ral.y, vec(ral.Ψ));
                                     verbose = :none, sparse_jacobian = true,
                                     jac_cache = jac_cache)
@test maximum(abs.(ral.z - sssout["z"])) < 1e-6
@test maximum(abs.(ral.y - sssout["y"])) < 1e-6
RiskAdjustedLinearizations.homotopy!(ral, 1.001 .* vcat(ral.z, ral.y, vec(ral.Ψ));
                                    verbose = :none, sparse_jacobian = true,
                                    sparsity_detection = false)
@test maximum(abs.(ral.z - sssout["z"])) < 1e-6
@test maximum(abs.(ral.y - sssout["y"])) < 1e-6

# Using SparsityDetection.jl fails
@test_broken RiskAdjustedLinearizations.homotopy!(ral, 1.001 .* vcat(ral.z, ral.y, vec(ral.Ψ));
                                    verbose = :none, sparse_jacobian = true,
                                    sparsity_detection = true)
