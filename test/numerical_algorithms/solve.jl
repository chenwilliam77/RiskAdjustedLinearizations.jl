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
zguess = 1.01 .* copy(ral.z)
yguess = 1.01 .* copy(ral.y)

# Solve!
@info "The following series of print statements are expected."

# relaxation w/finite diff Jacobian
solve!(ral, zguess, yguess; verbose = :high, autodiff = :central, ftol = 1e-8) # first w/ calculating the deterministic steady state
@test ral.z ≈ sssout["z"] atol=1e-5                                            # and then proceeding to stochastic steady state
@test ral.y ≈ sssout["y"] atol=1e-5
@test ral.Ψ ≈ sssout["Psi"] atol=1e-5

update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ)
solve!(ral; verbose = :none, autodiff = :central, ftol = 1e-8) # Now just go straight to solving stochastic steady state
@test ral.z ≈ sssout["z"] atol=1e-5
@test ral.y ≈ sssout["y"] atol=1e-5
@test ral.Ψ ≈ sssout["Psi"] atol=1e-5

solve!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ;
       verbose = :none, autodiff = :central, ftol = 1e-8) # Now just go straight to solving stochastic steady state
@test ral.z ≈ sssout["z"] atol=1e-5
@test ral.y ≈ sssout["y"] atol=1e-5
@test ral.Ψ ≈ sssout["Psi"] atol=1e-5

# homotopy w/finite diff Jacobian
for i in 1:10
    try
        solve!(ral, zguess, yguess; algorithm = :homotopy, step = .5,
               verbose = :high, autodiff = :central, ftol = 1e-8) # first w/ calculating the deterministic steady state
        break
    catch e
        zguess .= 1.01 * vec(sssout["z"])
        yguess .= 1.01 * vec(sssout["y"])
    end
    if i == 10
        solve!(ral, zguess, yguess; algorithm = :homotopy, step = .5,
               verbose = :high, autodiff = :central, ftol = 1e-8) # first w/ calculating the deterministic steady state
    end
end
@test ral.z ≈ sssout["z"] atol=1e-5                               # and then proceeding to stochastic steady state
@test ral.y ≈ sssout["y"] atol=1e-5
@test ral.Ψ ≈ sssout["Psi"] atol=1e-5

update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ)
for i in 1:2
    try
        solve!(ral; verbose = :none, algorithm = :homotopy, step = .5,
               autodiff = :central, ftol = 1e-8) # Now just go straight to solving stochastic steady state
        break
    catch e
        update!(ral, 1.01 * vec(sssout["z"]), 1.01 * vec(sssout["y"]), 1.01 * sssout["Psi"])
    end
    if i == 2
        solve!(ral; verbose = :none, algorithm = :homotopy, step = .5,
               autodiff = :central, ftol = 1e-8) # Now just go straight to solving stochastic steady state
    end
end
@test ral.z ≈ sssout["z"] atol=1e-5
@test ral.y ≈ sssout["y"] atol=1e-5
@test ral.Ψ ≈ sssout["Psi"] atol=1e-5

for i in 1:2
    try
        solve!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ;
               verbose = :none, algorithm = :homotopy, step = .5,
               autodiff = :central, ftol = 1e-8) # Now just go straight to solving stochastic steady state
        break
    catch e
        z .= vec(sssout["z"])
        y .= 1.01 * vec(sssout["y"])
        Ψ .= sssout["Psi"]
    end
    if i == 2
        solve!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ;
               verbose = :none, algorithm = :homotopy, step = .5,
               autodiff = :central, ftol = 1e-8) # Now just go straight to solving stochastic steady state
    end
end
@test ral.z ≈ sssout["z"] atol=1e-5
@test ral.y ≈ sssout["y"] atol=1e-5
@test ral.Ψ ≈ sssout["Psi"] atol=1e-5

# Now autodiff Jacobian
solve!(ral, zguess, yguess; verbose = :high, autodiff = :forward, ftol = 1e-8)
update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ)
solve!(ral; verbose = :high, autodiff = :forward, ftol = 1e-8)
solve!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ; verbose = :none, autodiff = :forward, ftol = 1e-8)

# currently can't autodiff w/homotopy b/c chunksize inference is not working
@test_broken solve!(ral, zguess, yguess; verbose = :high, autodiff = :forward, ftol = 1e-8, algorithm = :homotopy)
update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ)
@test_broken solve!(ral; verbose = :high, autodiff = :forward, ftol = 1e-8, algorithm = :homotopy)
@test_broken solve!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ; verbose = :none, autodiff = :forward, ftol = 1e-8, algorithm = :homotopy)

# relaxation w/Anderson
solve!(ral, zguess, yguess; verbose = :high, autodiff = :central,
       use_anderson = true, ftol = 1e-8) # first w/ calculating the deterministic steady state
@test ral.z ≈ sssout["z"] atol=1e-5      # and then proceeding to stochastic steady state
@test ral.y ≈ sssout["y"] atol=1e-5
@test ral.Ψ ≈ sssout["Psi"] atol=1e-5

update!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ)
solve!(ral; verbose = :none, autodiff = :central,
       use_anderson = true, ftol = 1e-8) # Now just go straight to solving stochastic steady state
@test ral.z ≈ sssout["z"] atol=1e-5
@test ral.y ≈ sssout["y"] atol=1e-5
@test ral.Ψ ≈ sssout["Psi"] atol=1e-5

solve!(ral, 1.01 .* z, 1.01 .* y, 1.01 .* Ψ;
       verbose = :none, autodiff = :central,
       use_anderson = true, ftol = 1e-8) # Now just go straight to solving stochastic steady state
@test ral.z ≈ sssout["z"] atol=1e-5
@test ral.y ≈ sssout["y"] atol=1e-5
@test ral.Ψ ≈ sssout["Psi"] atol=1e-5
