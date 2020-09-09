# Wrapper for calculating the RAL
# List the keywords for each method
function solve!(m::RiskAdjustedLinearization, z0::AbstractVector{S1}, y0::AbstractVector{S1};
                method::Symbol = :relaxation, ftol::S2 = 1e-8, autodiff::Symbol = :forward,
                verbose::Symbol = :high, kwargs...) where {S1 <: Number, S2 <: Real, S3 <: Real}

    @assert method in [:deterministic, :relaxation, :homotopy, :continuation]

    # Deterministic steady state
    deterministic_steadystate!(m, vcat(z0, y0); ftol = ftol, autodiff = autodiff, kwargs...)

    # Use deterministic steady state as guess for stochastic steady state?
    if method == :deterministic # If not, . . .
        # Zero the entropy and Jacobian terms
        m.nonlinear.𝒱_sss  .= 0.
        m.linearization.JV .= 0.

        # Calculate linearization
        update!(m)

        # Back out Ψ
        compute_Ψ(m; zero_entropy_jacobian = true)
    else
        solve!(m, m.z, m.y, zeros(m.Ny, m.Nz); method = method, ftol = ftol, autodiff = autodiff,
               verbose = verbose, kwargs...)
    end

    # Check Blanchard-Kahn
    blanchard_kahn(m; verbose = verbose)
end

function solve!(m::RiskAdjustedLinearization, z0::AbstractVector{S1}, y0::AbstractVector{S1}, Ψ0::AbstractMatrix{S1};
                method::Symbol = :relaxation, ftol::S2 = 1e-8, autodiff::Symbol = :forward,
                verbose::Symbol = :high, kwargs...) where {S1 <: Number, S2 <: Real, S3 <: Real}

    @assert method in [:relaxation, :homotopy, :continuation]

    # Stochastic steady state
    if method == :relaxation
        N_zy = m.Nz + m.Ny
        relaxation!(m, vcat(z0, y0), Ψ0; ftol = ftol, autodiff = autodiff,
                    verbose = verbose, kwargs...)
    elseif method in [:homotopy, :continuation]
        homotopy!(m, vcat(z0, y0, vec(Ψ0)); ftol = ftol, autodiff = autodiff,
                  verbose = verbose, kwargs...)
    end

    # Check Blanchard-Kahn
    blanchard_kahn(m; verbose = verbose)
end


"""
```
function deterministic_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1};
                                    ftol::S2 = 1e-8, autodiff::Symbol = :forward, kwargs...) where {S1 <: Real, S2 <: Real}
```

calculates the deterministic steady state.

### Inputs
- `x0`: initial guess, whose first `m.Nz` elements are `z` and whose remaining elements are `y`.
"""
function deterministic_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1};
                                    ftol::S2 = 1e-8, autodiff::Symbol = :forward, kwargs...) where {S1 <: Real, S2 <: Real}

    # Set up system of equations
    _my_eqn = function _my_deterministic_equations(F, x)
        # Unpack
        z = @view x[1:m.Nz]
        y = @view x[(m.Nz + 1):end]

        # Update μ(z, y) and ξ(z, y)
        update!(m.nonlinear, z, y, m.Ψ, m.linearization.Γ₅, m.linearization.Γ₆; select = Symbol[:μ, :ξ])

        # Calculate residuals
        F[1:m.Nz] = m.nonlinear.μ_sss  - z
        F[(m.Nz + 1):end] = m.nonlinear.ξ_sss + m.linearization.Γ₅ * z + m.linearization.Γ₆ * y
    end

    out = nlsolve(_my_eqn, x0, ftol = ftol, autodiff = autodiff, kwargs...)

    if out.f_converged
        m.z .= out.zero[1:m.Nz]
        m.y .= out.zero[(m.Nz + 1):end]
    else
        error("A deterministic steady state could not be found.")
    end
end
