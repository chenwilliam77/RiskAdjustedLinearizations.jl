"""
```
function relaxation!(m::RiskAdjustedLinearization, xₙ₋₁::AbstractVector{S1}, Ψₙ₋₁::AbstractMatrix{S1};
                    tol::S2 = 1e-10, maxit::Int = 1000, damping::S2 = .5, pnorm::S3 = Inf,
                    ftol::S2 = 1e-8, autodiff::Symbol = :forward, schur_fnct::Function = schur!,
                    verbose::Symbol = :none, kwargs...) where {S1 <: Number, S2 <: Real, S3 <: Real}
```

solves for the coefficients ``(z, y, \\Psi)`` of a risk-adjusted linearization by the following relaxation algorithm:

1. Initialize guesses for ``(z, y, \\Psi)``
2. Do until convergence
    a) Solve for ``(z, y)`` using the expectational and state transition equations and fixing ``\\Psi``.
    b) Use a QZ decomposition to solve for ``\\Psi`` while fixing ``(z, y)``.
"""
function relaxation!(m::RiskAdjustedLinearization, xₙ₋₁::AbstractVector{S1}, Ψₙ₋₁::AbstractMatrix{S1};
                    tol::S2 = 1e-10, maxit::Int = 1000, damping::S2 = .5, pnorm::S3 = Inf,
                    ftol::S2 = 1e-8, autodiff::Symbol = :forward, schur_fnct::Function = schur!,
                    verbose::Symbol = :none, kwargs...) where {S1 <: Number, S2 <: Real, S3 <: Real}
    # Set up
    err   = 1.
    count = 0
    nl = nonlinear_system(m)
    li = linearized_system(m)

    # Some aliases/views will be useful
    zₙ₋₁  = @view xₙ₋₁[1:m.Nz]
    yₙ₋₁  = @view xₙ₋₁[(m.Nz + 1):end]
    zₙ    = m.z
    yₙ    = m.y
    Ψₙ    = m.Ψ
    𝒱ₙ₋₁  = nl.𝒱_sss
    J𝒱ₙ₋₁ = li.JV

    while (err > tol) && (count < maxit)

        # Calculate entropy terms 𝒱ₙ₋₁, J𝒱ₙ₋₁
        update!(nl, zₙ₋₁, yₙ₋₁, Ψₙ₋₁, li.Γ₅, li.Γ₆; select = Symbol[:𝒱]) # updates nl.𝒱_sss
        update!(li, zₙ₋₁, yₙ₋₁, Ψₙ₋₁, nl.μ_sss, nl.ξ_sss, nl.𝒱_sss; select = Symbol[:JV]) # updates li.JV

        # Solve state transition and expectational equations for (zₙ, yₙ), taking 𝒱ₙ₋₁ and Ψₙ₋₁ as given
        solve_steadystate!(m, xₙ₋₁, Ψₙ₋₁, 𝒱ₙ₋₁, ftol = ftol, autodiff = autodiff, kwargs...) # updates m.z and m.y

        # Update Γ₁, Γ₂, Γ₃, Γ₄, given (zₙ, yₙ)
        update!(li, zₙ, yₙ, Ψₙ₋₁, nl.μ_sss, nl.ξ_sss, nl.𝒱_sss; select = Symbol[:Γ₁, :Γ₂, :Γ₃, :Γ₄]) # updates li.Γᵢ

        # QZ decomposition to get Ψₙ, taking Γ₁, Γ₂, Γ₃, Γ₄, and J𝒱ₙ₋₁ as given
        Ψₙ .= compute_Ψ(li; schur_fnct = schur_fnct)

        # Update zₙ, yₙ, and Ψₙ; then calculate error for convergence check
        zₙ .= damping .* zₙ + (1 - damping) .* zₙ₋₁
        yₙ .= damping .* yₙ + (1 - damping) .* yₙ₋₁
        Ψₙ .= damping .* Ψₙ + (1 - damping) .* Ψₙ₋₁
        err = norm(vcat(zₙ - zₙ₋₁, yₙ - yₙ₋₁, vec(Ψₙ - Ψₙ₋₁)), pnorm)

        # Update zₙ₋₁, yₙ₋₁, and Ψₙ₋₁ (without reallocating them)
        zₙ₋₁ .= zₙ
        yₙ₋₁ .= yₙ
        Ψₙ₋₁ .= Ψₙ

        if verbose == :high
            println("Iteration $(count): error under norm=$(pnorm) is $(err)")
        end

        count += 1
    end

    if count == maxit
        throw(RALRelaxationError("Relaxation method to find the risk-adjusted linearization did not converge."))
    else
        if verbose == :low
            println("Convergence achieved after $(count) iterations! Error under norm=$(pnorm) is $(err).")
        elseif verbose == :high
            println("")
            println("Convergence achieved after $(count) iterations! Error under norm=$(pnorm) is $(err).")
        end
        update!(m)

        return m
    end
end

function solve_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1},
                            Ψ::AbstractMatrix{<: Number}, 𝒱::AbstractVector{<: Number};
                            ftol::S2 = 1e-8, autodiff::Symbol = :forward, kwargs...) where {S1 <: Real, S2 <: Real}

    # Set up system of equations
    _my_eqn = function _my_stochastic_equations(F, x)
        # Unpack
        z = @view x[1:m.Nz]
        y = @view x[(m.Nz + 1):end]

        # Update μ(z, y) and ξ(z, y)
        update!(m.nonlinear, z, y, Ψ, m.linearization.Γ₅, m.linearization.Γ₆; select = Symbol[:μ, :ξ])

        # Calculate residuals
        F[1:m.Nz] = m.nonlinear.μ_sss - z
        F[(m.Nz + 1):end] = m.nonlinear.ξ_sss + m.linearization.Γ₅ * z + m.linearization.Γ₆ * y + 𝒱
    end

    out = nlsolve(_my_eqn, x0, ftol = ftol, autodiff = autodiff, kwargs...)

    if out.f_converged
        m.z .= out.zero[1:m.Nz]
        m.y .= out.zero[(m.Nz + 1):end]
    else
        throw(RALRelaxationError())
    end
end

mutable struct RALRelaxationError <: Exception
    msg::String
end
RALRelaxationError() =
    RALRelaxationError("A solution for (z, y), given Ψ and 𝒱, to the state transition and expectational equations could not be found.")
Base.showerror(io::IO, ex::RALRelaxationError) = print(io, ex.msg)
