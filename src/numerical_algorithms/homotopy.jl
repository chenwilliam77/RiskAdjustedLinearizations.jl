"""
```
homotopy!(m, xₙ₋₁; step = .1, verbose = :none, kwargs...)
```

solves the system of equations characterizing a risk-adjusted linearization by a homotopy method with
embedding parameter ``q``, which steps from 0 to 1, with ``q = 1`` obtaining the true solution.

Currently, the only algorithm for choosing ``q`` is a simple uniform step search. Given a step size
``\\Delta```, we solve the homotopy starting from ``q = \\Delta`` and increase ``q`` by ``\\Delta``
until ``q`` reaches 1 or passes 1 (in which case, we force ``q = 1``).

### Types:
- `S1 <: Number`

### Inputs
- `m::RiskAdjustedLinearization`: object holding functions needed to calculate
    the risk-adjusted linearization
- `xₙ₋₁::AbstractVector{S1}`: initial guess for ``(z, y, \\Psi)``

### Keywords
- `step::Float64`: size of the uniform step from `step` to 1.
- `verbose::Symbol`: verbosity of information printed out during solution.
    a) `:low` -> statement when homotopy continuation succeeds
    b) `:high` -> statement when homotopy continuation succeeds and for each successful iteration
"""
function homotopy!(m::RiskAdjustedLinearization, xₙ₋₁::AbstractVector{S1};
                   step::Float64 = .1, verbose::Symbol = :none, kwargs...) where {S1 <: Number}
    # Set up
    nl = nonlinear_system(m)
    li = linearized_system(m)

    qguesses = step:step:1.
    if qguesses[end] != 1.
        qguesses = vcat(qguesses, 1.)
    end
    for (i, q) in enumerate(qguesses)
        solve_steadystate!(m, vcat(m.z, m.y, vec(m.Ψ)), q; kwargs...)

        if verbose == :high
            println("Success at iteration $(i) of $(length(qguesses))")
        end
    end

    if verbose in [:low, :high]
        println("Homotopy succeeded!")
    end

    update!(m)

    return m
end

function solve_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1},
                            q::Float64; kwargs...) where {S1 <: Real}

    # Set up system of equations
    N_zy = m.Nz + m.Ny
    nl = nonlinear_system(m)
    li = linearized_system(m)
    _my_eqn = function _my_stochastic_equations(F, x)
        # Unpack
        z = @view x[1:m.Nz]
        y = @view x[(m.Nz + 1):N_zy]
        Ψ = @view x[(N_zy + 1):end]
        Ψ = reshape(Ψ, m.Ny, m.Nz)

        # Given coefficients, update the model
        update!(m, z, y, Ψ)

        # Calculate residuals
        F[1:m.Nz] = nl.μ_sss - z
        F[(m.Nz + 1):N_zy] = nl.ξ_sss + li.Γ₅ * z + li.Γ₆ * y + q * nl.𝒱_sss
        F[(N_zy + 1):end] = li.Γ₃ + li.Γ₄ * Ψ + (li.Γ₅ + li.Γ₆ * Ψ) * (li.Γ₁ + li.Γ₂ * Ψ) + q * li.JV
    end

    out = nlsolve(_my_eqn, x0, kwargs...)

    if out.f_converged
        m.z .= out.zero[1:m.Nz]
        m.y .= out.zero[(m.Nz + 1):N_zy]
        m.Ψ .= reshape(out.zero[(N_zy + 1):end], m.Ny, m.Nz)
    else
        throw(RALHomotopyError("A solution for (z, y, Ψ) to the state transition, expectational, " *
                               "and linearization equations could not be found when the embedding " *
                               "parameter q equals $(q)"))
    end
end

mutable struct RALHomotopyError <: Exception
    msg::String
end
Base.showerror(io::IO, ex::RALHomotopyError) = print(io, ex.msg)
