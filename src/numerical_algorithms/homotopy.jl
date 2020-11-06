"""
```
homotopy!(m, xₙ₋₁; step = .1, pnorm = Inf, verbose = :none, kwargs...)
```

solves the system of equations characterizing a risk-adjusted linearization by a homotopy method with
embedding parameter ``q``, which steps from 0 to 1, with ``q = 1`` obtaining the true solution.

Currently, the only algorithm for choosing ``q`` is a simple uniform step search. Given a step size
``\\Delta``, we solve the homotopy starting from ``q = \\Delta`` and increase ``q`` by ``\\Delta``
until ``q`` reaches 1 or passes 1 (in which case, we force ``q = 1``).

### Types:
- `S1 <: Number`

### Inputs
- `m::RiskAdjustedLinearization`: object holding functions needed to calculate
    the risk-adjusted linearization
- `xₙ₋₁::AbstractVector{S1}`: initial guess for ``(z, y, \\Psi)``

### Keywords
- `step::Float64`: size of the uniform step from `step` to 1.
- `pnorm::Float64`: norm under which to evaluate the errors after homotopy succeeds.
- `verbose::Symbol`: verbosity of information printed out during solution.
    a) `:low` -> statement when homotopy continuation succeeds
    b) `:high` -> statement when homotopy continuation succeeds and for each successful iteration
"""
function homotopy!(m::RiskAdjustedLinearization, xₙ₋₁::AbstractVector{S1};
                   step::Float64 = .1, pnorm::Float64 = Inf,
                   verbose::Symbol = :none, autodiff::Symbol = :central,
                   kwargs...) where {S1 <: Number}
    # Set up
    nl = nonlinear_system(m)
    li = linearized_system(m)

    qguesses = step:step:1.
    if qguesses[end] != 1.
        qguesses = vcat(qguesses, 1.)
    end
    for (i, q) in enumerate(qguesses)
        solve_steadystate!(m, getvecvalues(m), q; autodiff = autodiff, kwargs...)

        if verbose == :high
            println("Success at iteration $(i) of $(length(qguesses))")
        end
    end

    if verbose in [:low, :high]
        errvec = vcat(m[:μ_sss] - m.z, m[:ξ_sss] + m[:Γ₅] * m.z + m[:Γ₆] * m.y + m[:𝒱_sss],
                      vec(m[:Γ₃] + m[:Γ₄] * m.Ψ + (m[:Γ₅] + m[:Γ₆] * m.Ψ) * (m[:Γ₁] + m[:Γ₂] * m.Ψ) + m[:JV]))

        println("Homotopy succeeded!")
        println("Error under norm = $(pnorm) is $(norm(errvec, pnorm)).")
    end

    update!(m)

    return m
end

function solve_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1}, q::Float64;
                            autodiff::Symbol = :central, kwargs...) where {S1 <: Real}

    # Set up system of equations
    N_zy = m.Nz + m.Ny
    nl = nonlinear_system(m)
    li = linearized_system(m)
    _my_eqn = if Λ_eltype(nl) <: RALF1 && Σ_eltype(nl) <: RALF1 # only difference in this block and the next block
        function _my_stochastic_equations1(F, x)                # is the number of args to retrieve 𝒱_sss and JV
            # Unpack
            z = @view x[1:m.Nz]
            y = @view x[(m.Nz + 1):N_zy]
            Ψ = @view x[(N_zy + 1):end]
            Ψ = reshape(Ψ, m.Ny, m.Nz)

            # Given coefficients, update the model
            update!(nl, z, y, Ψ)
            update!(li, z, y, Ψ)

            # Calculate residuals
            μ_sss              = get_tmp(nl.μ.cache, z, y, (1, 1)) # select the first DiffCache b/c that one corresponds to autodiffing both z and y
            ξ_sss              = get_tmp(nl.ξ.cache, z, y, (1, 1))
            𝒱_sss              = get_tmp(nl.𝒱.cache, z, Ψ, (1, 1))
            Γ₁                 = get_tmp(li.μz.cache, z, y, (1, 1))
            Γ₂                 = get_tmp(li.μy.cache, z, y, (1, 1))
            Γ₃                 = get_tmp(li.ξz.cache, z, y, (1, 1))
            Γ₄                 = get_tmp(li.ξy.cache, z, y, (1, 1))
            JV                 = get_tmp(li.J𝒱.cache, z, Ψ, (1, 1))
            F[1:m.Nz]          = μ_sss - z
            F[(m.Nz + 1):N_zy] = ξ_sss + li[:Γ₅] * z + li[:Γ₆] * y + q * 𝒱_sss
            F[(N_zy + 1):end]  = Γ₃ + Γ₄ * Ψ + (li[:Γ₅] + li[:Γ₆] * Ψ) * (Γ₁ + Γ₂ * Ψ) + q * JV
        end
    else
        function _my_stochastic_equations2(F, x)
            # Unpack
            z = @view x[1:m.Nz]
            y = @view x[(m.Nz + 1):N_zy]
            Ψ = @view x[(N_zy + 1):end]
            Ψ = reshape(Ψ, m.Ny, m.Nz)

            # Given coefficients, update the model
            update!(nl, z, y, Ψ)
            update!(li, z, y, Ψ)

            # Calculate residuals
            μ_sss              = get_tmp(nl.μ.cache, z, y, (1, 1)) # select the first DiffCache b/c that one corresponds to autodiffing both z and y
            ξ_sss              = get_tmp(nl.ξ.cache, z, y, (1, 1))
            𝒱_sss              = get_tmp(nl.𝒱.cache, z, y, Ψ, z, (1, 1))
            Γ₁                 = get_tmp(li.μz.cache, z, y, (1, 1))
            Γ₂                 = get_tmp(li.μy.cache, z, y, (1, 1))
            Γ₃                 = get_tmp(li.ξz.cache, z, y, (1, 1))
            Γ₄                 = get_tmp(li.ξy.cache, z, y, (1, 1))
            JV                 = get_tmp(li.J𝒱.cache, z, y, Ψ, (1, 1))
            F[1:m.Nz]          = μ_sss - z
            F[(m.Nz + 1):N_zy] = ξ_sss + li[:Γ₅] * z + li[:Γ₆] * y + q * 𝒱_sss
            F[(N_zy + 1):end]  = Γ₃ + Γ₄ * Ψ + (li[:Γ₅] + li[:Γ₆] * Ψ) * (Γ₁ + Γ₂ * Ψ) + q * JV
        end
    end

    # Need to declare chunk size to ensure no problems with reinterpreting the cache
    out = nlsolve(OnceDifferentiable(_my_eqn, x0, copy(x0), autodiff,
                                     ForwardDiff.Chunk(ForwardDiff.pickchunksize(min(m.Nz, m.Ny)))), x0; kwargs...)

    if out.f_converged
        m.z .= out.zero[1:m.Nz]
        m.y .= out.zero[(m.Nz + 1):N_zy]
        m.Ψ .= reshape(out.zero[(N_zy + 1):end], m.Ny, m.Nz)
    else
        if verbose == :high
            println(out)
        end
        throw(RALHomotopyError("A solution for (z, y, Ψ) to the state transition, expectational, " *
                               "and linearization equations could not be found when the embedding " *
                               "parameter q equals $(q)"))
    end
end

mutable struct RALHomotopyError <: Exception
    msg::String
end
Base.showerror(io::IO, ex::RALHomotopyError) = print(io, ex.msg)
