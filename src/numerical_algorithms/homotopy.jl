# Homotopy or Continuation algorithm
# Implement an SEIR criterion for choosing q
function homotopy!(m::RiskAdjustedLinearization, xₙ₋₁::AbstractVector{S1};
                   step::Float64 = .1, ftol::S2 = 1e-8, autodiff::Symbol = :forward,
                   verbose::Symbol = :none, kwargs...) where {S1 <: Number, S2 <: Real, S3 <: Real}
    # Set up
    nl = nonlinear_system(m)
    li = linearized_system(m)

    qguesses = step:step:1.
    if qguesses[end] != 1.
        qguesses = vcat(qguesses, 1.)
    end
    for (i, q) in enumerate(qguesses)
        solve_steadystate!(m, vcat(m.z, m.y, vec(m.Ψ)), q;
                           ftol = ftol, autodiff = autodiff, kwargs...)

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

function solve_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1}, q::Float64;
                            ftol::S2 = 1e-8, autodiff::Symbol = :forward, kwargs...) where {S1 <: Real, S2 <: Real}

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

    out = nlsolve(_my_eqn, x0, autodiff = autodiff, ftol = ftol, kwargs...)

    if out.f_converged
        m.z .= out.zero[1:m.Nz]
        m.y .= out.zero[(m.Nz + 1):N_zy]
        m.Ψ .= reshape(out.zero[(N_zy + 1):end], m.Ny, m.Nz)
    else
        throw(RALHomotopyError("A solution for (z, y, Ψ) to the state transition, expectational, " *
                               "and linearization equations could not be found when the convergence-control " *
                               "parameter q equals $(q)"))
    end
end

mutable struct RALHomotopyError <: Exception
    msg::String
end
Base.showerror(io::IO, ex::RALHomotopyError) = print(io, ex.msg)
