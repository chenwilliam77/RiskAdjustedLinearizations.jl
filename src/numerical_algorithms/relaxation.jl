"""
```
relaxation!(ral, xₙ₋₁, Ψₙ₋₁; tol = 1e-10, max_iters = 1000, damping = .5, pnorm = Inf,
            schur_fnct = schur!, autodiff = :central, use_anderson = false, m = 5,
            verbose = :none, kwargs...)
```

solves for the coefficients ``(z, y, \\Psi)`` of a risk-adjusted linearization by the following relaxation algorithm:

1. Initialize guesses for ``(z, y, \\Psi)``
2. Do until convergence

    a) Solve for ``(z, y)`` using the expectational and state transition equations and fixing ``\\Psi``.

    b) Use a QZ decomposition to solve for ``\\Psi`` while fixing ``(z, y)``.

### Types:
- `S1 <: Number`
- `S2 <: Real`
- `S3 <: Real`

### Inputs
- `m::RiskAdjustedLinearization`: object holding functions needed to calculate
    the risk-adjusted linearization
- `xₙ₋₁::AbstractVector{S1}`: initial guess for ``(z, y)``
- `Ψₙ₋₁::AbstractVector{S1}`: initial guess for ``\\Psi``

### Keywords
- `tol::S2`: convergence tolerance of residual norm for relaxation algorithm
- `max_iters::Int`: maximumm number of iterations
- `damping::S2`: guesses are updated as the weighted average
    `xₙ = damping * proposal + (1 - damping) * xₙ₋₁`.
- `pnorm::S3`: norm for residual tolerance
- `schur_fnct::Function`: function for calculating the Schur factorization during QZ decomposition
- `autodiff::Symbol`: specifies whether to use autoamtic differentiation in `nlsolve`
    (and is the same keyword as the `autodiff` keyword for `nlsolve`)
- `use_anderson::Bool`: set to true to apply Anderson acceleration to the
    fixed point iteration of the relaxation algorithm
- `m::Int`: `m` coefficient if using Anderson acceleration
- `verbose::Symbol`: verbosity of information printed out during solution.
    a) `:low` -> statement when homotopy continuation succeeds
    b) `:high` -> statement when homotopy continuation succeeds and for each successful iteration
"""
function relaxation!(ral::RiskAdjustedLinearization, xₙ₋₁::AbstractVector{S1}, Ψₙ₋₁::AbstractMatrix{S1};
                     tol::S2 = 1e-10, max_iters::Int = 1000, damping::S2 = .5, pnorm::S3 = Inf,
                     schur_fnct::Function = schur!, autodiff::Symbol = :central,
                     use_anderson::Bool = false, m::Int = 5, verbose::Symbol = :none,
                     kwargs...) where {S1 <: Number, S2 <: Real, S3 <: Real}
    # Set up
    err   = 1.
    count = 0
    nl  = nonlinear_system(ral)
    li  = linearized_system(ral)
    Nzy = ral.Nz + ral.Ny
    AA  = Matrix{Complex{S1}}(undef, Nzy, Nzy) # pre-allocate these matrices to calculate QZ decomp for Ψ
    BB  = similar(AA)

    if use_anderson
        # Some aliases/views will be useful
        zₙ    = ral.z
        yₙ    = ral.y
        Ψₙ    = ral.Ψ
        𝒱ₙ₋₁  = nl[:𝒱_sss]
        J𝒱ₙ₋₁ = li[:JV]

        _anderson_f = function _my_anderson(F::AbstractArray{T}, xₙ₋₁::AbstractVector{T}) where {T <: Number}
            zₙ₋₁  = @view xₙ₋₁[1:ral.Nz]
            yₙ₋₁  = @view xₙ₋₁[(ral.Nz + 1):Nzy]
            Ψₙ₋₁  = @view xₙ₋₁[(Nzy + 1):end]
            Ψₙ₋₁  = reshape(Ψₙ₋₁, ral.Ny, ral.Nz)

            # Calculate entropy terms 𝒱ₙ₋₁, J𝒱ₙ₋₁
            update!(nl, zₙ₋₁, yₙ₋₁, Ψₙ₋₁; select = Symbol[:𝒱]) # updates nl.𝒱_sss
            update!(li, zₙ₋₁, yₙ₋₁, Ψₙ₋₁; select = Symbol[:JV]) # updates li.JV

            # Solve state transition and expectational equations for (zₙ, yₙ), taking 𝒱ₙ₋₁ and Ψₙ₋₁ as given
            solve_steadystate!(ral, vcat(zₙ₋₁, yₙ₋₁), Ψₙ₋₁, 𝒱ₙ₋₁; autodiff = autodiff, # updates ral.z and ral.y
                               verbose = verbose, kwargs...)

            # Update Γ₁, Γ₂, Γ₃, Γ₄, given (zₙ, yₙ)
            update!(li, zₙ, yₙ, Ψₙ₋₁; select = Symbol[:Γ₁, :Γ₂, :Γ₃, :Γ₄]) # updates li.Γᵢ

            # QZ decomposition to get Ψₙ, taking Γ₁, Γ₂, Γ₃, Γ₄, and J𝒱ₙ₋₁ as given
            Ψₙ .= compute_Ψ!(AA, BB, li; schur_fnct = schur_fnct)

            # Update zₙ, yₙ, and Ψₙ; then calculate error for convergence check
            zₙ .= damping .* zₙ + (1 - damping) .* zₙ₋₁
            yₙ .= damping .* yₙ + (1 - damping) .* yₙ₋₁
            Ψₙ .= damping .* Ψₙ + (1 - damping) .* Ψₙ₋₁
            err = norm(vcat(zₙ - zₙ₋₁, yₙ - yₙ₋₁, vec(Ψₙ - Ψₙ₋₁)), pnorm)

            # Calculate residual
            F[1:ral.Nz] = zₙ - zₙ₋₁
            F[(ral.Nz + 1):Nzy] = yₙ - yₙ₋₁
            F[(Nzy + 1):end] = vec(Ψₙ - Ψₙ₋₁)

            return F
        end

        out   = nlsolve(_anderson_f, vcat(xₙ₋₁, vec(Ψₙ₋₁)); m = m, ftol = tol, iterations = max_iters)
        count = out.iterations
        if out.f_converged
            update!(ral, out.zero[1:ral.Nz], out.zero[(ral.Nz + 1):Nzy],
                    reshape(out.zero[(Nzy + 1):end], ral.Ny, ral.Nz); update_cache = false)
        end
    else
        # Some aliases/views will be useful
        zₙ₋₁  = @view xₙ₋₁[1:ral.Nz]
        yₙ₋₁  = @view xₙ₋₁[(ral.Nz + 1):end]
        zₙ    = ral.z
        yₙ    = ral.y
        Ψₙ    = ral.Ψ
        𝒱ₙ₋₁  = nl[:𝒱_sss]
        J𝒱ₙ₋₁ = li[:JV]

        while (err > tol) && (count < max_iters)

            # Calculate entropy terms 𝒱ₙ₋₁, J𝒱ₙ₋₁
            update!(nl, zₙ₋₁, yₙ₋₁, Ψₙ₋₁; select = Symbol[:𝒱]) # updates nl.𝒱_sss
            update!(li, zₙ₋₁, yₙ₋₁, Ψₙ₋₁; select = Symbol[:JV]) # updates li.JV

            # Solve state transition and expectational equations for (zₙ, yₙ), taking 𝒱ₙ₋₁ and Ψₙ₋₁ as given
            solve_steadystate!(ral, xₙ₋₁, Ψₙ₋₁, 𝒱ₙ₋₁; autodiff = autodiff, # updates ral.z and ral.y
                               verbose = verbose, kwargs...)

            # Update Γ₁, Γ₂, Γ₃, Γ₄, given (zₙ, yₙ)
            update!(li, zₙ, yₙ, Ψₙ₋₁; select = Symbol[:Γ₁, :Γ₂, :Γ₃, :Γ₄]) # updates li.Γᵢ

            # QZ decomposition to get Ψₙ, taking Γ₁, Γ₂, Γ₃, Γ₄, and J𝒱ₙ₋₁ as given
            Ψₙ .= compute_Ψ!(AA, BB, li; schur_fnct = schur_fnct)

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
    end

    if count == max_iters
        throw(RALRelaxationError("Relaxation method to find the risk-adjusted linearization did not converge."))
    else
        update!(ral)

        if verbose == :low
            errvec = vcat(ral[:μ_sss] - ral.z, ral[:ξ_sss] + ral[:Γ₅] * ral.z + ral[:Γ₆] * ral.y + ral[:𝒱_sss],
                          vec(ral[:Γ₃] + ral[:Γ₄] * ral.Ψ + (ral[:Γ₅] + ral[:Γ₆] * ral.Ψ) * (ral[:Γ₁] + ral[:Γ₂] * ral.Ψ) + ral[:JV]))
            println("Convergence achieved after $(count) iterations! Error under norm = $(pnorm) is " *
                    "$(norm(errvec, pnorm)).")
        elseif verbose == :high
            errvec = vcat(ral[:μ_sss] - ral.z, ral[:ξ_sss] + ral[:Γ₅] * ral.z + ral[:Γ₆] * ral.y + ral[:𝒱_sss],
                          vec(ral[:Γ₃] + ral[:Γ₄] * ral.Ψ + (ral[:Γ₅] + ral[:Γ₆] * ral.Ψ) * (ral[:Γ₁] + ral[:Γ₂] * ral.Ψ) + ral[:JV]))
            println("")
            println("Convergence achieved after $(count) iterations! Error under norm = $(pnorm) is " *
                    "$(norm(errvec, pnorm)).")
        end

        return ral
    end
end

function solve_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1},
                            Ψ::AbstractMatrix{<: Number}, 𝒱::AbstractVector{<: Number};
                            autodiff::Symbol = :central, verbose::Symbol = :none,
                            kwargs...) where {S1 <: Real, S2 <: Real}

    # Set up system of equations
    nl = nonlinear_system(m)
    li = linearized_system(m)
    _my_eqn = function _my_stochastic_equations(F, x)
        # Unpack
        z = @view x[1:m.Nz]
        y = @view x[(m.Nz + 1):end]

        # Update μ(z, y) and ξ(z, y)
        update!(nl, z, y, Ψ; select = Symbol[:μ, :ξ])

        # Calculate residuals
        μ_sss             = get_tmp(nl.μ.cache, z, y, (1, 1)) # select the first DiffCache b/c that one corresponds to autodiffing both z and y
        ξ_sss             = get_tmp(nl.ξ.cache, z, y, (1, 1))
        F[1:m.Nz]         = μ_sss - z
        F[(m.Nz + 1):end] = ξ_sss + li[:Γ₅] * z + li[:Γ₆] * y + 𝒱
    end

    out = nlsolve(OnceDifferentiable(_my_eqn, x0, copy(x0), autodiff,
                                     ForwardDiff.Chunk(ForwardDiff.pickchunksize(min(m.Nz, m.Ny)))), x0; kwargs...)

    if out.f_converged
        m.z .= out.zero[1:m.Nz]
        m.y .= out.zero[(m.Nz + 1):end]
    else
        if verbose == :high
            println(out)
        end
        throw(RALRelaxationError())
    end
end

mutable struct RALRelaxationError <: Exception
    msg::String
end
RALRelaxationError() =
    RALRelaxationError("A solution for (z, y), given Ψ and 𝒱, to the state transition and expectational equations could not be found.")
Base.showerror(io::IO, ex::RALRelaxationError) = print(io, ex.msg)
