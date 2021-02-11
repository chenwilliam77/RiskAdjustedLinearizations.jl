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
- `sparse_jacobian::Bool = false`: if true, exploit sparsity in the Jacobian in calls to `nlsolve` using SparseDiffTools.jl.
    If `jac_cache` and `sparsity` are `nothing`, then `relaxation!` will attempt to determine the sparsity pattern.
- `sparsity::Union{AbstractArray, Nothing} = nothing`: sparsity pattern for the Jacobian in calls to `nlsolve`
- `colorvec = nothing`: matrix coloring vector for sparse Jacobian in calls to `nlsolve`
- `jac_cache = nothing`: pre-allocated Jacobian cache for calls to `nlsolve` during the numerical algorithms
- `sparsity_detection::Bool = false`: If true, use SparsityDetection.jl to detect sparsity pattern (only relevant if
    both `jac_cache` and `sparsity` are `nothing`). If false,  then the sparsity pattern is
    determined by using finite differences to calculate a Jacobian and assuming any zeros will always be zero.
    Currently, SparsityDetection.jl fails to work.
- `verbose::Symbol`: verbosity of information printed out during solution.
    a) `:low` -> statement when homotopy continuation succeeds
    b) `:high` -> statement when homotopy continuation succeeds and for each successful iteration
"""
function relaxation!(ral::RiskAdjustedLinearization, xₙ₋₁::AbstractVector{S1}, Ψₙ₋₁::AbstractMatrix{S1};
                     tol::S2 = 1e-10, max_iters::Int = 1000, damping::S2 = .5, pnorm::S3 = Inf,
                     schur_fnct::Function = schur!, autodiff::Symbol = :central,
                     use_anderson::Bool = false, m::Int = 5,
                     sparse_jacobian::Bool = false,  sparsity::Union{AbstractArray, Nothing} = nothing,
                     colorvec = nothing, jac_cache = nothing,
                     sparsity_detection::Bool = true, verbose::Symbol = :none,
                     kwargs...) where {S1 <: Number, S2 <: Real, S3 <: Real}
    # Set up
    err = 1.
    nl  = nonlinear_system(ral)
    li  = linearized_system(ral)
    Nzy = ral.Nz + ral.Ny
    AA  = Matrix{Complex{S1}}(undef, Nzy, Nzy) # pre-allocate these matrices to calculate QZ decomp for Ψ
    BB  = similar(AA)

    # Initialize system of equations
    _my_eqn = (F, x, Ψ, 𝒱) -> _relaxation_equations(F, x, ral, Ψ, 𝒱)

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
            solve_steadystate!(ral, vcat(zₙ₋₁, yₙ₋₁), _my_eqn, Ψₙ₋₁, 𝒱ₙ₋₁; autodiff = autodiff, # updates ral.z and ral.y
                               sparse_jacobian = sparse_jacobian,
                               sparsity = sparsity, colorvec = colorvec,
                               jac_cache = jac_cache,
                               sparsity_detection = sparsity_detection,
                               verbose = verbose, kwargs...)

            # Update Γ₁, Γ₂, Γ₃, Γ₄, given (zₙ, yₙ)
            update!(li, zₙ, yₙ, Ψₙ₋₁; select = Symbol[:Γ₁, :Γ₂, :Γ₃, :Γ₄]) # updates li.Γᵢ

            # QZ decomposition to get Ψₙ, taking Γ₁, Γ₂, Γ₃, Γ₄, and J𝒱ₙ₋₁ as given
            Ψₙ .= compute_Ψ!(AA, BB, li; schur_fnct = schur_fnct)

            # Update zₙ, yₙ, and Ψₙ; then calculate error for convergence check
            zₙ .= (1 - damping) .* zₙ + damping .* zₙ₋₁
            yₙ .= (1 - damping) .* yₙ + damping .* yₙ₋₁
            Ψₙ .= (1 - damping) .* Ψₙ + damping .* Ψₙ₋₁
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
        count = 1

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
            solve_steadystate!(ral, xₙ₋₁, _my_eqn, Ψₙ₋₁, 𝒱ₙ₋₁; autodiff = autodiff, # updates ral.z and ral.y
                               sparse_jacobian = sparse_jacobian,
                               sparsity = sparsity, colorvec = colorvec,
                               jac_cache = jac_cache,
                               sparsity_detection = sparsity_detection,
                               verbose = verbose, kwargs...)

            # Update Γ₁, Γ₂, Γ₃, Γ₄, given (zₙ, yₙ)
            update!(li, zₙ, yₙ, Ψₙ₋₁; select = Symbol[:Γ₁, :Γ₂, :Γ₃, :Γ₄]) # updates li.Γᵢ

            # QZ decomposition to get Ψₙ, taking Γ₁, Γ₂, Γ₃, Γ₄, and J𝒱ₙ₋₁ as given
            Ψₙ .= compute_Ψ!(AA, BB, li; schur_fnct = schur_fnct)

            # Update zₙ, yₙ, and Ψₙ; then calculate error for convergence check
            zₙ .= (1 - damping) .* zₙ + damping .* zₙ₋₁
            yₙ .= (1 - damping) .* yₙ + damping .* yₙ₋₁
            Ψₙ .= (1 - damping) .* Ψₙ + damping .* Ψₙ₋₁
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
            errvec = steady_state_errors(ral)
            println("Convergence achieved after $(count) iterations! Error under norm = $(pnorm) is " *
                    "$(norm(errvec, pnorm)).")
        elseif verbose == :high
            errvec = steady_state_errors(ral)
            println("")
            println("Convergence achieved after $(count) iterations! Error under norm = $(pnorm) is " *
                    "$(norm(errvec, pnorm)).")
        end

        return ral
    end
end

function solve_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1},
                            f::Function, Ψ::AbstractMatrix{<: Number}, 𝒱::AbstractVector{<: Number};
                            sparse_jacobian::Bool = false, sparsity::Union{AbstractArray, Nothing} = nothing,
                            colorvec = nothing, jac_cache = nothing,
                            sparsity_detection::Bool = true, autodiff::Symbol = :central,
                            verbose::Symbol = :none, kwargs...) where {S1 <: Real, S2 <: Real}

    # Exploit sparsity?
    if sparse_jacobian
        nlsolve_jacobian!, jac =
            construct_sparse_jacobian_function(m, (F, x) -> f(F, x, Ψ, 𝒱), :relaxation, autodiff;
                                               sparsity = sparsity, colorvec = colorvec,
                                               jac_cache = jac_cache, sparsity_detection = sparsity_detection)
        out = nlsolve(OnceDifferentiable((F, x) -> f(F, x, Ψ, 𝒱), nlsolve_jacobian!, x0, copy(x0), jac), x0; kwargs...)
    else
        out = nlsolve(OnceDifferentiable((F, x) -> f(F, x, Ψ, 𝒱), x0, copy(x0), autodiff,
                                         ForwardDiff.Chunk(ForwardDiff.pickchunksize(min(m.Nz, m.Ny)))), x0; kwargs...)
    end

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

function _relaxation_equations(F::AbstractArray, x::AbstractArray, m::RiskAdjustedLinearization,
                               Ψ::AbstractMatrix{<: Number}, 𝒱::AbstractVector{<: Number})
    # Unpack
    z = @view x[1:m.Nz]
    y = @view x[(m.Nz + 1):end]

    # Update μ(z, y) and ξ(z, y)
    update!(m.nonlinear, z, y, Ψ; select = Symbol[:μ, :ξ])

    # Calculate residuals
    μ_sss             = get_tmp(m.nonlinear.μ.cache, z, y, (1, 1)) # select the first DiffCache b/c that one
    ξ_sss             = get_tmp(m.nonlinear.ξ.cache, z, y, (1, 1)) # corresponds to autodiffing both z and y
    F[1:m.Nz]         = μ_sss - z
    F[(m.Nz + 1):end] = ξ_sss + m.linearization[:Γ₅] * z + m.linearization[:Γ₆] * y + 𝒱
end

mutable struct RALRelaxationError <: Exception
    msg::String
end
RALRelaxationError() =
    RALRelaxationError("A solution for (z, y), given Ψ and 𝒱, to the state transition and expectational equations could not be found.")
Base.showerror(io::IO, ex::RALRelaxationError) = print(io, ex.msg)
