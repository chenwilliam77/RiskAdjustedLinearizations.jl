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
- `sparse_jacobian::Bool = false`: if true, exploit sparsity in the Jacobian in calls to `nlsolve` using SparseDiffTools.jl.
    If `jac_cache` and `sparsity` are `nothing`, then `homotopy!` will attempt to determine the sparsity pattern.
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
function homotopy!(m::RiskAdjustedLinearization, xₙ₋₁::AbstractVector{S1};
                   step::Float64 = .1, pnorm::Float64 = Inf,
                   autodiff::Symbol = :central,
                   sparse_jacobian::Bool = false,
                   sparsity::Union{AbstractArray, Nothing} = nothing, colorvec = nothing,
                   jac_cache = nothing, sparsity_detection::Bool = true,
                   verbose::Symbol = :none,
                   kwargs...) where {S1 <: Number}
    # Set up
    nl = nonlinear_system(m)
    li = linearized_system(m)
    _my_eqn = if Λ_eltype(nl) <: RALF1 && Σ_eltype(nl) <: RALF1 # only difference in this block and the next block
        (F, x, q) -> _homotopy_equations1(F, x, m, q)                # is the number of args to retrieve 𝒱_sss and JV
    else
        (F, x, q) -> _homotopy_equations2(F, x, m, q)
    end

    qguesses = step:step:1.
    if qguesses[end] != 1.
        qguesses = vcat(qguesses, 1.)
    end
    for (i, q) in enumerate(qguesses)
        solve_steadystate!(m, getvecvalues(m), _my_eqn, q; autodiff = autodiff,
                           sparse_jacobian = sparse_jacobian,
                           sparsity = sparsity, colorvec = colorvec,
                           jac_cache = jac_cache,
                           sparsity_detection = sparsity_detection,
                           verbose = verbose, kwargs...)

        if verbose == :high
            println("Success at iteration $(i) of $(length(qguesses))")
        end
    end

    if verbose in [:low, :high]
        errvec = steady_state_errors(m)

        println("Homotopy succeeded!")
        println("Error under norm = $(pnorm) is $(norm(errvec, pnorm)).")
    end

    update!(m)

    return m
end

function solve_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1}, f::Function, q::Float64;
                            sparse_jacobian::Bool = false,
                            sparsity::Union{AbstractArray, Nothing} = nothing, colorvec = nothing,
                            jac_cache = nothing, sparsity_detection::Bool = true, autodiff::Symbol = :central,
                            verbose::Symbol = :none, kwargs...) where {S1 <: Real}

    if sparse_jacobian # Exploit sparsity?
        nlsolve_jacobian!, jac =
            construct_sparse_jacobian_function(m, (F, x) -> f(F, x, q), :homotopy, autodiff;
                                               sparsity = sparsity, colorvec = colorvec,
                                               jac_cache = jac_cache,
                                               sparsity_detection = sparsity_detection)
        out = nlsolve(OnceDifferentiable((F, x) -> f(F, x, q), nlsolve_jacobian!, x0, copy(x0), jac), x0; kwargs...)
    else
        # Need to declare chunk size to ensure no problems with reinterpreting the cache
        # Potential solution: https://discourse.julialang.org/t/issue-with-pdmp-and-forwardiff-differentialequation/17925/22
        # Essentially, it may require developing another cache.
        out = nlsolve(OnceDifferentiable((F, x) -> f(F, x, q), x0, copy(x0), autodiff,
                                         ForwardDiff.Chunk(ForwardDiff.pickchunksize(min(m.Nz, m.Ny)))), x0; kwargs...)
    end

    if out.f_converged
        N_zy = m.Nz + m.Ny
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

function _homotopy_equations1(F::AbstractArray, x::AbstractArray, m::RiskAdjustedLinearization, q::Number)
    # Unpack
    z = @view x[1:m.Nz]
    y = @view x[(m.Nz + 1):(m.Nz + m.Ny)]
    Ψ = @view x[(m.Nz + m.Ny + 1):end]
    Ψ = reshape(Ψ, m.Ny, m.Nz)

    # Given coefficients, update the model
    update!(m.nonlinear, z, y, Ψ)
    update!(m.linearization, z, y, Ψ)

    # Calculate residuals
    μ_sss              = get_tmp(m.nonlinear.μ.cache, z, y, (1, 1)) # select the first DiffCache b/c that one
    ξ_sss              = get_tmp(m.nonlinear.ξ.cache, z, y, (1, 1)) # corresponds to autodiffing both z and y
    𝒱_sss              = get_tmp(m.nonlinear.𝒱.cache, z, Ψ, (1, 1)) # This line is different than in _homotopy_equations2
    Γ₁                 = get_tmp(m.linearization.μz.cache, z, y, (1, 1))
    Γ₂                 = get_tmp(m.linearization.μy.cache, z, y, (1, 1))
    Γ₃                 = get_tmp(m.linearization.ξz.cache, z, y, (1, 1))
    Γ₄                 = get_tmp(m.linearization.ξy.cache, z, y, (1, 1))
    JV                 = get_tmp(m.linearization.J𝒱.cache, z, Ψ, (1, 1)) # This line is different than in _homotopy_equations2
    F[1:m.Nz]          = μ_sss - z
    F[(m.Nz + 1):(m.Nz + m.Ny)] = ξ_sss + m.linearization[:Γ₅] * z + m.linearization[:Γ₆] * y + q * 𝒱_sss
    F[((m.Nz + m.Ny) + 1):end]  = Γ₃ + Γ₄ * Ψ + (m.linearization[:Γ₅] + m.linearization[:Γ₆] * Ψ) * (Γ₁ + Γ₂ * Ψ) + q * JV
end

function _homotopy_equations2(F::AbstractArray, x::AbstractArray, m::RiskAdjustedLinearization, q::Number)

    # Unpack
    z = @view x[1:m.Nz]
    y = @view x[(m.Nz + 1):(m.Nz + m.Ny)]
    Ψ = @view x[(m.Nz + m.Ny + 1):end]
    Ψ = reshape(Ψ, m.Ny, m.Nz)

    # Given coefficients, update the model
    update!(m.nonlinear, z, y, Ψ)
    update!(m.linearization, z, y, Ψ)

    # Calculate residuals
    μ_sss              = get_tmp(m.nonlinear.μ.cache, z, y, (1, 1)) # select the first DiffCache b/c that one
    ξ_sss              = get_tmp(m.nonlinear.ξ.cache, z, y, (1, 1)) # corresponds to autodiffing both z and y
    𝒱_sss              = get_tmp(m.nonlinear.𝒱.cache, z, y, Ψ, z, (1, 1))
    Γ₁                 = get_tmp(m.linearization.μz.cache, z, y, (1, 1))
    Γ₂                 = get_tmp(m.linearization.μy.cache, z, y, (1, 1))
    Γ₃                 = get_tmp(m.linearization.ξz.cache, z, y, (1, 1))
    Γ₄                 = get_tmp(m.linearization.ξy.cache, z, y, (1, 1))
    JV                 = get_tmp(m.linearization.J𝒱.cache, z, y, Ψ, (1, 1))
    F[1:m.Nz]          = μ_sss - z
    F[(m.Nz + 1):(m.Nz + m.Ny)] = ξ_sss + m.linearization[:Γ₅] * z + m.linearization[:Γ₆] * y + q * 𝒱_sss
    F[((m.Nz + m.Ny) + 1):end]  = Γ₃ + Γ₄ * Ψ + (m.linearization[:Γ₅] + m.linearization[:Γ₆] * Ψ) * (Γ₁ + Γ₂ * Ψ) + q * JV
end

mutable struct RALHomotopyError <: Exception
    msg::String
end
Base.showerror(io::IO, ex::RALHomotopyError) = print(io, ex.msg)
