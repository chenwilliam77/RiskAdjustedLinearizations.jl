"""
```
solve!(m; algorithm = :relaxation, autodiff = :central, verbose = :high, kwargs...)
solve!(m, z0, y0; kwargs...)
solve!(m, z0, y0, Ψ0; kwargs...)
```

computes the risk-adjusted linearization of the dynamic economic model
described by `m` and updates `m` with the solution,
e.g. the coefficients ``(z, y, \\Psi)``.

The three available `solve!` algorithms are slight variations on each other.

- Method 1: uses the `z`, `y`, and `Ψ` fields of `m` as initial guesses
    for ``(z, y, \\Psi)`` and proceeds with the numerical algorithm
    specified by `algorithm`

- Method 2: uses `z0` and `y0` as initial guesses for the deterministic
    steady state, which is then used as the initial guess for ``(z, Y, \\Psi)``
    for the numerical algorithm specified by `algorithm`.

- Method 3: uses `z0`, `y0`, and `Ψ0` as initial guesses for ``(z, Y, \\Psi)``
    and proceeds with the numerical algorithm specified by `algorithm`.

### Inputs
- `m::RiskAdjustedLinearization`: object holding functions needed to calculate
    the risk-adjusted linearization
- `z0::AbstractVector{S1}`: initial guess for ``z``
- `y0::AbstractVector{S1}`: initial guess for ``y``
- `Ψ0::AbstractVector{S1}`: initial guess for ``\\Psi``
- `S1 <: Real`

### Keywords
- `algorithm::Symbol = :relaxation`: which numerical algorithm to use? Can be one of `[:relaxation, :homotopy, :deterministic]`
- `autodiff::Symbol = :central`: use autodiff or not? This keyword is the same as in `nlsolve`
- `use_anderson::Bool = false`: use Anderson acceleration if the relaxation algorithm is applied. Defaults to `false`
- `step::Float64 = .1`: size of step from 0 to 1 if the homotopy algorithm is applied. Defaults to 0.1
- `sparse_jacobian::Bool = false`: if true, exploit sparsity in the Jacobian in calls to `nlsolve` using SparseDiffTools.jl.
    If `jac_cache` and `sparsity` are `nothing`, then `solve!` will attempt to determine the sparsity pattern.
- `sparsity::Union{AbstractArray, Nothing} = nothing`: sparsity pattern for the Jacobian in calls to `nlsolve`
- `colorvec = nothing`: matrix coloring vector for sparse Jacobian in calls to `nlsolve`
- `jac_cache = nothing`: pre-allocated Jacobian cache for calls to `nlsolve` during the numerical algorithms
- `sparsity_detection::Bool = false`: If true, use SparsityDetection.jl to detect sparsity pattern (only relevant if
    both `jac_cache` and `sparsity` are `nothing`). If false,  then the sparsity pattern is
    determined by using finite differences to calculate a Jacobian and assuming any zeros will always be zero.
    Currently, SparsityDetection.jl fails to work.

The solution algorithms all use `nlsolve` to calculate the solution to systems of nonlinear
equations. The user can pass in any of the keyword arguments for `nlsolve` to adjust
the settings of the nonlinear solver.

For the keywords relevant to specific methods, see the docstring for the underlying method being called.
Note these methods are not exported.

- `:relaxation` -> `relaxation!`
- `:homotopy` -> `homotopy!`
- `:deterministic` -> `deterministic_steadystate!`
"""
function solve!(m::RiskAdjustedLinearization; algorithm::Symbol = :relaxation,
                autodiff::Symbol = :central, use_anderson::Bool = false,
                step::Float64 = .1, sparse_jacobian::Bool = false,
                sparsity::Union{AbstractArray, Nothing} = nothing, colorvec = nothing,
                jac_cache = nothing, sparsity_detection::Bool = false,
                verbose::Symbol = :high, kwargs...)
    if algorithm == :deterministic
        solve!(m, m.z, m.y; algorithm = algorithm, autodiff = autodiff,
               sparse_jacobian = sparse_jacobian,
               sparsity = sparsity, colorvec = colorvec,
               jac_cache = jac_cache, sparsity_detection = sparsity_detection,
               verbose = verbose, kwargs...)
    else
        solve!(m, m.z, m.y, m.Ψ; algorithm = algorithm, autodiff = autodiff,
               use_anderson = use_anderson, step = step,
               sparse_jacobian = sparse_jacobian, sparsity = sparsity,
               colorvec = colorvec, jac_cache = jac_cache,
               sparsity_detection = sparsity_detection, verbose = verbose, kwargs...)
    end
end

function solve!(m::RiskAdjustedLinearization, z0::AbstractVector{S1}, y0::AbstractVector{S1};
                algorithm::Symbol = :relaxation, autodiff::Symbol = :central,
                use_anderson::Bool = false, step::Float64 = .1,
                sparse_jacobian::Bool = false,
                sparsity::Union{AbstractArray, Nothing} = nothing, colorvec = nothing,
                jac_cache = nothing, sparsity_detection::Bool = false,
                verbose::Symbol = :high, kwargs...) where {S1 <: Real}

    @assert algorithm in [:deterministic, :relaxation, :homotopy] "The algorithm must be :deterministic, :relaxation, or :homotopy"

    # Deterministic steady state
    deterministic_steadystate!(m, vcat(z0, y0); autodiff = autodiff,
                               sparse_jacobian = sparse_jacobian,
                               sparsity = sparsity, colorvec = colorvec,
                               jac_cache = jac_cache,
                               sparsity_detection = sparsity_detection,
                               verbose = verbose, kwargs...)

    # Calculate linearization
    nl = nonlinear_system(m)
    li = linearized_system(m)
    update!(nl, m.z, m.y, m.Ψ; select = Symbol[:μ, :ξ])
    update!(li, m.z, m.y, m.Ψ; select = Symbol[:Γ₁, :Γ₂, :Γ₃, :Γ₄])

    # Back out Ψ
    m.Ψ .= compute_Ψ(m; zero_entropy_jacobian = true)

    # Use deterministic steady state as guess for stochastic steady state?
    if algorithm == :deterministic
        # Zero the entropy and Jacobian terms so they are not undefined or something else
        m.nonlinear[:𝒱_sss]  .= 0.
        m.linearization[:JV] .= 0.

        # Check Blanchard-Kahn
        blanchard_kahn(m; deterministic = true, verbose = verbose)
    else
        solve!(m, m.z, m.y, m.Ψ; algorithm = algorithm,
               use_anderson = use_anderson, step = step,
               sparse_jacobian = sparse_jacobian,
               jac_cache = jac_cache, sparsity = sparsity,
               colorvec = colorvec, sparsity_detection = sparsity_detection,
               verbose = verbose, kwargs...)
    end

    m
end

function solve!(m::RiskAdjustedLinearization, z0::AbstractVector{S1}, y0::AbstractVector{S1}, Ψ0::AbstractMatrix{S1};
                algorithm::Symbol = :relaxation, autodiff::Symbol = :central,
                use_anderson::Bool = false, step::Float64 = .1,
                sparse_jacobian::Bool = false, sparsity::Union{AbstractArray, Nothing} = nothing,
                colorvec = nothing, jac_cache = nothing, sparsity_detection::Bool = false,
                verbose::Symbol = :high, kwargs...) where {S1 <: Number}

    @assert algorithm in [:relaxation, :homotopy] "The algorithm must be :relaxation or :homotopy because this function calculates the stochastic steady state"

    # Stochastic steady state
    if algorithm == :relaxation
        N_zy = m.Nz + m.Ny
        relaxation!(m, vcat(z0, y0), Ψ0; autodiff = autodiff,
                    use_anderson = use_anderson, sparse_jacobian = sparse_jacobian,
                    sparsity = sparsity, colorvec = colorvec,
                    jac_cache = jac_cache, sparsity_detection = sparsity_detection,
                    verbose = verbose, kwargs...)
    elseif algorithm == :homotopy
        homotopy!(m, vcat(z0, y0, vec(Ψ0)); autodiff = autodiff, step = step,
                  sparse_jacobian = sparse_jacobian,
                  sparsity = sparsity, colorvec = colorvec,
                  jac_cache = jac_cache, sparsity_detection = sparsity_detection,
                  verbose = verbose, kwargs...)
    end

    # Check Blanchard-Kahn
    blanchard_kahn(m; deterministic = false, verbose = verbose)

    m
end

"""
```
function deterministic_steadystate!(m, x0; verbose = :none, kwargs...)
```

calculates the deterministic steady state.

### Types:
- `S1 <: Number`
- `S2 <: Real`

### Inputs
- `m::RiskAdjustedLinearization`: object holding functions needed to calculate
    the risk-adjusted linearization
- `x0::AbstractVector{S1}`: initial guess for ``(z, y)``
"""
function deterministic_steadystate!(m::RiskAdjustedLinearization, x0::AbstractVector{S1};
                                    autodiff::Symbol = :central,
                                    sparse_jacobian::Bool = false,
                                    sparsity::Union{AbstractArray, Nothing} = nothing, colorvec = nothing,
                                    jac_cache = nothing, sparsity_detection::Bool = false,
                                    verbose::Symbol = :none, kwargs...) where {S1 <: Real, S2 <: Real}

    # Set up system of equations
    _my_eqn = (F, x) -> _deterministic_equations(F, x, m)

    # Exploit sparsity?
    if sparse_jacobian
        nlsolve_jacobian!, jac =
            construct_sparse_jacobian_function(m, _my_eqn, :deterministic, autodiff;
                                               sparsity = sparsity, colorvec = colorvec,
                                               jac_cache = jac_cache,
                                               sparsity_detection = sparsity_detection)
        out = nlsolve(OnceDifferentiable(_my_eqn, nlsolve_jacobian!, x0, copy(x0), jac), x0; kwargs...)
    else
        out = nlsolve(OnceDifferentiable(_my_eqn, x0, copy(x0), autodiff,
                                         ForwardDiff.Chunk(min(m.Nz, m.Ny))), x0; kwargs...)
    end

    if out.f_converged
        m.z .= out.zero[1:m.Nz]
        m.y .= out.zero[(m.Nz + 1):end]

        if verbose in [:low, :high]
            println("A deterministic steady state has been found")
        end
    else
        error("A deterministic steady state could not be found.")
    end
end

function _deterministic_equations(F::AbstractVector{<: Number}, x::AbstractVector{<: Number},
                                  m::RiskAdjustedLinearization)
    # Unpack input vector
    z = @view x[1:m.Nz]
    y = @view x[(m.Nz + 1):end]

    # Update μ(z, y) and ξ(z, y)
    update!(m.nonlinear, z, y, m.Ψ; select = Symbol[:μ, :ξ])

    # Calculate residuals
    μ_sss             = get_tmp(m.nonlinear.μ.cache, z, y, (1, 1)) # select the first DiffCache b/c that
    ξ_sss             = get_tmp(m.nonlinear.ξ.cache, z, y, (1, 1)) # one corresponds to autodiffing both z and y
    F[1:m.Nz]         = μ_sss - z
    F[(m.Nz + 1):end] = ξ_sss + m.linearization[:Γ₅] * z + m.linearization[:Γ₆] * y
end


"""
```
steady_state_errors(m::RiskAdjustedLinearization, stochastic_steady_state::Bool = true)
```

calculates the errors in the system of equations characterizing the steady state.
When the second input is `true`, the steady state is the stochastic steady state,
and when it is false, the steady state is the deterministic steady state.
"""
function steady_state_errors(m::RiskAdjustedLinearization, stochastic_steady_state::Bool = true)
    if stochastic_steady_state
        return vcat(m[:μ_sss] - m.z, m[:ξ_sss] + m[:Γ₅] * m.z + m[:Γ₆] * m.y + m[:𝒱_sss],
                    vec(m[:Γ₃] + m[:Γ₄] * m.Ψ + (m[:Γ₅] + m[:Γ₆] * m.Ψ) * (m[:Γ₁] + m[:Γ₂] * m.Ψ) + m[:JV]))
    else
        return vcat(m[:μ_sss] - m.z, m[:ξ_sss] + m[:Γ₅] * m.z + m[:Γ₆] * m.y)
    end
end
