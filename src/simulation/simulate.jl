"""
```
simulate(m, horizon, shock_matrix, z₀)
simulate(m, horizon, z₀)
simulate(m, shock_vector, z₀)
simulate(m, horizon, shock_matrix)
simulate(m, horizon)
```
simulates the economy approximated by a risk-adjusted linearization. The first method
incorporates an arbitrary path of shocks across the horizon while the second method
assumes no shocks occur during the horizon. The third method calculates next
period's states and jump variables, given a vector of shocks. The fourth and fifth
methods are the same as the first two but assume the economy begins at the
stochastic steady state.

### Inputs
- `m::RiskAdjustedLinearization`: a solved risk-adjusted linearization of a dynamic economy
- `z₀::AbstractVector`: the initial state from which the economy begins
- `horizon::Int`: number of periods to be simulated
- `shock_matrix::AbstractMatrix`: a `Nε × T` length matrix, whose columns are draws from
    the distributions of exogenous shocks driving the economy approximated by `m`. The number of columns
    must be at least as long as `horizon`. If the number of columns is larger, then we do not use
    draws for columns `horizon + 1:T`.

### Outputs
- `states`: a matrix of the simulated path of states `z`, with type specified by the array type of `z₀`
- `jumps`: a matrix of the simulated path of jump variables `y`, with type specified by the array type of `z₀`
"""
function simulate(m::RiskAdjustedLinearization, horizon::Int, shock_matrix::AbstractMatrix, z₀::AbstractVector)
    @assert horizon <= size(shock_matrix, 2) "There are not enough draws in shock_matrix (horizon <= size(shock_matrix, 2))"

    # Set up
    states = similar(z₀, m.Nz, horizon)
    jumps  = similar(z₀, m.Ny, horizon)
    Γ₁     = m[:Γ₁]
    Γ₂     = m[:Γ₂]
    y₀     = m.y + m.Ψ * (z₀ - m.z)

    # Create "shock function" which creates the matrix mapping shocks to states
    shock_fnct = create_shock_function(m.nonlinear.Σ, m.nonlinear.Λ, m.z, m.y, m.Ψ)

    # Iterate forward!
    states[:, 1] = expected_transition(m.z, m.y, m.Ψ, Γ₁, Γ₂, z₀, y₀) + shock_fnct(z₀, shock_matrix[:, 1])
    jumps[:, 1]  = m.y + m.Ψ * ((@view states[:, 1]) - m.z)
    for t in 2:horizon
        states[:, t] = expected_transition(m.z, m.y, m.Ψ, Γ₁, Γ₂, (@view states[:, t - 1]), (@view jumps[:, t - 1])) +
            shock_fnct((@view states[:, t - 1]), shock_matrix[:, t])
        jumps[:, t]  = m.y + m.Ψ * ((@view states[:, t]) - m.z)
    end

    return states, jumps
end

function simulate(m::RiskAdjustedLinearization, horizon::Int, z₀::AbstractVector)

    # Set up
    states = similar(z₀, m.Nz, horizon)
    jumps  = similar(z₀, m.Ny, horizon)
    Γ₁     = m[:Γ₁]
    Γ₂     = m[:Γ₂]
    y₀     = m.y + m.Ψ * (z₀ - m.z)

    # Iterate forward!
    states[:, 1] = expected_transition(m.z, m.y, m.Ψ, Γ₁, Γ₂, z₀, y₀)
    jumps[:, 1]  = m.y + m.Ψ * ((@view states[:, 1]) - m.z)
    for t in 2:horizon
        states[:, t] = expected_transition(m.z, m.y, m.Ψ, Γ₁, Γ₂, (@view states[:, t - 1]), (@view jumps[:, t - 1]))
        jumps[:, t]  = m.y + m.Ψ * ((@view states[:, t]) - m.z)
    end

    return states, jumps
end

function simulate(m::RiskAdjustedLinearization, shock_vector::AbstractVector, z₀::AbstractVector)

    # Set up
    Γ₁     = m[:Γ₁]
    Γ₂     = m[:Γ₂]
    y₀     = m.y + m.Ψ * (z₀ - m.z)

    # Create "shock function" which creates the matrix mapping shocks to states
    shock_fnct = create_shock_function(m.nonlinear.Σ, m.nonlinear.Λ, m.z, m.y, m.Ψ)

    # Iterate forward!
    states = expected_transition(m.z, m.y, m.Ψ, Γ₁, Γ₂, z₀, y₀) + shock_fnct(z₀, shock_vector)
    jumps  = m.y + m.Ψ * (states - m.z)

    return states, jumps
end

function expected_transition(z::AbstractVector, y::AbstractVector, Ψ::AbstractMatrix,
                             Γ₁::AbstractMatrix, Γ₂::AbstractMatrix, zₜ::AbstractVector, yₜ::AbstractVector)
    return z + Γ₁ * (zₜ - z) + Γ₂ * (yₜ - y)
end

function simulate(m::RiskAdjustedLinearization, horizon::Int, shock_matrix::AbstractMatrix)
    simulate(m, horizon, shock_matrix, m.z)
end

function simulate(m::RiskAdjustedLinearization, horizon::Int)
    simulate(m, horizon, m.z)
end

# Use multiple dispatch to construct the correct shock function
function create_shock_function(Σ::RALF1{S}, Λ::RALF1{L}, z::AbstractVector,
                               y::AbstractVector, Ψ::AbstractMatrix) where {S <: AbstractMatrix, L <: AbstractMatrix}
    R = all(Λ.cache .≈ 0.) ? Σ.cache : (I - Λ.cache * Ψ) \ Σ.cache
    f = function _both_mat(z::AbstractVector, ε::AbstractVector)
        R * ε
    end

    return f
end

function create_shock_function(Σ::RALF1{S}, Λ::RALF1{L}, z::AbstractVector,
                               y::AbstractVector, Ψ::AbstractMatrix) where {S <: DiffCache, L <: AbstractMatrix}
    f = if all(Λ.cache .≈ 0.)
        function _only_nonzero_Λ_mat(z::AbstractVector, ε::AbstractVector)
            Σ(z) * ε
        end
    else
        function _only_zero_Λ_mat(z::AbstractVector, ε::AbstractVector)
            (I - Λ.cache * Ψ) \ (Σ(z) * ε)
        end
    end

    return f
end

function create_shock_function(Σ::RALF1{S}, Λ::RALF1{L}, z::AbstractVector,
                               y::AbstractVector, Ψ::AbstractMatrix) where {S <: AbstractMatrix, L <: DiffCache}
    f = function _only_Σ_mat(z::AbstractVector, ε::AbstractVector)
           (I - Λ(z) * Ψ) \ (Σ.cache * ε)
       end

    return f
end

function create_shock_function(Σ::RALF1{S}, Λ::RALF1{L}, z::AbstractVector,
                               y::AbstractVector, Ψ::AbstractMatrix) where {S <: DiffCache, L <: DiffCache}
    f = function _both_fnct(z::AbstractVector, ε::AbstractVector)
           (I - Λ(z) * Ψ) \ (Σ(z) * ε)
       end

    return f
end

function create_shock_function(Σ::RALF2{S}, Λ::RALF2{L}, z::AbstractVector,
                               y::AbstractVector, Ψ::AbstractMatrix) where {S <: AbstractMatrix, L <: AbstractMatrix}
    R = all(Λ.cache .≈ 0.) ? Σ.cache : (I - Λ.cache * Ψ) \ Σ.cache
    f = function _both_mat(z::AbstractVector, ε::AbstractVector)
        R * ε
    end

    return f
end

function create_shock_function(Σ::RALF2{S}, Λ::RALF2{L}, z::AbstractVector,
                               y::AbstractVector, Ψ::AbstractMatrix) where {S <: TwoDiffCache, L <: AbstractMatrix}
    f = if all(m[:Λ_sss] .≈ 0.)
        function _only_nonzero_Λ_mat(zₜ::AbstractVector, ε::AbstractVector)
            Σ(zₜ, y + Ψ * (zₜ - z)) * ε
        end
    else
        function _only_zero_Λ_mat(zₜ::AbstractVector, ε::AbstractVector)
            (I - Λ.cache * Ψ) \ (Σ(zₜ, y + Ψ * (zₜ - z)) * ε)
        end
    end

    return f
end

function create_shock_function(Σ::RALF2{S}, Λ::RALF2{L}, z::AbstractVector,
                               y::AbstractVector, Ψ::AbstractMatrix) where {S <: AbstractMatrix, L <: TwoDiffCache}
    f = function _only_Σ_mat(zₜ::AbstractVector, ε::AbstractVector)
           (I - Λ(zₜ, y + Ψ * (zₜ - z)) * Ψ) \ (Σ.cache * ε)
       end

    return f
end

function create_shock_function(Σ::RALF2{S}, Λ::RALF2{L}, z::AbstractVector,
                               y::AbstractVector, Ψ::AbstractMatrix) where {S <: TwoDiffCache, L <: TwoDiffCache}
    f = function _both_fnct(zₜ::AbstractVector, ε::AbstractVector)
        yₜ = y + Ψ * (zₜ - z)
        (I - Λ(zₜ, yₜ) * Ψ) \ (Σ(zₜ, yₜ) * ε)
       end

    return f
end
