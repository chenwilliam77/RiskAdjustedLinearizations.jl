"""
```
simulate(m, z₀, horizon, shock_matrix)
simulate(m, z₀, horizon)
```
simulates the economy approximated by a risk-adjusted linearization. The first method
incorporates an arbitrary path of shocks across the horizon while the
second method assumes no shocks occur during the horizon.

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
function simulate(m::RiskAdjustedLinearization, z₀::AbstractVector, horizon::Int, shock_matrix::AbstractMatrix)
    @assert horizon <= size(shock_matrix, 2) "There are not enough draws in shock_matrix (horizon <= size(shock_matrix, 2))"

    # Set up
    states = similar(z₀, m.Nz, horizon)
    jumps  = similar(z₀, m.Ny, horizon)
    Γ₁     = m[:Γ₁]
    Γ₂     = m[:Γ₂]
    y₀     = m.y + m.Ψ * (z₀ - m.z)

    # Create "shock function" which creates the mamtrix mapping shocks to states
    is_Λ_mat  = isa(m.nonlinear.Λ.cache, AbstractMatrix)
    is_Λ_zero = is_Λ_mat ? all(m[:Λ_sss] .≈ 0.) : false
    is_Σ_mat  = isa(m.nonlinear.Σ.cache, AbstractMatrix)
    shock_fnct = if is_Σ_mat && is_Λ_mat
        R = is_Λ_zero ? m.nonlinear[:Σ_sss] : (I - m.nonlinear[:Λ_sss] * m.Ψ) \ m.nonlinear[:Σ_sss]
       function _both_mat(z::AbstractVector, ε::AbstractVector)
           R * ε
       end
    elseif is_Λ_mat
        if is_Λ_zero
            function _only_nonzero_Λ_mat(z::AbstractVector, ε::AbstractVector)
                m.nonlinear.Σ(z) * ε
            end
        else
            function _only_zero_Λ_mat(z::AbstractVector, ε::AbstractVector)
                (I - m.nonlinear[:Λ_sss] * m.Ψ) \ (m.nonlinear.Σ(z) * ε)
            end
        end
    elseif is_Σ_mat
       function _only_Σ_mat(z::AbstractVector, ε::AbstractVector)
           (I - m.nonlinear.Λ(z) * m.Ψ) \ (m.nonlinear[:Σ_sss] * ε)
       end
    else
       function _both_fnct(z::AbstractVector, ε::AbstractVector)
           (I - m.nonlinear.Λ(z) * m.Ψ) \ (m.nonlinear.Σ(z) * ε)
       end
    end

    # Iterate forward!
    states[:, 1] = expected_transition(m.y, m.z, m.Ψ, Γ₁, Γ₂, z₀, y₀) + shock_fnct(z₀, shock_matrix[:, 1])
    jumps[:, 1]  = m.y + m.Ψ * ((@view states[:, 1]) - m.z)
    for t in 2:horizon
        states[:, t] = expected_transition(m.z, m.y, m.Ψ, Γ₁, Γ₂, (@view states[:, t - 1]), (@view jumps[:, t - 1])) +
            shock_fnct((@view states[:, t - 1]), shock_matrix[:, t])
        jumps[:, t]  = m.y + m.Ψ * ((@view states[:, t]) - m.z)
    end

    return states, jumps
end

function simulate(m::RiskAdjustedLinearization, z₀::AbstractVector, horizon::Int)

    # Set up
    states = similar(z₀, m.Nz, horizon)
    jumps  = similar(z₀, m.Ny, horizon)
    Γ₁     = m[:Γ₁]
    Γ₂     = m[:Γ₂]
    y₀     = m.y + m.Ψ * (z₀ - m.z)

    # Iterate forward!
    states[:, 1] = expected_transition(m.y, m.z, m.Ψ, Γ₁, Γ₂, z₀, y₀)
    jumps[:, 1]  = m.y + m.Ψ * ((@view states[:, 1]) - m.z)
    for t in 2:horizon
        states[:, t] = expected_transition(m.z, m.y, m.Ψ, Γ₁, Γ₂, (@view states[:, t - 1]), (@view jumps[:, t - 1]))
        jumps[:, t]  = m.y + m.Ψ * ((@view states[:, t]) - m.z)
    end

    return states, jumps
end

function expected_transition(z::AbstractVector, y::AbstractVector, Ψ::AbstractMatrix,
                             Γ₁::AbstractMatrix, Γ₂::AbstractMatrix, zₜ::AbstractVector, yₜ::AbstractVector)
    return z + Γ₁ * (zₜ - z) + Γ₂ * (yₜ - y)
end
