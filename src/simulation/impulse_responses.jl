"""
```
impulse_responses(m, horizon, shock_ind, shock_size, z₀; deviations = true)
impulse_responses(m, horizon, shock_ind, shock_size; deviations = true)
```

calculates impulse responses according to a risk-adjusted linearization,
given a size for the shock, the index of the shock, and the initial state.
The second method assumes that the initial state is the stochastic steady state.

### Inputs
- `m::RiskAdjustedLinearization`: a solved risk-adjusted linearization of a dynamic economy
- `z₀::AbstractVector`: the initial state from which the economy begins
- `horizon::Int`: number of periods to be simulated
- `shock_ind::Int`: index of the shock that should be nonzero (other shocks are zero)
- `shock_size::Number`: size of the shock (can be positive or negative)

### Keywords
- `deviations::Bool`: if true, the impulse responses are returned in deviations from steady state.

### Outputs
- `states`: a matrix of the simulated path of states `z`, with type specified by the array type of `z₀`
- `jumps`: a matrix of the simulated path of jump variables `y`, with type specified by the array type of `z₀`
"""
function impulse_responses(m::RiskAdjustedLinearization, horizon::Int, shock_ind::Int,
                           shock_size::Number, z₀::AbstractVector;
                           deviations::Bool = true)

    # Create shock vector and output matrices
    shock            = zeros(eltype(z₀), m.Nε, 1)
    shock[shock_ind] = shock_size
    states           = similar(z₀, m.Nz, horizon)
    jumps            = similar(z₀, m.Ny, horizon)

    # Calculate state after impact
    states[:, 1], jumps[:, 1] = simulate(m, 1, shock, z₀)

    # Simulate with no other shocks drawn
    states[:, 2:end], jumps[:, 2:end] = simulate(m, horizon - 1, (@view states[:, 1]))

    if deviations
        return states .- m.z, jumps .- m.y
    else
        return states, jumps
    end
end

function impulse_responses(m::RiskAdjustedLinearization, horizon::Int, shock_ind::Int, shock_size::Number;
                           deviations::Bool = true)

    # Calculate starting at stochastic steady state
    return impulse_responses(m, horizon, shock_ind, shock_size, m.z; deviations = deviations)
end
