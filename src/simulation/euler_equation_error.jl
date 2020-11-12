"""
```
euler_equation_error(m, cₜ, logSDFxR, 𝔼_quadrature, zₜ)
euler_equation_error(m, cₜ, logSDFxR, 𝔼_quadrature, shock_matrix, p, zₜ)
```
calculates standard Euler equation errors, as recommended by Judd (1992).
The first method calculates the error at some state `zₜ`, which defaults
to the stochastic steady state. The second method simulates the state
vector from an initial state `zₜ` (defaults to stochastic steady state)
given a sequence of drawn shocks and evaluates the Euler equation errors
according to one of three norms (see below).

The Euler equation is
``math
\\begin{aligned}
0 = \\log \\mathbb{E}_t \\exp(m_{t + 1} + r_{t + 1}) = \\log \\mathbb{E}_t[M_{t + 1} R_{t + 1}],
\\end{aligned}
``
where ``m_{t + 1} = \\log(M_{t + 1})`` is the log stochastic discount factor and ``r_{t + 1} = \\log(R_{t + 1})``
is the risk free rate.

### Inputs
- `m::RiskAdjustedLinearization`: A solved instance of a risk-adjusted linearization
- `cₜ::Function`: a function of `(m, zₜ)` that calculates consumption at state `zₜ`, given the
    state-space representation implied by `m`.
- `logSDFxR::Function`: a `Function` evaluating ``m_{t + 1} + r_{t + 1}``. The `Function` must
    take as input `(m, zₜ, εₜ₊₁, cₜ)`, where `m` is a `RiskAdjustedLinearization`,
    `zₜ` is a state vector at which to evaluate, `εₜ₊₁` is a draw from the distribution
    of exogenous shocks, and `cₜ` is the a guess for consumption at `zₜ` implied by
    the conditional expectation in the Euler equation when calculated with a quadrature rule.
- `𝔼_quadrature::Function`: a quadrature rule whose single input is a `Function` with a single
    input, which is a shock `εₜ₊₁`.
- `zₜ::AbstractVector`: a state at which to evaluate the Euler equation error
- `shock_matrix::Abstractmatrix`: a `Nε × T` matrix of shocks drawn from the distribution of exogenous shocks.
- `p::Number`: norm at which to evaluate the errors. This norm can be (a) `p=1` for expected absolute error,
    (b) `p=2` for expected squared error, and (c) `p=Inf` for maximum absolute error.
"""
function euler_equation_error(m::RiskAdjustedLinearization, cₜ::Function, logSDFxR::Function, 𝔼_quadrature::Function,
                              zₜ::AbstractVector = m.z; c_init::Number = NaN, kwargs...)

    # Compute expected consumption according to RAL
    c_ral = cₜ(m, zₜ)

    # Compute implied consumption according to the quadrature rule
    out = nlsolve(c -> [log(𝔼_quadrature(εₜ₊₁ -> exp(logSDFxR(m, zₜ, εₜ₊₁, c[1]))))], [isnan(c_init) ? c_ral : c_init];
                  kwargs...)
    if out.f_converged
        c_impl = out.zero[1]
    else
        error("Failed to solve implied consumption.")
    end

    # Return error in unit-free terms
    return (c_ral - c_impl) / c_ral
end

function euler_equation_error(m::RiskAdjustedLinearization, cₜ::Function, logSDFxR::Function, 𝔼_quadrature::Function,
                              shock_matrix::AbstractMatrix, zₜ::AbstractVector = m.z; c_init::Number = NaN,
                              summary_statistic::Function = x -> norm(x, Inf), kwargs...)

    # Set up
    T = size(shock_matrix, 2)

    # Simulate states
    states, _ = simulate(m, T, shock_matrix, zₜ)

    # Compute implied consumption according to the quadrature rule for each state
    # and expected consumption according to RAL
    err = [euler_equation_error(m, cₜ, logSDFxR, 𝔼_quadrature, (@view states[:, t]); c_init = c_init, kwargs...) for t in 1:T]

    # Return error in unit-free terms
    return summary_statistic(err)
end

function dynamic_euler_equation_error(m::RiskAdjustedLinearization, cₜ::Function, logSDFxR::Function,
                                      𝔼_quadrature::Function, endo_states::Function,
                                      shock_matrix::AbstractMatrix, z₀::AbstractVector = m.z;
                                      c_init::Number = NaN, summary_statistic::Function = x -> norm(x, Inf),
                                      raw_output::Bool = false, n_aug::Int = 0, kwargs...)

    # Set up
    T = size(shock_matrix, 2)
    c_impl = Vector{eltype(shock_matrix)}(undef, T)

    # Simulate states and calculate consumption according to RAL
    states, _  = simulate(m, T, shock_matrix, z₀)
    c_ral      = [cₜ(m, (@view states[:, t])) for t in 1:T]
    orig_i     = 1:size(states, 1)

    # Additional set up
    endo_states_impl = similar(states, length(orig_i) + n_aug, T)
    endo_states_ral  = similar(endo_states_impl)

    # For each state, calculate conditional expectation using quadrature rule
    # and compute the implied states
    out = nlsolve(c -> [log(𝔼_quadrature(εₜ₊₁ -> exp(logSDFxR(m, (@view states[:, 1]), εₜ₊₁, c[1]))))], [isnan(c_init) ? c_ral[1] : c_init];
                  kwargs...) # Do period 1 separately b/c needed to initialize endo_states_impl
    if out.f_converged
        c_impl[1] = out.zero[1]
    else
        error("Failed to solve implied consumption in period 1 of $T.")
    end
    endo_states_impl[:, 1] = endo_states(m, (@view states[:, 1]), z₀, c_impl[1])
    endo_states_ral[:, 1]  = endo_states(m, (@view states[:, 1]), z₀, c_ral[1])

    for t in 2:T
        out = nlsolve(c -> [log(𝔼_quadrature(εₜ₊₁ -> exp(logSDFxR(m, (@view states[:, t]), εₜ₊₁, c[1]))))], [isnan(c_init) ? c_ral[t] : c_init];
                      kwargs...)
        if out.f_converged
            c_impl[t] = out.zero[1]
        else
            error("Failed to solve implied consumption in period $t of $T.")
        end
        endo_states_impl[:, t] = endo_states(m, (@view states[:, t]), (@view endo_states_impl[orig_i, t - 1]), c_impl[t])
        endo_states_ral[:, t]  = endo_states(m, (@view states[:, t]), (@view endo_states_ral[orig_i, t - 1]),  c_ral[t])
    end

    # Calculate the errors
    if raw_output
        return c_ral, c_impl, endo_states_ral, endo_states_impl
    else
        return summary_statistic((c_ral - c_impl) ./ c_ral), summary_statistic(vec(endo_states_ral - endo_states_impl) ./ vec(endo_states_ral))
    end
end
