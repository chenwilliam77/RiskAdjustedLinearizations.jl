"""
```
euler_equation_error(m, cₜ, logSDFxR, 𝔼_quadrature, zₜ = m.z;
    c_init = NaN, kwargs...)
euler_equation_error(m, cₜ, logSDFxR, 𝔼_quadrature, shock_matrix, p, zₜ = m.z;
    c_init = NaN, summary_statistic = x -> norm(x, Inf), burnin = 0, kwargs...)
```
calculates standard Euler equation errors, as recommended by Judd (1992).
The first method calculates the error at some state `zₜ`, which defaults
to the stochastic steady state. The second method simulates the state
vector from an initial state `zₜ` (defaults to stochastic steady state)
given a sequence of drawn shocks, evaluates the Euler equation errors,
and returns some summary statistic of the errors specified by the keyword
`summary_statistic`.

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

### Keywords
- `c_init::Number`: an initial guess to be used when solving the "true" consumption policy using
    quadrature. The default is the consumption policy according to the `RiskAdjustedLinearization`
- `summary_statistic::Function`: a `Function` used to compute a summary statistic from the
    ergodic set of Euler equation errors. The default is the maximum absolute error.
- `burnin::Int`: number of periods to drop as burn-in
- `kwargs`: Any keyword arguments for `nlsolve` can be passed, too, e.g. `ftol` or `autodiff`
    since `nlsolve` is used to calculate the "true" consumption policy.
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
                              summary_statistic::Function = x -> norm(x, Inf), burnin::Int = 0, kwargs...)

    # Set up
    T = size(shock_matrix, 2)

    # Simulate states
    states, _ = simulate(m, T, shock_matrix, zₜ)

    # Compute implied consumption according to the quadrature rule for each state
    # and expected consumption according to RAL
    err = [euler_equation_error(m, cₜ, logSDFxR, 𝔼_quadrature, (@view states[:, t]); c_init = c_init, kwargs...) for t in (burnin + 1):T]

    # Return error in unit-free terms
    return summary_statistic(err)
end

"""
```
dynamic_euler_equation_error(m, cₜ, logSDFxR, 𝔼_quadrature, endo_states, n_aug,
    shock_matrix, zₜ = m.z; c_init = NaN, summary_statistic = x -> norm(x, Inf),
    burnin = 0, raw_output = false, kwargs...)
```
calculates dynamic Euler equation errors, as proposed in Den Haan (2009).
The Euler equation is

``math
\\begin{aligned}
0 = \\log \\mathbb{E}_t \\exp(m_{t + 1} + r_{t + 1}) = \\log \\mathbb{E}_t[M_{t + 1} R_{t + 1}],
\\end{aligned}
``

where ``m_{t + 1} = \\log(M_{t + 1})`` is the log stochastic discount factor and ``r_{t + 1} = \\log(R_{t + 1})``
is the risk free rate.

The dynamic errors are computed according the following algorithm.

1. Simulate according to the risk-adjusted linearization time series for the state variables
2. Using the time series from 1, compute time series for consumption and
    some state variable (usually capital) that can ensure budget constraints hold and markets
    clear when computing consumption by applying quadrature.
3. Generate a second "implied" time series for consumption and the "capital" state variable,
    starting from the same initial state as 2. Repeat the following steps at each time period.
    (i)  Compute the conditional expectation in the Euler equation using quadrature to
         obtain implied consumption.
    (ii) Use budget constraint/market-clearing to compute implied capital.

By default, `dynamic_euler_equation_error` returns some summary statistic of the errors
specified by the keyword `summary_statistic`.

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
- `endo_states::Function`: augments the state variables in the risk-adjusted linearization,
    usually with one additional variable, which represents capital or assets.
- `n_aug::Int`: number of extra state variables added by `endo_states` (usually 1).
- `zₜ::AbstractVector`: a state at which to evaluate the Euler equation error
- `shock_matrix::Abstractmatrix`: a `Nε × T` matrix of shocks drawn from the distribution of exogenous shocks.

### Keywords
- `c_init::Number`: an initial guess to be used when solving the true consumption policy using
    quadrature. The default is the consumption policy according to the `RiskAdjustedLinearization`
- `summary_statistic::Function`: a `Function` used to compute a summary statistic from the
    ergodic set of Euler equation errors. The default is the maximum absolute error.
- `burnin::Int`: number of periods to drop as burn-in
- `kwargs`: Any keyword arguments for `nlsolve` can be passed, too, e.g. `ftol` or `autodiff`
    since `nlsolve` is used to calculate the "true" consumption policy.
"""
function dynamic_euler_equation_error(m::RiskAdjustedLinearization, cₜ::Function, logSDFxR::Function,
                                      𝔼_quadrature::Function, endo_states::Function, n_aug::Int,
                                      shock_matrix::AbstractMatrix, z₀::AbstractVector = m.z;
                                      c_init::Number = NaN, summary_statistic::Function = x -> norm(x, Inf),
                                      burnin::Int = 0, raw_output::Bool = false, kwargs...)

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
        return c_ral[(burnin + 1):end], c_impl[(burnin + 1):end], endo_states_ral[(burnin + 1):end], endo_states_impl[(burnin + 1):end]
    else
        return summary_statistic(((@view c_ral[(burnin + 1):end]) - (@view c_impl[(burnin + 1):end])) ./ (@view c_ral[(burnin + 1):end])), summary_statistic(vec((@view endo_states_ral[:, (burnin + 1):end]) - (@view endo_states_impl[:, (burnin + 1):end])) ./ vec((@view endo_states_ral[:, (burnin + 1):end])))
    end
end
