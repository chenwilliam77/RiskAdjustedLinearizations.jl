# This script actually solves the WachterDisasterRisk model with a risk-adjusted linearization
# and times the methods, if desired
using RiskAdjustedLinearizations, JLD2, LinearAlgebra, Test

# Number of quadrature points
n_GH = 5
include("nk_with_capital.jl")

# Settings
testing               = true         # check model's solution under default parameters against saved output
autodiff              = false
algorithm             = :relaxation
euler_equation_errors = false
test_price_dispersion = false        # check if price dispersion in steady state is always bounded below by 1
plot_irfs             = false
horizon               = 40    # horizon for IRFs
N_approx              = 5     # Number of periods ahead used for forward-difference equations

# Set up
m_nk = NKCapital(; N_approx = N_approx) # create parameters
m = nk_capital(m_nk) # instantiate risk-adjusted linearization
autodiff_method = autodiff ? :forward : :central

# Solve!
solve!(m; algorithm = algorithm, autodiff = autodiff_method)

if testing
    out = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "nk_with_capital_output.jld2"), "r")

    test_m_nk = NKCapital(; N_approx = 1) # create parameters
    test_m = nk_capital(test_m_nk) # instantiate risk-adjusted linearization

    solve!(test_m; algorithm = :deterministic, verbose = :none)
    @test test_m.z ≈ out["z_det"]
    @test test_m.y ≈ out["y_det"]
    @test test_m.Ψ ≈ out["Psi_det"]

    solve!(test_m; algorithm = :relaxation, verbose = :none)
    @test test_m.z ≈ out["z"]
    @test test_m.y ≈ out["y"]
    @test test_m.Ψ ≈ out["Psi"]

    test_5_m_nk = NKCapital(; N_approx = 5) # create parameters
    test_5_m = nk_capital(test_5_m_nk) # instantiate risk-adjusted linearization
    solve!(test_5_m; algorithm = :relaxation, verbose = :none)
    @test test_5_m.z ≈ out["z_5"]
    @test test_5_m.y ≈ out["y_5"]
    @test test_5_m.Ψ ≈ out["Psi_5"]
end

if test_price_dispersion
    π_ss_vec = log.(range(1 - .005, stop = 1 + .005, length = 10)) # On annualized basis, range from -2% to 2% target inflation
    det_soln = Dict()
    sss_soln = Vector{RiskAdjustedLinearization}(undef, length(π̃_ss_vec))

    for (i, π_ss) in enumerate(π_ss_vec)
        local m_nk = NKCapital(; π_ss = π_ss)
        local m = nk_capital(m_nk)
        solve!(m; algorithm = :deterministic, verbose = :none)
        det_soln[i] = Dict()
        det_soln[i][:z] = copy(m.z)
        det_soln[i][:y] = copy(m.y)
        det_soln[i][:Ψ] = copy(m.Ψ)
        solve!(m; algorithm = algorithm, verbose = :none)
        sss_soln[i] = m
    end

    det_v = exp.([det_soln[i][:z][3] for i in 1:length(det_soln)])
    sss_v = exp.([sss_soln[i].z[3] for i in 1:length(sss_soln)])
    @test all(det_v .> 1.)
    @test all(sss_v .> 1.)
end

if euler_equation_errors

    # Load shocks. Using CRW ones b/c that model also has 2 standard normal random variables
    shocks = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "nk_with_capital_shocks.jld2"), "r")["shocks"]

    # Calculate Euler equation for bonds
    @test abs(euler_equation_error(m, nk_cₜ, (a, b, c, d) -> nk_log_euler(a, b, c, d; β = m_nk.β, γ = m_nk.γ, J = m_nk.J),
                                   nk_𝔼_quadrature, shocks, summary_statistic = x -> norm(x, Inf))) ≈ 0.

    # Can calculate the Euler equation error for q, s₁, and s₂ as well by treating these variables as "consumption variables"
    # but doing the Euler equation error calculation "semi-manually" b/c of the forward difference equations
    impl_output = Dict()
    for k in [:dq, :pq, :ds₁, :ps₁, :ds₂, :ps₂]
        impl_output[k] = Dict()
    end

    _states, _jumps = simulate(m, 10, shocks[:, 1:10], m.z)
    q_ral  = _jumps[m_nk.J[:q], :]
    s₁_ral = _jumps[m_nk.J[:s₁], :]
    s₂_ral = _jumps[m_nk.J[:s₂], :]

    for i in 1:m_nk.N_approx
        impl_output[:dq][i] = log.(euler_equation_error(m, (m, zₜ) -> nk_dqₜ(m, zₜ, i, m_nk.J),
                                                        (a, b, c, d) -> nk_log_dq(a, b, c, d; β = m_nk.β,
                                                                                  γ = m_nk.β, i = i, J = m_nk.J, S = m_nk.S),
                                                        nk_𝔼_quadrature, shocks[:, 1:10], return_soln = true))
        impl_output[:pq][i] = log.(euler_equation_error(m, (m, zₜ) -> nk_pqₜ(m, zₜ, i, m_nk.J),
                                                        (a, b, c, d) -> nk_log_pq(a, b, c, d; β = m_nk.β,
                                                                                  γ = m_nk.β, i = i, J = m_nk.J, S = m_nk.S),
                                                        nk_𝔼_quadrature, shocks[:, 1:10], return_soln = true))
        impl_output[:ds₁][i - 1] = log.((i == 1) ? [nk_ds₁ₜ(m, _states[:, t], i - 1, m_nk.J) for t in 1:size(_states, 2)] :
                                        euler_equation_error(m, (m, zₜ) -> nk_ds₁ₜ(m, zₜ, i - 1, m_nk.J),
                                                             (a, b, c, d) -> nk_log_ds₁(a, b, c, d; β = m_nk.β,
                                                                                        γ = m_nk.β, θ = m_nk.θ, ϵ = m_nk.ϵ,
                                                                                        i = i - 1, J = m_nk.J, S = m_nk.S),
                                                             nk_𝔼_quadrature, shocks[:, 1:10], return_soln = true))
        impl_output[:ps₁][i] = log.(euler_equation_error(m, (m, zₜ) -> nk_ps₁ₜ(m, zₜ, i, m_nk.J),
                                                         (a, b, c, d) -> nk_log_ps₁(a, b, c, d; β = m_nk.β,
                                                                                    γ = m_nk.β, θ = m_nk.θ, ϵ = m_nk.ϵ,
                                                                                    i = i, J = m_nk.J, S = m_nk.S),
                                                         nk_𝔼_quadrature, shocks[:, 1:10], return_soln = true))
        impl_output[:ds₂][i - 1] = log.((i == 1) ? [nk_ds₂ₜ(m, _states[:, t], i - 1, m_nk.J) for t in 1:size(_states, 2)] :
                                        euler_equation_error(m, (m, zₜ) -> nk_ds₂ₜ(m, zₜ, i - 1, m_nk.J),
                                                             (a, b, c, d) -> nk_log_ds₂(a, b, c, d; β = m_nk.β,
                                                                                        γ = m_nk.β, θ = m_nk.θ, ϵ = m_nk.ϵ,
                                                                                        i = i - 1, J = m_nk.J, S = m_nk.S),
                                                             nk_𝔼_quadrature, shocks[:, 1:10], return_soln = true))
        impl_output[:ps₂][i] = log.(euler_equation_error(m, (m, zₜ) -> nk_ps₂ₜ(m, zₜ, i, m_nk.J),
                                                         (a, b, c, d) -> nk_log_ps₂(a, b, c, d; β = m_nk.β,
                                                                                    γ = m_nk.β, θ = m_nk.θ, ϵ = m_nk.ϵ,
                                                                                    i = i, J = m_nk.J, S = m_nk.S),
                                                         nk_𝔼_quadrature, shocks[:, 1:10], return_soln = true))
    end

    q_impl = log.(sum([exp.(x) for x in collect(values(impl_output[:dq]))]) + exp.(impl_output[:pq][m_nk.N_approx]))
    s₁_impl = log.(sum([exp.(x) for x in collect(values(impl_output[:ds₁]))]) + exp.(impl_output[:ps₁][m_nk.N_approx]))
    s₂_impl = log.(sum([exp.(x) for x in collect(values(impl_output[:ds₂]))]) + exp.(impl_output[:ps₂][m_nk.N_approx]))

    @show maximum(abs.((exp.(q_impl) - exp.(q_ral)) ./ exp.(q_ral)))
    @show maximum(abs.((exp.(s₁_impl) - exp.(s₁_ral)) ./ exp.(s₁_ral)))
    @show maximum(abs.((exp.(s₂_impl) - exp.(s₂_ral)) ./ exp.(s₂_ral)))
end

if plot_irfs
    # Show IRFs of interesting variables (discount rate, labor supply, productivity, and MP shocks)
    m_nk = NKCapital()
    m = nk_capital(m_nk)

    solve!(m; algorithm = algorithm, autodiff = autodiff_method)

    z_irfs = Dict()
    y_irfs = Dict()

    for k in keys(m_nk.SH)
        z_irfs[k], y_irfs[k] = impulse_responses(m, horizon, m_nk.SH[k], 1.) # 1 positive standard deviation shock
    end

    using Plots
    plot_dicts = Dict()

    for k in keys(m_nk.SH)
        plot_dicts[k] = Dict()

        plot_dicts[k][:output] = plot(1:horizon, y_irfs[k][m_nk.J[:output], :], label = "Output",
                                      linewidth = 3, color = :black)
        plot_dicts[k][:l] = plot(1:horizon, y_irfs[k][m_nk.J[:l], :], label = "Hours",
                                 linewidth = 3, color = :black)
        plot_dicts[k][:w] = plot(1:horizon, y_irfs[k][m_nk.J[:w], :], label = "Real Wage",
                                 linewidth = 3, color = :black)
        plot_dicts[k][:rk] = plot(1:horizon, y_irfs[k][m_nk.J[:rk], :], label = "Rental Rate of Capital",
                                  linewidth = 3, color = :black)
        plot_dicts[k][:k] = plot(1:horizon, z_irfs[k][m_nk.S[:k₋₁], :], label = "Capital Stock",
                                 linewidth = 3, color = :black)
        plot_dicts[k][:π] = plot(1:horizon, y_irfs[k][m_nk.J[:π], :], label = "Inflation",
                                 linewidth = 3, color = :black)
        plot_dicts[k][:q] = plot(1:horizon, y_irfs[k][m_nk.J[:q], :], label = "Price of Capital",
                                 linewidth = 3, color = :black)
        plot_dicts[k][:x] = plot(1:horizon, y_irfs[k][m_nk.J[:x], :], label = "Investment",
                                 linewidth = 3, color = :black)
        plot_dicts[k][:r] = plot(1:horizon, y_irfs[k][m_nk.J[:r], :], label = "Nominal Interest Rate",
                                 linewidth = 3, color = :black)

        # excess returns on capital (exploits properties of IRFs)
        EₜRₖₜ₊₁ = exp.(y_irfs[k][m_nk.J[:rk], 2:end] .+ m.y[m_nk.J[:rk]])
        EₜQₜ₊₁ = exp.(y_irfs[k][m_nk.J[:q], 2:end] .+ m.y[m_nk.J[:q]])
        EₜΩₜ₊₁ = exp.(y_irfs[k][m_nk.J[:ω], 2:end] .+ m.y[m_nk.J[:ω]])
        exc_ret = (EₜRₖₜ₊₁ + EₜQₜ₊₁ .* EₜΩₜ₊₁) ./ exp.(y_irfs[k][m_nk.J[:q], 1:end - 1] .+ m.y[m_nk.J[:q]]) -
            exp.(y_irfs[k][m_nk.J[:r], 1:end - 1] - y_irfs[k][m_nk.J[:π], 1:end - 1] .+ (m.y[m_nk.J[:r]] - m.y[m_nk.J[:π]])) .-
            (exp.(m.y[m_nk.J[:rk]]) + exp.(m.y[m_nk.J[:q]] + m.y[m_nk.J[:ω]])) / exp.(m.y[m_nk.J[:q]])
        plot_dicts[k][:real_excess_ret] = plot(1:(horizon - 1), exc_ret, label = "Real Excess Returns",
                                               linewidth = 3, color = :black)
    end
end

nothing
