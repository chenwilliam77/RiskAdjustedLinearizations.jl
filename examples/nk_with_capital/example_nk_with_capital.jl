# This script actually solves the WachterDisasterRisk model with a risk-adjusted linearization
# and times the methods, if desired
using RiskAdjustedLinearizations, JLD2, LinearAlgebra, Test
# include("nk_with_capital.jl")
out = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "nk_with_capital_output.jld2"), "r")

# Settings
autodiff              = false
algorithm             = :relaxation
euler_equation_errors = false
test_price_dispersion = false # check if price dispersion in steady state is always bounded below by 1
plot_irfs             = true
horizon               = 40    # horizon for IRFs
N_approx              = 1     # Number of periods ahead used for forward-difference equations

# Set up
m_nk = NKCapital() # create parameters
m = nk_capital(m_nk) # instantiate risk-adjusted linearization
autodiff_method = autodiff ? :forward : :central

# Solve!
solve!(m; algorithm = :deterministic, autodiff = autodiff_method)
@test m.z ≈ out["z_det"]
@test m.y ≈ out["y_det"]
@test m.Ψ ≈ out["Psi_det"]
z_det = copy(m.z)
y_det = copy(m.y)
Ψ_det = copy(m.Ψ)

solve!(m; algorithm = algorithm, autodiff = autodiff_method)
@test m.z ≈ out["z"]
@test m.y ≈ out["y"]
@test m.Ψ ≈ out["Psi"]

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
    # shocks = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "nk_with_capital_shocks.jld2"), "r")["shocks"]

    # With this simple model, the Euler equation holds exactly
    @test abs(euler_equation_error(m, nk_cₜ, (a, b, c, d) -> nk_logSDFxR(a, b, c, d; β = m_nk.β, σ = m_nk.σ),
                                   nk_𝔼_quadrature, shocks, summary_statistic = x -> norm(x, Inf))) ≈ 0.

    # Can calculate the Euler equation error for q, s₁, and s₂ as well by treating these variables as "consumption variables"
end

if plot_irfs
    # Show IRFs of interesting variables (discount rate, labor supply, productivity, and MP shocks)
    m_nk = NKCapital()
    m = nk_capital(m_nk)

    solve!(m; algorithm = algorithm, autodiff = autodiff_method)

    z_irfs = Dict()
    y_irfs = Dict()

    for k in keys(m_nk.SH)
        z_irfs[k], y_irfs[k] = impulse_responses(m, horizon, m_nk.SH[k], 1.)
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
            exp.(y_irfs[k][m_nk.J[:r], 1:end - 1] - y_irfs[k][m_nk.J[:π], 1:end - 1] .+ (m.y[m_nk.J[:r]] - m.y[m_nk.J[:π]]))
        plot_dicts[k][:real_excess_ret] = plot(1:(horizon - 1), exc_ret, label = "Real Excess Returns",
                                               linewidth = 3, color = :black)
    end
end

nothing
