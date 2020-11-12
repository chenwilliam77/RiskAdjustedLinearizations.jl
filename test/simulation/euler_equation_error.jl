using RiskAdjustedLinearizations, JLD2, Statistics, Test
include(joinpath(dirname(@__FILE__), "..", "..", "examples", "crw", "crw.jl"))

# Solve model
m_crw = CoeurdacierReyWinant()
m = crw(m_crw)
solve!(m, m.z, m.y, m.Ψ; algorithm = :homotopy)

# Calculate consumption at state zₜ
crw_cₜ(m, zₜ) = exp(m.y[1] + (m.Ψ * (zₜ - m.z))[1])

# Evaluates m_{t + 1} + r_{t + 1}
function crw_logSDFxR(m, zₜ, εₜ₊₁, Cₜ)
    zₜ₊₁, yₜ₊₁ = simulate(m, εₜ₊₁, zₜ)

    return log(m_crw.β) - m_crw.γ * (yₜ₊₁[1] - log(Cₜ)) + zₜ₊₁[2]
end

# Calculate 𝔼ₜ[exp(mₜ₊₁ + rₜ₊₁)] via quadrature
std_norm_mean = zeros(2)
std_norm_sig  = ones(2)
crw_𝔼_quadrature(f::Function) = gausshermite_expectation(f, std_norm_mean, std_norm_sig, 10)

# Augment state variables with foreign assets
function crw_augment_states(m, zₜ, zₜ₋₁)
    yₜ = m.y + m.Ψ * (zₜ - m.z)
    Cₜ = exp(yₜ[1])
    # Aₜ = zₜ[1] - Cₜ
    return vcat(zₜ, zₜ[1] - Cₜ)
end

# Calculate implied state variable(s)
function crw_endo_states(m, zₜ, zₜ₋₁, c_impl)
    # rₜ, yₜ are exogenous while Nₜ = exp(rₜ) * Aₜ₋₁ + Yₜ is entirely pre-determined.
    # Thus, our implied state variable will be foreign asset Aₜ = Nₜ - Cₜ.

    # zₜ₋₁ may be the previous period's implied state, so we start from there
    # to calculate Aₜ₋₁.
    yₜ₋₁ = m.y + m.Ψ * (zₜ₋₁ - m.z) # Calculate implied jump variables last period
    Cₜ₋₁ = exp(yₜ₋₁[1])             # to get the implied consumption last period.
    Aₜ₋₁ = zₜ₋₁[1] - Cₜ₋₁           # Given that consumption, we compute implied foreign assets yesterday.
    Nₜ   = exp(zₜ[2]) * Aₜ₋₁ + exp(zₜ[3]) # Now we can get implied resources available today.

    return vcat(zₜ, Nₜ - exp(c_impl)) # This gives us implied foreign assets today, along with other state variables
end

# Load draws from bivariate standard normal
shocks = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "reference", "crw_shocks.jld2"), "r")["shocks"]

@testset "Calculate Euler Equation Errors using Gauss-Hermite quadrature" begin

    # Calculate Euler Equation errors
    @test abs.(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature; c_init = m.y[1] * 1.1)) < 1e-10
    @test abs.(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, m.z * 1.1; c_init = m.y[1] * 1.1)) < 2.3e-5
    @test abs.(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, m.z * 1.1; c_init = m.y[1] * 1.1, method = :newton)) < 1e-3
    @test abs(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, shocks, summary_statistic = x -> norm(x, Inf))) < 3e-5
    @test abs(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, shocks, summary_statistic = x -> norm(x, 2))) < 9e-5
    @test abs(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, shocks, summary_statistic = x -> mean(abs.(x)))) < 7e-6

    c_ral, c_impl, endo_states_ral, endo_states_impl =
        dynamic_euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, crw_endo_states, shocks;
                                     raw_output = true, n_aug = 1)
    c_err, endo_states_err = dynamic_euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, crw_endo_states, shocks;
                                                          raw_output = false, n_aug = 1)
    @test_throws BoundsError dynamic_euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, crw_endo_states, shocks;
                                                          raw_output = false)
    @test c_err < 2e-5
    @test endo_states_err < 1e-3
    @test c_err == norm((c_ral - c_impl) ./ c_ral, Inf)
    @test endo_state_err == norm(vec(endo_states_ral - endo_states_impl) ./ vec(endo_states_ral), Inf)
end
