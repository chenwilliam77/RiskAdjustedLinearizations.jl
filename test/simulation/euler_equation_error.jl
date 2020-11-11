using RiskAdjustedLinearizations, JLD2, Test
include(joinpath(dirname(@__FILE__), "..", "..", "examples", "crw", "crw.jl"))

# Solve model
m_crw = CoeurdacierReyWinant()
m = crw(m_crw)
solve!(m, m.z, m.y, m.Ψ; algorithm = :homotopy)

# Calculate consumption at state zₜ
crw_cₜ(m, zₜ) = m.y[1] + (m.Ψ * (zₜ - m.z))[1]

# Evaluates m_{t + 1} + r_{t + 1}
function crw_logSDFxR(m, zₜ, εₜ₊₁, cₜ)
    zₜ₊₁, yₜ₊₁ = simulate(m, εₜ₊₁, zₜ)

    return log(m_crw.β) - m_crw.γ * (yₜ₊₁[1] - cₜ) + zₜ₊₁[2]
end

# Calculate 𝔼ₜ[exp(mₜ₊₁ + rₜ₊₁)] via quadrature
std_norm_mean = zeros(2)
std_norm_sig  = ones(2)
crw_𝔼_quadrature(f::Function) = gausshermite_expectation(f, std_norm_mean, std_norm_sig, 10)

# Load draws from bivariate standard normal
shocks = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "reference", "crw_shocks.jld2"), "r")["shocks"]

# Calculate Euler Equation errors
@test abs.(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature; c_init = m.y[1] * 1.1)) < 1e-11
@test abs.(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, m.z * 1.1; c_init = m.y[1] * 1.1)) < 1e-5
@test abs.(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, m.z * 1.1; c_init = m.y[1] * 1.1, method = :newton)) < 1e-5
@test abs(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, shocks, x -> norm(x, Inf))) < 7e-4
@test abs(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, shocks, x -> norm(x, 2))) < 2e-3
@test abs(euler_equation_error(m, crw_cₜ, crw_logSDFxR, crw_𝔼_quadrature, shocks, x -> mean(abs.(x)))) < 6e-5
