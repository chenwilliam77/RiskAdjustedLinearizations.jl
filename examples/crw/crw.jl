using UnPack, OrderedCollections, LinearAlgebra, JLD2, SparseArrays

# Load guesses
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "crw_sss.jld2"), "r")

zguess = vec(sssout["z_rss"])
yguess = vec(sssout["y_rss"])
Psiguess = sssout["Psi_rss"]

mutable struct CoeurdacierReyWinant{T <: Real}
    σr::T # SD of interest rate shocks
    σy::T # SD of endowment shocks
    β::T  # intertemporal discount rate
    γ::T  # risk aversion coefficient
    θ::T
    ρr::T # persistence of interest rate
    ρy::T # persistence of endowment
    rr::T # long-run interest rate
    yy::T # long-run endowment
end

function CoeurdacierReyWinant(; σr::T = .025, σy::T = .025, β::T = .96, γ::T = 2.,
                              θ::T = 1., ρr::T = .9, ρy::T = .9, rr::T = .01996, yy::T = log(θ)) where {T <: Real}
    return CoeurdacierReyWinant{T}(σr, σy, β, γ, θ, ρr, ρy, rr, yy)
end

function crw(m::CoeurdacierReyWinant{T}; Ψ = nothing, sparse_jacobian::Vector{Symbol} = Symbol[],
             sparse_arrays::Bool = false) where {T <: Real}
    @unpack σr, σy, β, γ, θ, ρr, ρy, rr, yy = m

    # Nₜ = exp(rₜ) * Aₜ₋₁ + Yₜ, where Aₜ is foreign assets and Yₜ is the endowment
    # The jump variables are consumption, expected return on assets Xₜ = 𝔼ₜ[Rₜ₊₁], and
    # Wₜ = 𝔼ₜ[Yₜ₊₁]
    S  = OrderedDict{Symbol, Int}(:N => 1, :r => 2, :y => 3) # State variables
    J  = OrderedDict{Symbol, Int}(:c => 1, :x => 2, :w => 3) # Jump variables
    SH = OrderedDict{Symbol, Int}(:εr => 1, :εy => 2)        # Exogenous shocks
    Nz = length(S)
    Ny = length(J)
    Nε = length(SH)

    function μ(F, z, y) # note that y here refers to jump variables
        F[S[:N]] = exp(y[J[:w]]) + exp(y[J[:x]]) * (z[S[:N]] - exp(y[J[:c]]))
        F[S[:r]] = (1 - ρr) * rr + ρr * z[S[:r]]
        F[S[:y]] = (1 - ρy) * yy + ρy * z[S[:y]]
    end

    function ξ(F, z, y)
        F[J[:c]] = log(β) + γ * y[J[:c]]    # Euler equation
        F[J[:x]] = -y[J[:x]]                # rₜ₊₁ - xₜ, rational expectations
        F[J[:w]] = -y[J[:w]]                # yₜ₊₁ - wₜ
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Λ(F, z, y)
        F[S[:N], J[:x]] = z[S[:N]] - exp(y[J[:c]])
        F[S[:N], J[:w]] = 1.
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Σ(F, z, y)
        F[S[:r], SH[:εr]] = σr
        F[S[:y], SH[:εy]] = σy
    end

    Γ₅ = zeros(T, Ny, Nz)
    Γ₅[J[:c], S[:r]] = 1.
    Γ₅[J[:x], S[:r]] = 1.
    Γ₅[J[:w], S[:y]] = 1.

    Γ₆ = zeros(T, Ny, Ny)
    Γ₆[J[:c], J[:c]] = -γ

    if sparse_arrays
        Γ₅ = sparse(Γ₅)
        Γ₆ = sparse(Γ₆)
    end

    z = zguess
    y = yguess
    if isnothing(Ψ)
        Ψ = Psiguess
    end

    if sparse_arrays
        return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, crw_ccgf, z, y, Ψ, Nε; sparse_jacobian = sparse_jacobian,
                                         Λ_cache_init = dims -> spzeros(dims...),
                                         Σ_cache_init = dims -> spzeros(dims...),
                                         jump_dependent_shock_matrices = true)
    else
        return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, crw_ccgf, z, y, Ψ, Nε; sparse_jacobian = sparse_jacobian,
                                         jump_dependent_shock_matrices = true)
    end
end

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

function crw_ccgf(F, α, z)
    # F .= .5 * diag(α * α') # slower but this is the underlying math
    sum!(F, α.^2) # faster implementation
    F .*= .5
end
