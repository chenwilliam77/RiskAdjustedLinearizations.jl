using UnPack, OrderedCollections, LinearAlgebra, JLD2

# Load guesses
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../test/reference/crw_sss.jld2"), "r")

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

function crw(m::CoeurdacierReyWinant{T}) where {T <: Real}
    @unpack σr, σy, β, γ, θ, ρr, ρy, rr, yy = m

    # N = exp(rₜ) * Aₜ₋₁ + Yₜ, where Aₜ is foreign assets and Yₜ is the endowment
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
        F_type = eltype(F)
        F[S[:N], J[:x]] = z[S[:N]] - exp(y[J[:c]])
        F[S[:N], J[:w]] = 1.
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Σ(F, z, y)
        F_type = eltype(F)
        F[S[:r], SH[:εr]] = σr
        F[S[:y], SH[:εy]] = σy
    end

    function ccgf(F, α, z)
        F .= .5 * diag(α * α')
    end

    Γ₅ = zeros(T, Ny, Nz)
    Γ₅[J[:c], S[:r]] = 1.
    Γ₅[J[:x], S[:r]] = 1.
    Γ₅[J[:w], S[:y]] = 1.

    Γ₆ = zeros(T, Ny, Ny)
    Γ₆[J[:c], J[:c]] = -γ

    z = zguess
    y = yguess
    Ψ = Psiguess
    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nε; jump_dependent_shock_matrices = true)
end
