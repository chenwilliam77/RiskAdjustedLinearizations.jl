# FIX THE EQUATIONS B/C NOT CORRECT CURRENTLY
using UnPack, OrderedCollections, LinearAlgebra

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

        # This equation is mis-specified b/c y_t should also have a Lambda term, namely we're missing an expectational Y_t term, in the same way the w jump variable seems pointeless (replace with expectational Y_t?) but first let's solve this as is
        # Nt = e^(rt) A_{t - 1} + Y_t
        # E_t N_{t + 1} = E_t [e^{r_{t + 1}} * A_t + Y_{t + 1}]
        #               = E_t [R_{t + 1} * (N_t - C_t) + Y_{t + 1}]
        #               = E_t [R_{t + 1} * (N_t - C_t) + Y_{t + 1}]
        # N_{t + 1}     = R_{t + 1} * (N_t - C_t) + Y_{t + 1}
        #               = E_t [R_{t + 1} * (N_t - C_t)] + R_{t + 1} * (N_t - C_t) - E_t [R_{t + 1} * (N_t - C_t)]
        #                 + (Y_{t + 1} - E_t[Y_{t + 1}]) + E_t[Y_{t + 1}]
        #               = E_t [R_{t + 1}] * (N_t - C_t) + E_t[Y_{t + 1}] +
        #                 + (N_t - C_t) (R_{t + 1} - E_t[R_{t + 1}]) + (Y_{t + 1} - E_t[Y_{t + 1}])

    function μ(F, z, y) # note that y here refers to jump variables
        #               = E_t [R_{t + 1}] * (N_t - C_t) + E_t[Y_{t + 1}] +
        #                 + (N_t - C_t) (R_{t + 1} - E_t[R_{t + 1}]) + (Y_{t + 1} - E_t[Y_{t + 1}])

        F[S[:N]] = exp(y[J[:w]]) + exp(y[J[:x]]) * (z[S[:N]] - exp(y[J[:c]]))
        # F[S[:N]] = exp(z[S[:y]]) + exp(y[J[:x]]) * (z[S[:N]] - exp(y[J[:c]]))
        F[S[:r]] = (1 - ρr) * rr + ρr * z[S[:r]]
        F[S[:y]] = (1 - ρy) * yy + ρy * z[S[:y]]
    end

    function ξ(F, z, y)
        F[J[:c]] = log(β) + γ * y[J[:c]]    # Euler equation
        F[J[:x]] = -y[J[:x]]                # rₜ₊₁ - xₜ, rational expectations
        F[J[:w]] = -y[J[:w]]                # yₜ₊₁ - wₜ
        # F[J[:w]] = exp(z[S[:r]]) - y[J[:w]]                # yₜ₊₁ - wₜ
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

    z = [θ, rr, 1e-4]
    y = [0.005, rr + .5 * σr^2, exp(rr)]
    Ψ = zeros(T, Ny, Nz)
    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nε; jump_dependent_shock_matrices = true)
end
