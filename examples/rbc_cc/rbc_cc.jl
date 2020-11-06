using UnPack, OrderedCollections, ForwardDiff, JLD2

init_z = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../test/reference/rbccc_det_ss_output.jld2"), "r")["z_dss"]
init_y = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../test/reference/rbccc_det_ss_output.jld2"), "r")["y_dss"]

mutable struct RBCCampbellCochraneHabits{T <: Real}
    IK̄::T
    β::T
    δ::T
    α::T
    ξ₃::T
    μₐ::T
    σₐ::T
    γ::T
    ρₛ::T
    S::T
end

function RBCCampbellCochraneHabits(; α::T = .36, ξ₃ = .23, δ::T = .1337, IK̄::T = (.025 * (1 + 1 / .23)) / (1 + 1 / .23),
                                   μₐ = .0049875, σₐ = .0064 / (1 - .36), γ = 2., ρₛ = .96717, β = .989, S = .057) where {T <: Real}
    return RBCCampbellCochraneHabits{T}(IK̄, β, δ, α, ξ₃, μₐ, σₐ, γ, ρₛ, S)
end

function rbc_cc(m::RBCCampbellCochraneHabits{T}) where {T <: Real}
    @unpack IK̄, β, δ, α, ξ₃, μₐ, σₐ, γ, ρₛ, S = m

    s  = OrderedDict{Symbol, Int}(:kₐ => 1,  :hats => 2) # State variables
    J  = OrderedDict{Symbol, Int}(:cₖ => 1, :iₖ => 2, :log_D_plus_Q => 3, :rf => 4, :q => 5, :log_E_RQ => 6) # Jump variables
    SH = OrderedDict{Symbol, Int}(:εₐ => 1) # Exogenous shocks
    Nz = length(s)
    Ny = length(J)
    Nε = length(SH)

    # Some additional functions used by μ and ξ
    Φ(x)  = exp(μₐ) - 1. + δ / (1. - ξ₃^2) + (IK̄ / (1. - 1. / ξ₃)) * ((x / IK̄)^(1. - 1. / ξ₃))
    Φ′(x) = (x / IK̄) ^ (-1. / ξ₃) # Φ′(x) = IK̄ * (IK̄) ^ (1. / ξ₃. - 1.) * x ^ (-1. / ξ₃)

    function μ(F, z, y)
        F_type      = eltype(F)
        F[s[:kₐ]]   = -μₐ + z[s[:kₐ]] + log((1. - δ) + Φ(exp(y[J[:iₖ]])))
        F[s[:hats]] = ρₛ * z[s[:hats]]
    end

    function ξ(F, z, y)
        mt                  = -log(β) - γ * (y[J[:cₖ]] + z[s[:hats]] - log(1. + Φ(exp(y[J[:iₖ]])) - δ))
        mtp1                = -γ * (y[J[:cₖ]] + z[s[:hats]])
        Y                   = exp(y[J[:cₖ]]) + exp(y[J[:iₖ]])
        DivQ                = α * Y - exp(y[J[:iₖ]]) + (Φ(exp(y[J[:iₖ]])) - δ) * exp(y[J[:q]])
        F[J[:cₖ]]           = log(Y) - (α - 1.) * z[s[:kₐ]]
        F[J[:iₖ]]           = -log(Φ′(exp(y[J[:iₖ]]))) - y[J[:q]]
        F[J[:log_D_plus_Q]] = log(DivQ + exp(y[J[:q]])) - y[J[:log_D_plus_Q]]
        F[J[:rf]]           = -y[J[:q]] - y[J[:log_E_RQ]]
        F[J[:q]]            = -mt - y[J[:q]]
        F[J[:log_E_RQ]]     = -mt - (-y[J[:rf]])
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Λ(F, z)
        F_type = eltype(F)
        F[s[:hats], SH[:εₐ]] = 1. / S * sqrt(1. - 2. * z[s[:hats]]) - 1.
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Σ(F, z)
        F_type = eltype(F)
        F[s[:kₐ], SH[:εₐ]]   = -σₐ
        F[s[:hats], SH[:εₐ]] = 0.
    end

    function ccgf(F, α, z)
        # F .= .5 * RiskAdjustedLinearizations.diag(α * α') # slower but this is the underlying math
        F .= .5 * sum(α.^2, dims = 2) # faster implementation
    end

    Γ₅ = zeros(T, Ny, Nz)
    Γ₅[J[:q],        s[:hats]] = -γ
    Γ₅[J[:log_E_RQ], s[:hats]] = -γ

    Γ₆ = zeros(T, Ny, Ny)
    Γ₆[J[:rf],       J[:log_D_plus_Q]] = 1.
    Γ₆[J[:q],        J[:cₖ]] = -γ
    Γ₆[J[:q],        J[:log_D_plus_Q]] = 1.
    Γ₆[J[:log_E_RQ], J[:cₖ]] = -γ

    z = vec(init_z)
    y = vec(init_y)
    Ψ = zeros(T, Ny, Nz)
    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nε)
end
