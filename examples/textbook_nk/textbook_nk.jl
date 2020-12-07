using UnPack, OrderedCollections, ForwardDiff, JLD2

mutable struct TextbookNK{T <: Real}
    β::T
    σ::T
    ψ::T
    η::T
    ϵ::T
    ϕ::T
    ρₐ::T
    σₐ::T
    ρᵢ::T
    σᵢ::T
    ϕ_π::T
    π̃_ss::T
end

function TextbookNK(; β::T = .99, σ::T = 2., ψ::T = 1., η::T = 1., ϵ::T = 4.45, ϕ::T = .7,
                    ρₐ::T = 0.9, σₐ::T = .004, ρᵢ::T = .7, σᵢ::T = .025 / 4.,
                    ϕ_π::T = 1.5, π̃_ss::T = 0.) where {T <: Real}
    return TextbookNK{T}(β, σ, ψ, η, ϵ, ϕ, ρₐ, σₐ, ρᵢ, σᵢ, ϕ_π, π̃_ss)
end

function textbook_nk(m::TextbookNK{T}) where {T <: Real}
    @unpack β, σ, ψ, η, ϵ, ϕ, ρₐ, σₐ, ρᵢ, σᵢ, ϕ_π, π̃_ss = m
    ĩ_ss = π̃_ss - log(β)

    # On notation: x̃ = log(1 + x); x′ = 𝔼ₜ[xₜ₊₁]
    S  = OrderedDict{Symbol, Int}(:a => 1, :ĩ₋₁ => 2, :v₋₁ => 3, :i_sh => 4) # State Variables
    J  = OrderedDict{Symbol, Int}(:c => 1, :π̃ => 2, :n => 3, :w => 4, :mc => 5, :v => 6,
                                  :x₁ => 7, :x₂ => 8, :ĩ => 9) # Jump variables
    E  = OrderedDict{Symbol, Int}(:euler => 1, :mrs => 2, :eq_mc => 3, :output => 4,
                                  :dispersion => 5, :phillips_curve => 6, :eq_x₁ => 7,
                                  :eq_x₂ => 8, :eq_mp => 9) # Equations
    SH = OrderedDict{Symbol, Int}(:εₐ => 1, :εᵢ => 2) # Exogenous shocks

    @unpack a, ĩ₋₁, v₋₁, i_sh = S
    @unpack c, π̃, n, w, mc, v, x₁, x₂, ĩ = J
    @unpack euler, mrs, eq_mc, output, dispersion, phillips_curve, eq_x₁, eq_x₂, eq_mp = E
    @unpack εₐ, εᵢ = SH

    Nz = length(S)
    Ny = length(J)
    Nε = length(SH)

    function μ(F, z, y)
        F_type  = eltype(F)
        F[a]    = ρₐ * z[a]
        F[ĩ₋₁]  = y[ĩ]
        F[v₋₁]  = y[v]
        F[i_sh] = zero(F_type)
    end

    function ξ(F, z, y)
        F_type = eltype(F)
        π̃_star = log(ϵ / (ϵ - 1.)) + y[π̃] + (y[x₁] - y[x₂])
        F[euler] = log(β) + σ * y[c] + y[ĩ]
        F[mrs] = log(ψ) + η * y[n] - (-σ * y[n] + y[w])
        F[eq_mc] = y[w] - (z[a] + y[mc])
        F[output] = y[c] - (z[a] + y[n] - y[v])
        F[dispersion] = y[v] - (ϵ * y[π̃] + log((1. - ϕ) * exp(π̃_star)^(-ϵ) + ϕ * exp(z[v₋₁])))
        F[phillips_curve] = (1. - ϵ) * y[π̃] - log((1. - ϕ) * exp(π̃_star)^(1 - ϵ) + ϕ)
        F[eq_x₁] = log(ϕ) + log(β) - log(exp(y[x₁]) - exp((1. - σ) * y[c] + y[mc]))
        F[eq_x₂] = log(ϕ) + log(β) - log(exp(y[x₂]) - exp((1. - σ) * y[c]))
        F[eq_mp] = y[ĩ] - ((1. - ρᵢ) * ĩ_ss + ρᵢ * z[ĩ₋₁]  + (1 - ρᵢ) * ϕ_π * (y[π̃] - π̃_ss) + z[i_sh])
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    Λ = zeros(T, Nz, Ny)

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Σ(F, z)
        F[a, εₐ]    = σₐ
        F[i_sh, εᵢ] = σᵢ
    end

    function ccgf(F, α, z)
        # F .= .5 * RiskAdjustedLinearizations.diag(α * α') # slower but this is the underlying math
        F .= vec(.5 * sum(α.^2, dims = 2)) # faster implementation
    end

    Γ₅ = zeros(T, Ny, Nz)

    Γ₆ = zeros(T, Ny, Ny)
    Γ₆[euler, c] = -σ
    Γ₆[euler, π̃] = -one(T)
    Γ₆[eq_x₁, x₁] = one(T)
    Γ₆[eq_x₁, π̃] = one(T)
    Γ₆[eq_x₂, x₂] = one(T)
    Γ₆[eq_x₂, π̃] = one(T)

    Ψ = zeros(T, Ny, Nz)

    # Deterministic steady state as initial guess

    # z
    a0    = 0.
    ĩ₋₁0  = ĩ_ss
    v₋₁0  = 0.
    i_sh0 = 0.
    z     = [a0, ĩ₋₁0, v₋₁0, i_sh0]

    # y
    ĩ0  = ĩ_ss
    π̃0  = π̃_ss
    v0  = 0.
    mc0 = log((ϵ - 1.) / ϵ)
    x₁0 = 1.2 + mc0
    x₂0 = 1.2
    n0  = (1 / (η + σ)) * log(1. / ψ * (exp(v0))^σ * exp(mc0))
    c0  = n0 - v0
    w0  = a0 + mc0
    y   = [c0, π̃0, n0, w0, mc0, v0, x₁0, x₂0, ĩ0]

    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, vec(z), vec(y), Ψ, Nε)
end
