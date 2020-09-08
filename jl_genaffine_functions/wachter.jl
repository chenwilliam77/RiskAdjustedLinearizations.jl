using UnPack, OrderedCollections

# TODO
# 1. Create model object
# 2. Run code in MATLAB and generate comparison test output for all functions, say at the deterministic steady state
# 3. Test ForwardDiff works properly for all cases, for both in place and non in place models
# 4. Label equations for μ, ξ so creating Γ₅ and Γ₆ makes more sense
#=function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int = -1) where {T <: Number, M <: Function, L <: Function,
                                                        S <: Function, X <: Function, CF <: Function}
end=#
mutable struct WachterDisasterRisk{T <: Real}
    μₐ::T
    σₐ::T
    ν::T
    δ::T
    ρₚ::T
    pp::T
    ϕₚ::T
    ρ::T
    γ::T
    β::T
end

function WachterDisasterRisk(; μₐ::T = .0252 / 4., σₐ::T = .02 / sqrt(4.), ν::T = .3, δ::T = 0., ρₚ::T = .08^(1. / 4.), pp::T = .0355 / 4.,
                             ϕₚ::T = .0114 / 4. / (.02 / sqrt(4.)) / sqrt(.0355 / 4.), ρ::T = 2.0, γ::T = 3.0,
                             β::T = exp(-.012 / 4.)) where {T <: Real}
    return WachterDisasterRisk{T}(μₐ, σₐ, ν, δ, ρₚ, pp, ϕₚ, ρ, γ, β)
end

function inplace_wachter_disaster_risk(m::WachterDisasterRisk{T}) where {T <: Real}
    @unpack μₐ, σₐ, ν, δ, ρₚ, pp, ϕₚ, ρ, γ, β = m

    @assert ρ != 1. # Forcing ρ to be non-unit for this example

    S  = OrderedDict{Symbol, Int}(:p => 1,  :εc => 2, :εξ => 3) # State variables
    J  = OrderedDict{Symbol, Int}(:vc => 1, :xc => 2, :rf => 3) # Jump variables
    SH = OrderedDict{Symbol, Int}(:εₚ => 1, :εc => 2, :εξ => 3) # Exogenous shocks
    Nz = length(S)
    Ny = length(J)
    Nε = length(SH)

    function μ(F, z, y) # the equation labeling could be wrong
        F_type    = eltype(F)
        F[S[:p]]  = (1 - ρₚ) * pp + ρₚ * z[S[:p]]
        F[S[:εc]] = zero(F_type)
        F[S[:εξ]] = zero(F_type)
    end

    function ξ(F, z, y) # the equation labeling could be wrong
        F[J[:vc]] = log(β) - γ * μₐ + γ * ν * z[S[:p]] - (ρ - γ) * y[J[:xc]] + y[J[:rf]]
        F[J[:xc]] = log(1. - β + β * exp((1. - ρ) * y[J[:xc]])) - (1. - ρ) * y[J[:vc]]
        F[J[:rf]] = (1. - γ) * (μₐ - ν * z[S[:p]] - y[J[:xc]])
    end

    Λ = zeros(T, Nz, Ny)

    function Σ(F, z)
        F_type = eltype(F)
        F[SH[:εₚ], SH[:εₚ]] = sqrt(z[S[:p]]) * ϕₚ * σₐ
        F[SH[:εc], SH[:εc]] = one(F_type)
        F[SH[:εξ], SH[:εξ]] = one(F_type)
    end

    function ccgf(F, α, z)
        F .= .5 .* α[:, 1].^2 + .5 * α[:, 2].^2 + (exp.(α[:, 3] + α[:, 3].^2 .* δ^2 ./ 2.) .- 1. - α[:, 3]) * z[S[:p]]
    end

    Γ₅ = zeros(T, Ny, Nz)
    Γ₅[J[:vc], S[:εc]] = (-γ * σₐ)
    Γ₅[J[:vc], S[:εξ]] = (γ * ν)
    Γ₅[J[:rf], S[:εc]] = (1. - γ) * σₐ
    Γ₅[J[:rf], S[:εξ]] = -(1. - γ) * ν

    Γ₆ = zeros(T, Ny, Ny)
    Γ₆[J[:vc], J[:vc]] = (ρ - γ)
    Γ₆[J[:rf], J[:vc]] = (1. - γ)

    z = [pp, 0., 0.]
    xc_sss = log((1. - β) / (exp((1. - ρ) * (ν * pp - μₐ)) - β)) / (1. - ρ)
    vc_sss = xc_sss + ν * pp - μₐ
    y = [vc_sss, xc_sss, -log(β) + γ * (μₐ - ν * pp) - (ρ - γ) * (vc_sss - xc_sss)]
    Ψ = zeros(T, Ny, Nz)
    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nε)
end

function outofplace_wachter_disaster_risk(m::WachterDisasterRisk{T}) where {T}
    @unpack μₐ, σₐ, ν, δ, ρₚ, pp, ϕₚ, ρ, γ, β = m

    @assert ρ != 1. # Forcing ρ to be non-unit for this example

    S  = OrderedDict{Symbol, Int}(:p => 1,  :εc => 2, :εξ => 3) # State variables
    J  = OrderedDict{Symbol, Int}(:vc => 1, :xc => 2, :rf => 3) # Jump variables
    SH = OrderedDict{Symbol, Int}(:εₚ => 1, :εc => 2, :εξ => 3) # Exogenous shocks
    Nz = length(S)
    Ny = length(J)
    Nε = length(SH)

    function μ(z, y) # the equation labeling could be wrong
        F = Vector{T}(undef, Nz)
        F[S[:p]]  = (1 - ρₚ) * pp + ρₚ * z[S[:p]]
        F[S[:εc]] = zero(T)
        F[S[:εξ]] = zero(T)
        return F
    end

    function ξ(z, y) # the equation labeling could be wrong
        F = Vector{T}(undef, Ny)
        F[J[:vc]] = log(β) - γ * μₐ + γ * ν * z[S[:p]] - (ρ - γ) * y[J[:xc]] + y[J[:rf]]
        F[J[:xc]] = log(1. - β + β * exp((1. - ρ) * J[:xc])) - (1. - ρ) * J[:vc]
        F[J[:rf]] = (1. - γ) * (μₐ - ν * z[S[:p]] - y[J[:xc]])
        return F
    end

    Λ = zeros(T, Nz, Nz)

    function Σ(z)
        F = zeros(Nz, Nz)
        F[SH[:εₚ], SH[:εₚ]] = sqrt(z[S[:p]] * ϕₚ * σₐ)
        F[SH[:εc], SH[:εc]] = 1.
        F[SH[:εξ], SH[:εξ]] = 1.
        return F
    end

    ccgf(α, z) = .5 * α[:, 1].^2 + .5 * α[:, 2].^2 + (eαp(α[:, 3] + α[:, 3].^2 * δ^2 / 2.) - 1. - α[:, 3]) * z[S[:p]]

    Γ₅ = zeros(T, Ny, Nz)
    Γ₅[J[:vc], S[:εc]] = (-γ * σₐ)
    Γ₅[J[:vc], S[:εξ]] = (γ * ν)
    Γ₅[J[:rf], S[:εc]] = (1. - γ) * σₐ
    Γ₅[J[:rf], S[:εξ]] = -(1. - γ) * ν

    Γ₆ = zeros(T, Ny, Ny)
    Γ₆[J[:vc], J[:vc]] = (ρ - γ)
    Γ₆[J[:rf], J[:vc]] = (1. - γ)

    z = [pp, 0., 0.]
    xc_sss = log((1. - β) / (exp((1. - ρ) * (ν * pp - μ)) - β)) / (1. - ρ)
    vc_sss = xc_sss + ν * pp - μ
    y = [vc_sss, xc_sss, -log(β) + γ * (μ - ν * pp) - (ρ - γ) * (vc_sss - xc_sss)]
    Ψ = zeros(T, Ny, Nz)
    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nε)
end
