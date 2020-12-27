using UnPack, OrderedCollections, ForwardDiff, JLD2

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

function rbc_cc(m::RBCCampbellCochraneHabits{T}, n_strips::Int = 0) where {T <: Real}
    @unpack IK̄, β, δ, α, ξ₃, μₐ, σₐ, γ, ρₛ, S = m

    s  = OrderedDict{Symbol, Int}(:kₐ => 1,  :hats => 2) # State variables
    J  = OrderedDict{Symbol, Int}(:cₖ => 1, :iₖ => 2, :log_D_plus_Q => 3, :rf => 4, :q => 5, :log_E_RQ => 6) # Jump variables
    SH = OrderedDict{Symbol, Int}(:εₐ => 1) # Exogenous shocks

    if n_strips > 0
        J[:div]   = 7
        J[:r_div] = 8
        J[:wres]  = 9
        q_div_vec = Symbol[Symbol("q_div$i") for i in 1:n_strips]
        for (i, q_div) in enumerate(q_div_vec)
            J[q_div] = i + 9
        end
    end

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

    ξ = if n_strips > 0
        function _ξ_withstrips(F, z, y)
            mt                  = -log(β) - γ * (y[J[:cₖ]] + z[s[:hats]] - log(1. + Φ(exp(y[J[:iₖ]])) - δ))
            mtp1                = -γ * (y[J[:cₖ]] + z[s[:hats]])
            Y                   = exp(y[J[:cₖ]]) + exp(y[J[:iₖ]])
            DivQ                = α * Y - exp(y[J[:iₖ]]) + (Φ(exp(y[J[:iₖ]])) - δ) * exp(y[J[:q]])
            F[J[:cₖ]]           = log(Y) - (α - 1.) * z[s[:kₐ]]
            F[J[:iₖ]]           = -log(Φ′(exp(y[J[:iₖ]]))) - y[J[:q]]
            F[J[:log_D_plus_Q]] = log(DivQ + exp(y[J[:q]])) - y[J[:log_D_plus_Q]]
            F[J[:rf]]           = -y[J[:q]] - y[J[:log_E_RQ]]
            F[J[:q]]            = log(sum(exp.(y[J[:q_div1]:end])) + exp(y[J[:r_div]])) - y[J[:q]]
            F[J[:log_E_RQ]]     = log(DivQ) - y[J[:div]]
            F[J[:div]]          = log(exp(y[J[Symbol("q_div$(n_strips)")]]) + exp(y[J[:r_div]])) - y[J[:wres]]
            F[J[:r_div]]        = -mt - y[J[:r_div]]
            F[J[:wres]]         = -mt - (-y[J[:rf]])

            for i in 1:n_strips
                F[J[Symbol("q_div$(i)")]] = -mt - y[J[Symbol("q_div$(i)")]]
            end
        end
    else
        function _ξ_nostrips(F, z, y)
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
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Λ(F, z)
        F_type = eltype(F)
        F[s[:hats], J[:cₖ]] = 1. / S * sqrt(1. - 2. * z[s[:hats]]) - 1.
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Σ(F, z)
        F_type = eltype(F)
        F[s[:kₐ], SH[:εₐ]]   = -σₐ
        F[s[:hats], SH[:εₐ]] = 0.
    end

    function ccgf(F, α, z)
        # F .= .5 * RiskAdjustedLinearizations.diag(α * α') # slower but this is the underlying math
        F .= vec(.5 * sum(α.^2, dims = 2)) # faster implementation
    end

    Γ₅ = zeros(T, Ny, Nz)
    if n_strips > 0
        Γ₅[J[:r_div],    s[:hats]] = -γ
        Γ₅[J[:wres],     s[:hats]] = -γ
    else
        Γ₅[J[:q],        s[:hats]] = -γ
        Γ₅[J[:log_E_RQ], s[:hats]] = -γ
    end

    Γ₆ = zeros(T, Ny, Ny)
    Γ₆[J[:rf], J[:log_D_plus_Q]] = 1.

    if n_strips > 0
        Γ₆[J[:r_div],    J[:cₖ]]   = -γ
        Γ₆[J[:r_div],    J[:wres]] = 1.
        Γ₆[J[:wres],     J[:cₖ]]   = -γ

        Γ₅[J[:q_div1], s[:hats]] = -γ
        Γ₆[J[:q_div1], J[:cₖ]]   = -γ
        Γ₆[J[:q_div1], J[:div]]  = 1.

        if n_strips > 1
            for i in 2:n_strips
                Γ₅[J[Symbol("q_div$(i)")], s[:hats]]                   = -γ
                Γ₆[J[Symbol("q_div$(i)")], J[:cₖ]]                     = -γ
                Γ₆[J[Symbol("q_div$(i)")], J[Symbol("q_div$(i - 1)")]] = 1.
            end
        end
    else
        Γ₆[J[:q],        J[:cₖ]]           = -γ
        Γ₆[J[:q],        J[:log_D_plus_Q]] = 1.
        Γ₆[J[:log_E_RQ], J[:cₖ]]           = -γ
    end

    if n_strips > 0
        z = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "rbccc_dss_N3_output.jld2"), "r")["z_dss"]
        y = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "rbccc_dss_N3_output.jld2"), "r")["y_dss"]

        if n_strips > 3
            y = vcat(y, -ones(n_strips - 3))
        elseif n_strips < 3
            y = y[1:end - (3 - n_strips)]
        end
    else
        z = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "rbccc_det_ss_output.jld2"), "r")["z_dss"]
        y = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "..", "test", "reference", "rbccc_det_ss_output.jld2"), "r")["y_dss"]
    end
    Ψ = zeros(T, Ny, Nz)

    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, vec(z), vec(y), Ψ, Nε)
end
