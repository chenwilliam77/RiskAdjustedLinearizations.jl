using UnPack, OrderedCollections, LinearAlgebra, JLD2, LabelledArrays, SparseArrays, RiskAdjustedLinearizations, NLsolve

if !isdefined(Main, :BansalYaron2004) || !isdefined(Main, :bansal_yaron_2004)
    include("bansal_yaron_2004.jl")
end

# BansalYaron2004 with heterogeneous risk aversion
# FIRST, we implement w/out extra forward expectations of the wealth share, to reduce the complexity.
mutable struct HetRiskAversion{T <: Real}
    p::LArray{T, 1, Array{T, 1}, (:μ_y, :ρ_x, :σ_x, :ρ_σ, :σ_y, :ς, :λ₁, :β, :ψ, :γ₁, :γ₂,
                                  :δ, :τ̅₁)}
    N_approx::LArray{Int64, 1, Array{Int64, 1}, (:q1, :q2, :ω1, :ω2)}
    S::OrderedDict{Symbol, Int}
    J::OrderedDict{Symbol, Int}
    E::OrderedDict{Symbol, Int}
    SH::OrderedDict{Symbol, Int}
end

# Parameters are based off Schorfheide et al. (2016) and converted from a monthly frequency to quarterly
function HetRiskAversion(; μ_y::T = 0.0016 * 3., ρ_x::T = 0.99^3, σ_x::T = sqrt((0.74 * sqrt(1. - ρ_x^2))^2 * 3.), ρ_σ::T = 0.99^3,
                         σ_y::T = sqrt(0.0021^2 * 3.), ς::T = sqrt(0.0014^2 * 3.), λ₁::T = .5,
                         β::T = 0.999^3, ψ::T = 2., γ₁::T = 8.9, γ₂::T = (9. - λ₁ * γ₁) / (1. - λ₁), # target 9 for average risk aversion
                         δ::T = .9, τ̅₁::T = .035,
                         N_approx::LArray{Int64, 1, Array{Int64, 1}, (:q1, :q1, :ω1, :ω2)} =
                         (@LArray [1, 1, 1, 1] (:q1, :q1, :ω1, :ω2))) where {T <: Real}

    @assert all(N_approx[k] > 0 for k in keys(N_approx)) "N_approx must be at least 1 for all variables."

    # Create indexing dictionaries
    S_init  = [:yy, :x, :σ²_y, :W1, :r₋₁, :b1₋₁, :s1₋₁]            # State variables
    J_init  = [:q, :v1, :v2, :ce1, :ce2, :ω1, :ω2, :r, :c1, :c2,   # Jump variables
               :b1, :b2, :s1, :s2, :Θ1, :Θ2, :logΘ1, :logΘ2,
               :Θ1_t1, :Θ2_t1]
    SH_init = [:ε_y, :ε_x, :ε_σ²]                                  # Exogenous shocks
    E_init  = [:value_fnct1, :value_fnct2,                         # Equations
               :certainty_equiv1, :certainty_equiv2,
               :ez_fwd_diff1, :ez_fwd_diff2,
               :euler1, :euler2, :cap_ret1, :cap_ret2,
               :budget_constraint1, :budget_constraint2,
               :wealth_per_agent1, :wealth_per_agent2,
               :eq_logΘ1, :eq_logΘ2,
               :expected_wealth_per_agent1,
               :expected_wealth_per_agent2,
               :consumption_mc, :bond_mc, :share_mc]
    for var in [:q1, :q2, :ω1, :ω2]
        inds = (var == :q) ? (1:N_approx[var]) : (0:(N_approx[var] - 1))
        push!(J_init, [Symbol(:d, var, "_t$(i)") for i in inds]...)
        push!(J_init, [Symbol(:p, var, "_t$(i)") for i in 1:N_approx[var]]...)
        push!(E_init, [Symbol(:eq_d, var, "_t$(i)") for i in inds]...)
        push!(E_init, [Symbol(:eq_p, var, "_t$(i)") for i in 1:N_approx[var]]...)
    end

    S  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(S_init))
    J  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(J_init))
    E  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(E_init))
    SH = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(SH_init))

    p = @LArray [μ_y, ρ_x, σ_x, ρ_σ, σ_y, ς, λ₁, β, ψ, γ₁, γ₂] (:μ_y, :ρ_x, :σ_x, :ρ_σ, :σ_y, :ς, :λ₁, :β, :ψ, :γ₁, :γ₂, :δ, :τ̅₁)

    return HetRiskAversion{T}(p, N_approx, S, J, E, SH)
end

function het_risk_aversion(m::HetRiskAversion{T}; sparse_arrays::Bool = false,
                           sparse_jacobian::Vector{Symbol} = Symbol[],
                           m_rep = nothing, algorithm::Symbol = :relaxation) where {T <: Real}

    # Unpack parameters and indexing dictionaries
    @unpack p, N_approx, S, J, E, SH = m

    @unpack yy, x, σ²_y, W1, r₋₁, b1₋₁, s1₋₁ = S
    @unpack q, v1, v2, ce1, ce2, ω1, ω2, r, c1, b1, s1 = J
    @unpack Θ1, Θ2, logΘ1, logΘ2, Θ1_t1, Θ2_t1, logΘ1, logΘ2 = J
    @unpack ε_y, ε_x, ε_σ² = SH
    @unpack value_fnct1, value_fnct2, certainty_equiv1, certainty_equiv2 = E
    @unpack ez_fwd_diff1, ez_fwd_diff2, cap_ret1, cap_ret2 = E
    @unpack budget_constraint1, budget_constraint2 = E
    @unpack wealth_per_agent1, wealth_per_agent2, eq_logΘ1, eq_logΘ2 = E
    @unpack expected_wealth_per_agent1, expected_wealth_per_agent2 = E
    @unpack consumption_mc, bond_mc, share_mc = E

    Nz = length(S)
    Ny = length(J)
    Nε = length(SH)

    ## Define nonlinear equations

    # Some helper functions
    m1_ξ(z, y) = log(p.β) + p.γ₁ * y[c1] - (p.ψ - p.γ₁) * y[ce1] - p.γ₁ * p.μ_y
    m2_ξ(z, y) = log(p.β) + p.γ₂ * y[c2] - (p.ψ - p.γ₂) * y[ce2] - p.γ₂ * p.μ_y
    function m1_fwd!(row, Γ₅, Γ₆)
        Γ₅[row, yy] = -p.γ₁
        Γ₆[row, c1] = -p.γ₁
        Γ₆[row, v1] = (p.ψ  - p.γ₁)
    end
    function m2_fwd!(row, Γ₅, Γ₆)
        Γ₅[row, yy] = -p.γ₂
        Γ₆[row, c2] = -p.γ₂
        Γ₆[row, v2]  = (p.ψ  - p.γ₂)
    end

    function μ(F, z, y) # note that y here refers to jump variables
        # Exogenous states
        F[yy]   = z[x]
        F[x]    = p.ρ_x * z[x]
        F[σ²_y] = (1. - p.ρ_σ) * p.σ²_y + p.ρ_σ * z[σ²_y]

        # Endogenous states
        F[W1]   = p.λ₁ * y[Θ1_t1]
        F[r₋₁]  = y[r]
        F[b1₋₁] = y[b1]
        F[s1₋₁] = y[s1]
    end

    function ξ(F, z, y)

        ## Preferences
        m1_ξv = m1_ξ(z, y) # evaluate SDF for type 1
        m2_ξv = m2_ξ(z, y) # evaluate SDF for type 2
        F[value_fnct1] = 1. / (1. - p.ψ) * (log(1. - p.β) + y[ω1]) - y[v1]
        F[value_fnct2] = 1. / (1. - p.ψ) * (log(1. - p.β) + y[ω2]) - y[v2]
        F[certainty_equiv1] = 1. / (1. - p.ψ) * (log(1. - p.β) - log(p.β) + log(exp(y[ω1]) - 1.)) - y[ce1]
        F[certainty_equiv2] = 1. / (1. - p.ψ) * (log(1. - p.β) - log(p.β) + log(exp(y[ω2]) - 1.)) - y[ce2]

        ## Euler equations
        F[euler1] = y[r] + m1_ξv
        F[euler2] = y[r] + m2_ξv

        ## Market-clearing equations and budget constraints
        Q = exp(y[q])
        C1 = exp(y[c1])
        C2 = exp(y[c2])
        B1 = -exp(y[B1]) # note that y[b1] = log(-B1) since B1 < 0
        B2 = exp(y[B2])
        S1 = exp(y[S1])
        S2 = exp(y[S2])
        F[consumption_mc] = log(p.λ₁ * C1 + (1 - p.λ₁) * C2)
        F[bond_mc]        = log(-p.λ₁ * B1) - log((1. - p.λ₁) * B2)
        F[share_mc]       = log(p.λ₁ * S1 + (1 - p.λ₁) * S2)

        λ₂ = 1. - p.λ₁
        F[budget_constraint1] = log(z[W1]) - log(p.λ₁) + log(1 + Q) - log(C1 + B1 + Q * S1)
        F[budget_constraint2] = log(1. - z[W1]) - log(λ₂) + log(1 + Q) - log(C2 + B2 + Q * S2)

        ## Wealth per agent (Θ) equations
        S1₋₁ = exp(z[s1₋₁])
        S2₋₁ = (1. - p.λ₁ * S1₋₁) / λ₂
        B1₋₁ = -exp(z[b1₋₁]) # note that z[b1₋₁] = log(-B1₋₁) since B1₋₁ < 0
        B2₋₁ = -p.λ₁ * B1₋₁ / λ₂
        R₋₁  = exp(z[r₋₁])
        τ₂   = -(p.τ̅₁ * (R₋₁ * B1₋₁ + (1 + Q) * S1₋₁) / (R₋₁ * B2₋₁ + (1. + Q) * S2₋₁))
        F[wealth_per_agent1] = log(1. - p.τ̅₁) + log(S1₋₁ + R₋₁ * B1₋₁ / (1. + Q)) - log(y[Θ1])
        F[wealth_per_agent2] = log(R₋₁ * B2₋₁ / (1. + Q) + S2₋₁ - p.τ̅₁ * (R₋₁ * B1₋₁ / (1. + Q) + S1₋₁)) - log(y[Θ2])
                             # log(1. - τ₂) + log(S2₋₁ + R₋₁ * B2₋₁ / (1. + Q)) - log(y[Θ2])
        F[eq_logΘ1]          = y[logΘ1] - log(y[Θ1])
        F[eq_logΘ2]          = y[logΘ2] - log(y[Θ2])
        F[expected_wealth_per_agent1] = -log(y[Θ1_t1])
        F[expected_wealth_per_agent2] = -log(y[Θ2_t1])

        ## Forward-difference equations separately handled b/c recursions
        F[cap_ret1]     = y[q] - log(sum([exp(y[J[Symbol("dq1_t$(i)")]]) for i in 1:N_approx[:q1]]) +
                                     exp(y[J[Symbol("pq1_t$(N_approx[:q1])")]]))
        F[cap_ret2]     = y[q] - log(sum([exp(y[J[Symbol("dq2_t$(i)")]]) for i in 1:N_approx[:q2]]) +
                                     exp(y[J[Symbol("pq2_t$(N_approx[:q2])")]]))
        F[ez_fwd_diff1] = y[ω] - log(sum([exp(y[J[Symbol("dω1_t$(i)")]]) for i in 0:(N_approx[:ω1] - 1)]) +
                                     exp(y[J[Symbol("pω1_t$(N_approx[:ω1])")]]))
        F[ez_fwd_diff2] = y[ω] - log(sum([exp(y[J[Symbol("dω2_t$(i)")]]) for i in 0:(N_approx[:ω2] - 1)]) +
                                     exp(y[J[Symbol("pω2_t$(N_approx[:ω2])")]]))

        # Set initial boundary conditions
        F[E[:eq_dq1_t1]] = p.μ_y - y[J[:dq1]] + m1_ξv
        F[E[:eq_pq1_t1]] = p.μ_y - y[J[:pq1]] + m1_ξv
        F[E[:eq_dq2_t1]] = p.μ_y - y[J[:dq2]] + m2_ξv
        F[E[:eq_pq2_t1]] = p.μ_y - y[J[:pq2]] + m2_ξv
        F[E[:eq_dω1_t0]] = y[J[:dω1_t0]]
        F[E[:eq_pω1_t1]] = p.μ_y - y[J[:pω1_t1]] + m1_ξv
        F[E[:eq_dω1_t0]] = y[J[:dω2_t0]]
        F[E[:eq_pω1_t1]] = p.μ_y - y[J[:pω2_t1]] + m2_ξv

        # Recursions for forward-difference equations
        for i in 2:N_approx[:q1]
            F[E[Symbol("eq_dq1_t$(i)")]] = p.μ_y - y[J[Symbol("dq1_t$(i)")]] + m1_ξv
            F[E[Symbol("eq_pq1_t$(i)")]] = p.μ_y - y[J[Symbol("pq1_t$(i)")]] + m1_ξv
        end
        for i in 2:N_approx[:q2]
            F[E[Symbol("eq_dq2_t$(i)")]] = p.μ_y - y[J[Symbol("dq2_t$(i)")]] + m2_ξv
            F[E[Symbol("eq_pq2_t$(i)")]] = p.μ_y - y[J[Symbol("pq2_t$(i)")]] + m2_ξv
        end
        for i in 2:N_approx[:ω1]
            F[E[Symbol("eq_dω1_t$(i-1)")]] = p.μ_y - y[J[Symbol("dω1_t$(i-1)")]] + m1_ξv
            F[E[Symbol("eq_pω1_t$(i)")]]   = p.μ_y - y[J[Symbol("pω1_t$(i)")]]   + m1_ξv
        end
        for i in 2:N_approx[:ω1]
            F[E[Symbol("eq_dω2_t$(i-1)")]] = p.μ_y - y[J[Symbol("dω2_t$(i-1)")]] + m2_ξv
            F[E[Symbol("eq_pω2_t$(i)")]]   = p.μ_y - y[J[Symbol("pω2_t$(i)")]]   + m2_ξv
        end
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Λ(F, z)
        # Heteroskedastic risk arises b/c the wealth share is a state variable
        F[W1, Θ1] = p.λ₁
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Σ(F, z)
        F[yy,   ε_y]  = sqrt(z[σ²_y])         # Take square root b/c Σ is not variance-covariance matrix.
        F[x,    ε_x]  = sqrt(z[σ²_y]) * p.σ_x # It is instead the "volatility" loading on the martingale difference sequences.
        F[σ²_y, ε_σ²] = sqrt(z[σ²_y]) * p.ς
    end

    function ccgf(F, A, z)
        # F .= .5 * diag(α * α') # slower but this is the underlying math
        sum!(F, A.^2) # faster implementation
        F .*= .5
    end

    if sparse_arrays
        Γ₅ = spzeros(T, Ny, Nz)
        Γ₆ = spzeros(T, Ny, Ny)
    else
        Γ₅ = zeros(T, Ny, Nz)
        Γ₆ = zeros(T, Ny, Ny)
    end

    # Forward difference equations: boundary conditions
    m1_fwd!(E[:eq_dq1_t1], Γ₅, Γ₆)
    Γ₅[E[:eq_dq1_t1], yy] = one(T)

    m1_fwd!(E[:eq_pq1_t1], Γ₅, Γ₆)
    Γ₅[E[:eq_pq1_t1], yy] = one(T)
    Γ₆[E[:eq_pq1_t1], q]  = one(T)

    m1_fwd!(E[:eq_pω1_t1], Γ₅, Γ₆)
    Γ₅[E[:eq_pω1_t1], yy] = one(T)
    Γ₆[E[:eq_pω1_t1], c1] = one(T)
    Γ₆[E[:eq_pω1_t1], ω1] = one(T)

    m2_fwd!(E[:eq_dq1_t1], Γ₅, Γ₆)
    Γ₅[E[:eq_dq2_t1], yy] = one(T)

    m2_fwd!(E[:eq_pq2_t1], Γ₅, Γ₆)
    Γ₅[E[:eq_pq2_t1], yy] = one(T)
    Γ₆[E[:eq_pq2_t1], q]  = one(T)

    m2_fwd!(E[:eq_pω2_t1], Γ₅, Γ₆)
    Γ₅[E[:eq_pω2_t1], yy] = one(T)
    Γ₆[E[:eq_pω2_t1], c2] = one(T)
    Γ₆[E[:eq_pω2_t1], ω2] = one(T)

    # Forward difference equations: recursions
    for i in 2:N_approx[:q1]
        m1_fwd!(E[Symbol("eq_dq1_t$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_dq1_t$(i)")], yy] = one(T)
        Γ₆[E[Symbol("eq_dq1_t$(i)")], J[Symbol("dq1_t$(i-1)")]] = one(T)

        m1_fwd!(E[Symbol("eq_pq1_t$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_pq1_t$(i)")], yy] = one(T)
        Γ₆[E[Symbol("eq_pq1_t$(i)")], J[Symbol("pq1_t$(i-1)")]] = one(T)
    end
    for i in 2:N_approx[:ω1]
        m1_fwd!(E[Symbol("eq_dω1_t$(i-1)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_dω1_t$(i-1)")], yy] = one(T)
        Γ₆[E[Symbol("eq_dω1_t$(i-1)")], c1] = one(T)
        Γ₆[E[Symbol("eq_dω1_t$(i-1)")], J[Symbol("dω1_t$(i-2)")]] = one(T)

        m1_fwd!(E[Symbol("eq_pω1_t$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_pω1_t$(i)")], yy] = one(T)
        Γ₆[E[Symbol("eq_pω1_t$(i)")], c1] = one(T)
        Γ₆[E[Symbol("eq_pω1_t$(i)")], J[Symbol("pω1_t$(i-1)")]] = one(T)
    end
    for i in 2:N_approx[:q2]
        m1_fwd!(E[Symbol("eq_dq2_t$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_dq2_t$(i)")], yy] = one(T)
        Γ₆[E[Symbol("eq_dq2_t$(i)")], J[Symbol("dq2_t$(i-1)")]] = one(T)

        m1_fwd!(E[Symbol("eq_pq2_t$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_pq2_t$(i)")], yy] = one(T)
        Γ₆[E[Symbol("eq_pq2_t$(i)")], J[Symbol("pq2_t$(i-1)")]] = one(T)
    end
    for i in 2:N_approx[:ω2]
        m1_fwd!(E[Symbol("eq_dω2_t$(i-1)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_dω2_t$(i-1)")], yy] = one(T)
        Γ₆[E[Symbol("eq_dω2_t$(i-1)")], c2] = one(T)
        Γ₆[E[Symbol("eq_dω2_t$(i-1)")], J[Symbol("dω2_t$(i-2)")]] = one(T)

        m1_fwd!(E[Symbol("eq_pω2_t$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_pω2_t$(i)")], yy] = one(T)
        Γ₆[E[Symbol("eq_pω2_t$(i)")], c2] = one(T)
        Γ₆[E[Symbol("eq_pω2_t$(i)")], J[Symbol("pω2_t$(i-1)")]] = one(T)
    end

    z, y, Ψ = create_guess(m, (isnothing(m_rep) ?
                               BansalYaron2004(μ_y = p.μ_y, ρ_x = p.ρ_x, σ_x = p.σ_x,
                                               ρ_σ = p.ρ_sigma, σ_y = p.σ_y, ς = p.ς,
                                               β = p.β, ψ = p.ψ, γ = p.λ₁ * p.γ₁ + (1. - p.λ₁) * p.γ₂,
                                               N_approx = (@LArray N_approx[[:q1, :ω1]] (:q, :ω))) : m_rep); algorithm = algorithm)

    if sparse_arrays
        return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, vec(z), vec(y), Ψ, Nε; sparse_jacobian = sparse_jacobian,
                                         Λ_Σ_cache_init = dims -> spzeros(dims...))
    else
        return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, vec(z), vec(y), Ψ, Nε; sparse_jacobian = sparse_jacobian)
    end
end

function create_guess(m::HetRiskAversion{T}, m_rep::BansalYaron2004; algorithm::Symbol = :relaxation) where {T <: Real}

    ## Set up

    # Get guesses from BansalYaron2004 representative agent model
    ral_rep = bansal_yaron_2004(m_rep)
    solve!(ral_rep, algorithm = algorithm, verbose = :none)
    yrep = ral_rep.y
    zrep = ral_rep.z
    Ψrep = ral_rep.Ψ

    # Unpack parameters and indexing dictionaries
    @unpack p, N_approx, S, J, E, SH = m
    @unpack yy, x, σ²_y, W1, r₋₁, b1₋₁, s1₋₁ = S
    @unpack q, v1, v2, ce1, ce2, ω1, ω2, r, c1, b1, s1 = J
    @unpack Θ1, Θ2, logΘ1, logΘ2, Θ1_t1, Θ2_t1, logΘ1, logΘ2 = J

    # Initialize deterministic steady state guess vectors
    z = Vector{T}(undef, Nz)
    y = Vector{T}(undef, Ny)

    ## Compute guesses

    # Steady state values of state variables known ex-ante
    z[yy]   = 0.
    z[x]    = 0.
    z[σ²_y] = p.σ²_y

    # Guesses based on representative agent model's solution
    y[ω1]  = yrep[m_rep.J[:ω]]
    y[ω2]  = yrep[m_rep.J[:ω]]
    y[q]   = yrep[m_rep.J[:q]]
    y[v1]  = yrep[m_rep.J[:v]]
    y[ce1] = yrep[m_rep.J[:ce]]
    y[v2]  = yrep[m_rep.J[:v]]
    y[ce2] = yrep[m_rep.J[:ce]]

    # Guesses for consumption and portfolio choice
    S1 = (1. / p.γ₁) / (1. / p.γ₁ + 1. / p.γ₂)
    S2 = (1. - p.λ₁ * S1) / (1. - p.λ₁)
    C1 = S1
    C2 = (1. - p.λ₁ * C1) / (1. - p.λ₁)
    M0 = m_rep.p.β * (yrep[m_rep.J[:v]] / yrep[m_rep.J[:ce]])^(m_rep.p.ψ - m_rep.p.γ) *
        exp(-m_rep.p.γ * m_rep.p.μ_y) # make a guess for SDF
    R = 1. / M0 # guess that the interest rate is just 1 / M0
    Q = exp(y_rep[m_rep.J[:q]])
    τ₂ = -(p.τ₁ * (R * B1 + (1 + Q) * S1)) / (R * B2 + (1 + Q) * S2)
    B1 = (S1 * ((1. - p.τ̅₁) - p.τ̅₁ * Q) - C1) / (1. - (1. - p.τ̅₁) * R)
    B2 = -p.λ₁ * B1 / (1. - p.λ₁)
    Θ10 = (1. - p.τ̅₁) * (S1 + R * B1 / (1. + Q))
    Θ20 = (1. - τ₂) * (S2 + R * B2 / (1. + Q))

    y[s1] = log(S1)
    y[s2] = log(S2)
    y[c1] = log(C1)
    y[c2] = log(C2)
    y[b1] = log(-B1) # agent 1 leveraged -> B10 < 0
    y[b2] = log(B2)
    y[Θ1] = Θ10
    y[Θ2] = Θ20
    y[logΘ1] = log(Θ10)
    y[logΘ2] = log(Θ20)
    y[Θ1_t1] = Θ10 # in a steady state, the one-period ahead expectation agrees with the current day value.
    y[Θ2_t1] = Θ20
    y[r] = log(R)
    z[W1] = p.λ₁ * Θ10
    z[r₋₁] = y[r]
    z[b1₋₁] = y[b1]
    z[s1₋₁] = y[s1]

    M10 = m_rep.p.β * (yrep[m_rep.J[:v]] / yrep[m_rep.J[:ce]])^(m_rep.p.ψ - p.γ₁) * # use agent 1's risk aversion here
        exp(-m_rep.p.γ * m_rep.p.μ_y) # make a guess for SDF
    M20 = m_rep.p.β * (yrep[m_rep.J[:v]] / yrep[m_rep.J[:ce]])^(m_rep.p.ψ - p.γ₂) * # use agent 1's risk aversion here
        exp(-m_rep.p.γ * m_rep.p.μ_y) # make a guess for SDF
    Y0 = 1. # steady state growth rate in endowment
    y[J[:dq1_t1]] = convert(T, log(exp(p.μ_y) * M10 * Y0))
    y[J[:pq1_t1]] = convert(T, log(exp(p.μ_y) * M10 * Y0 * Q))
    y[J[:dω1_t0]] = zero(T)
    y[J[:pω1_t1]] = convert(T, log(exp(p.μ_y) * M10 * Y0 * exp(y[ω1])))
    y[J[:dq2_t1]] = convert(T, log(exp(p.μ_y) * M20 * Y0))
    y[J[:pq2_t1]] = convert(T, log(exp(p.μ_y) * M20 * Y0 * Q))
    y[J[:dω2_t0]] = zero(T)
    y[J[:pω2_t1]] = convert(T, log(exp(p.μ_y) * M20 * Y0 * exp(y[ω2])))

    for i in 2:N_approx[:q1]
        y[J[Symbol("dq1_t$(i)")]] = convert(T, log(M10) + μ_y + log(Y0) + y[J[Symbol("dq1_t$(i-1)")]])
        y[J[Symbol("pq1_t$(i)")]] = convert(T, log(M10) + μ_y + log(Y0) + y[J[Symbol("pq1_t$(i-1)")]])
    end
    for i in 2:N_approx[:q2]
        y[J[Symbol("dq2_t$(i)")]] = convert(T, log(M20) + μ_y + log(Y0) + y[J[Symbol("dq2_t$(i-1)")]])
        y[J[Symbol("pq2_t$(i)")]] = convert(T, log(M20) + μ_y + log(Y0) + y[J[Symbol("pq2_t$(i-1)")]])
    end
    for i in 2:N_approx[:ω1]
        y[J[Symbol("dω1_t$(i-1)")]] = convert(T, μ_y + log(M10) + log(Y0) + y[J[Symbol("dω1_t$(i-2)")]])
        y[J[Symbol("pω1_t$(i)")]] = convert(T, μ_y + log(M10) + log(Y0) + y[J[Symbol("pω1_t$(i-1)")]])
    end
    for i in 2:N_approx[:ω2]
        y[J[Symbol("dω2_t$(i-1)")]] = convert(T, μ_y + log(M20) + log(Y0) + y[J[Symbol("dω2_t$(i-2)")]])
        y[J[Symbol("pω2_t$(i)")]] = convert(T, μ_y + log(M20) + log(Y0) + y[J[Symbol("pω2_t$(i-1)")]])
    end

    return z, y, Ψ
end
