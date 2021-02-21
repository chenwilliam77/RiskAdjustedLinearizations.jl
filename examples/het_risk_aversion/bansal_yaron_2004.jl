using UnPack, OrderedCollections, LinearAlgebra, JLD2, LabelledArrays, SparseArrays

# Representative agent version, which is essentially Bansal and Yaron (2004)
mutable struct BansalYaron2004{T <: Real}
    p::LArray{T, 1, Array{T, 1}, (:μ_y, :ρ_x, :σ_x, :ρ_σ, :σ_y, :ς, :β, :ψ, :γ)}
    N_approx::LArray{Int64, 1, Array{Int64, 1}, (:q, :ω)}
    S::OrderedDict{Symbol, Int}
    J::OrderedDict{Symbol, Int}
    E::OrderedDict{Symbol, Int}
    SH::OrderedDict{Symbol, Int}
end

# Parameters are based off Schorfheide et al. (2016) and converted from a monthly frequency to quarterly
function BansalYaron2004(; μ_y::T = 0.0016 * 3., ρ_x::T = 0.99^3, σ_x::T = sqrt((0.74 * sqrt(1. - ρ_x^2))^2 * 3.), ρ_σ::T = 0.99^3,
                         σ_y::T = sqrt(0.0021^2 * 3.), ς::T = sqrt(0.0014^2 * 3.),
                         β::T = 0.999^3, ψ::T = 2., γ::T = 9.,
                         N_approx::LArray{Int64, 1, Array{Int64, 1}, (:q, :ω)} = (@LArray [1, 1] (:q, :ω))) where {T <: Real}

    @assert all(N_approx[k] > 0 for k in keys(N_approx)) "N_approx must be at least 1 for all variables."

    # Create indexing dictionaries
    S_init  = [:yy, :x, :σ²_y]                                        # State variables
    J_init  = [:q, :v, :ce, :ω]                                       # Jump variables
    SH_init = [:ε_y, :ε_x, :ε_σ²]                                     # Exogenous shocks
    E_init  = [:value_fnct, :certainty_equiv, :ez_fwd_diff, :cap_ret] # Equations
    for var in [:q, :ω]
        inds = (var == :q) ? (1:N_approx[var]) : (0:(N_approx[var] - 1))
        push!(J_init, [Symbol(:d, var, "$(i)") for i in inds]...)
        push!(J_init, [Symbol(:p, var, "$(i)") for i in 1:N_approx[var]]...)
        push!(E_init, [Symbol(:eq_d, var, "$(i)") for i in inds]...)
        push!(E_init, [Symbol(:eq_p, var, "$(i)") for i in 1:N_approx[var]]...)
    end

    S  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(S_init))
    J  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(J_init))
    E  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(E_init))
    SH = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(SH_init))

    para = @LArray [μ_y, ρ_x, σ_x, ρ_σ, σ_y, ς, β, ψ, γ] (:μ_y, :ρ_x, :σ_x, :ρ_σ, :σ_y, :ς, :β, :ψ, :γ)
    return BansalYaron2004{T}(para, N_approx, S, J, E, SH)
end

function bansal_yaron_2004(m::BansalYaron2004{T}; sparse_arrays::Bool = false,
                           sparse_jacobian::Vector{Symbol} = Symbol[]) where {T <: Real}

    # Unpack parameters and indexing dictionaries
    @unpack p, N_approx, S, J, E, SH = m

    @unpack yy, x, σ²_y = S
    @unpack q, v, ce, ω = J
    @unpack ε_y, ε_x, ε_σ² = SH
    @unpack value_fnct, certainty_equiv, ez_fwd_diff, cap_ret = E

    Nz = length(S)
    Ny = length(J)
    Nε = length(SH)

    ## Define nonlinear equations

    # Some helper functions
    m_ξ(z, y) = log(p.β) - (p.ψ - p.γ) * y[ce] - p.γ * p.μ_y
    function m_fwd!(i, Γ₅, Γ₆)
        Γ₅[i, yy] = -p.γ
        Γ₆[i, v]  = (p.ψ  - p.γ)
    end

    function μ(F, z, y) # note that y here refers to jump variables
        F[yy]   = z[x]
        F[x]    = p.ρ_x * z[x]
        F[σ²_y] = (1. - p.ρ_σ) * p.σ_y^2 + p.ρ_σ * z[σ²_y]
    end

    function ξ(F, z, y)

        m_ξv = m_ξ(z, y) # evaluate SDF
        F[value_fnct] = 1. / (1. - p.ψ) * (log(1. - p.β) + y[ω]) - y[v]
        F[certainty_equiv] = 1. / (1. - p.ψ) * (log(1. - p.β) - log(p.β) + log(exp(y[ω]) - 1.)) - y[ce]

        ## Forward-difference equations separately handled b/c recursions
        F[cap_ret]     = y[q] - log(sum([exp(y[J[Symbol("dq$(i)")]]) for i in 1:N_approx[:q]]) +
                                     exp(y[J[Symbol("pq$(N_approx[:q])")]]))
        F[ez_fwd_diff] = y[ω] - log(sum([exp(y[J[Symbol("dω$(i)")]]) for i in 0:(N_approx[:ω] - 1)]) +
                                     exp(y[J[Symbol("pω$(N_approx[:ω])")]]))

        # Set initial boundary conditions
        F[E[:eq_dq1]] = p.μ_y - y[J[:dq1]] + m_ξv
        F[E[:eq_pq1]] = p.μ_y - y[J[:pq1]] + m_ξv
        F[E[:eq_dω0]] = y[J[:dω0]]
        F[E[:eq_pω1]] = p.μ_y - y[J[:pω1]] + m_ξv

        # Recursions for forward-difference equations
        for i in 2:N_approx[:q]
            F[E[Symbol("eq_dq$(i)")]] = p.μ_y - y[J[Symbol("dq$(i)")]] + m_ξv
            F[E[Symbol("eq_pq$(i)")]] = p.μ_y - y[J[Symbol("pq$(i)")]] + m_ξv
        end
        for i in 2:N_approx[:ω]
            F[E[Symbol("eq_dω$(i-1)")]] = p.μ_y - y[J[Symbol("dω$(i-1)")]] + m_ξv
            F[E[Symbol("eq_pω$(i)")]]   = p.μ_y - y[J[Symbol("pω$(i)")]]   + m_ξv
        end
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    Λ = sparse_arrays ? spzeros(T, Nz, Ny) : zeros(T, Nz, Ny)

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Σ(F, z)
        F[yy,   ε_y]  = sqrt(z[σ²_y])
        F[x,    ε_x]  = sqrt(z[σ²_y]) * p.σ_x
        F[σ²_y, ε_σ²] = sqrt(z[σ²_y]) * p.ς
    end

    function ccgf(F, α, z)
        # F .= .5 * RiskAdjustedLinearizations.diag(α * α') # slower but this is the underlying math
        sum!(F, α.^2) # faster implementation
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
    m_fwd!(E[:eq_dq1], Γ₅, Γ₆)
    Γ₅[E[:eq_dq1], yy] = one(T)

    m_fwd!(E[:eq_pq1], Γ₅, Γ₆)
    Γ₅[E[:eq_pq1], yy] = one(T)
    Γ₆[E[:eq_pq1], q]  = one(T)

    m_fwd!(E[:eq_pω1], Γ₅, Γ₆)
    Γ₅[E[:eq_pω1], yy] = one(T)
    Γ₆[E[:eq_pω1], ω]  = one(T)

    # Forward difference equations: recursions
    for i in 2:N_approx[:q]
        m_fwd!(E[Symbol("eq_dq$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_dq$(i)")], yy] = one(T)
        Γ₆[E[Symbol("eq_dq$(i)")], J[Symbol("dq$(i-1)")]] = one(T)

        m_fwd!(E[Symbol("eq_pq$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_pq$(i)")], yy] = one(T)
        Γ₆[E[Symbol("eq_pq$(i)")], J[Symbol("pq$(i-1)")]] = one(T)
    end
    for i in 2:N_approx[:ω]
        m_fwd!(E[Symbol("eq_dω$(i-1)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_dω$(i-1)")], yy] = one(T)
        Γ₆[E[Symbol("eq_dω$(i-1)")], J[Symbol("dω$(i-2)")]] = one(T)

        m_fwd!(E[Symbol("eq_pω$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_pω$(i)")], yy] = one(T)
        Γ₆[E[Symbol("eq_pω$(i)")], J[Symbol("pω$(i-1)")]] = one(T)
    end

    z, y = create_deterministic_guess(m)
    Ψ = zeros(T, Ny, Nz)

    if sparse_arrays
        return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, vec(z), vec(y), Ψ, Nε; sparse_jacobian = sparse_jacobian,
                                         Λ_cache_init = dims -> spzeros(dims...), Σ_cache_init = dims -> spzeros(dims...))
    else
        return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, vec(z), vec(y), Ψ, Nε; sparse_jacobian = sparse_jacobian)
    end
end

function create_deterministic_guess(m::BansalYaron2004{T}) where {T <: Real}

    ## Set up

    # Unpack parameters and indexing dictionaries
    @unpack p, N_approx, S, J, E, SH = m

    @unpack yy, x, σ²_y = S
    @unpack q, v, ce, ω = J
    @unpack ε_y, ε_x, ε_σ² = SH
    @unpack value_fnct, certainty_equiv, ez_fwd_diff, cap_ret = E
    Nz = length(S)
    Ny = length(J)

    # Initialize deterministic steady state guess vectors
    z = Vector{T}(undef, Nz)
    y = Vector{T}(undef, Ny)

    ## Compute guesses

    # Steady state values of state variables known ex-ante
    z[yy]   = 0.
    z[x]    = 0.
    z[σ²_y] = p.σ_y^2

    # Now make guesses for remaining quantities
    Y0  = 1. # long-run value is long-run value of X, which is 1
    Ω0  = 1. / (1. - (p.β * Y0 * exp(p.μ_y)) ^ (1. - p.ψ))
    V0  = ((1. - p.β) * Ω0) ^ (1. / (1. - p.ψ))
    CE0 = ((1. - p.β) / p.β * (Ω0 - 1.)) ^ (1. / (1. - p.ψ))
    M0  = p.β * (p.β * Ω0 / (Ω0 - 1.)) ^ ((p.ψ - p.γ) / (1. - p.ψ)) * (Y0 * exp(p.μ_y)) ^ (-p.γ)
    Q0  = exp(p.μ_y) * M0 * Y0 / (1. - exp(p.μ_y) * M0 * Y0)

    y[q]  = log(Q0)
    y[v]  = log(V0)
    y[ce] = log(CE0)
    y[ω]  = log(Ω0)

    y[J[:dq1]] = convert(T, log(exp(p.μ_y) * M0 * Y0))
    y[J[:pq1]] = convert(T, log(exp(p.μ_y) * M0 * Y0 * Q0))
    y[J[:dω0]] = zero(T)
    y[J[:pω1]] = convert(T, log(exp(p.μ_y) * M0 * Y0 * Ω0))

    for i in 2:N_approx[:q]
        y[J[Symbol("dq$(i)")]] = convert(T, log(M0) + μ_y + log(Y0) + y[J[Symbol("dq$(i-1)")]])
        y[J[Symbol("pq$(i)")]] = convert(T, log(M0) + μ_y + log(Y0) + y[J[Symbol("pq$(i-1)")]])
    end
    for i in 2:N_approx[:ω]
        y[J[Symbol("dω$(i-1)")]] = convert(T, μ_y + log(M0) + log(Y0) + y[J[Symbol("dω$(i-2)")]])
        y[J[Symbol("pω$(i)")]] = convert(T, μ_y + log(M0) + log(Y0) + y[J[Symbol("pω$(i-1)")]])
    end

    return z, y
end
