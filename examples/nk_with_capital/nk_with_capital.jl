using UnPack, OrderedCollections, ForwardDiff, JLD2, NLsolve

mutable struct NKCapital{T <: Real}
    β::T
    γ::T
    φ::T
    ν::T
    χ::T
    δ::T
    α::T
    ϵ::T
    θ::T
    π_ss::T
    ϕ_r::T
    ϕ_π::T
    ϕ_y::T
    ρ_β::T
    ρ_l::T
    ρ_a::T
    ρ_r::T
    σ_β::T
    σ_l::T
    σ_a::T
    σ_r::T
    N_approx::Int
    S::OrderedDict{Symbol, Int}
    J::OrderedDict{Symbol, Int}
    E::OrderedDict{Symbol, Int}
    SH::OrderedDict{Symbol, Int}
end

function NKCapital(; β::T = .99, γ::T = 3.8, φ::T = 1., ν::T = 1., χ::T = 4.,
                   δ::T = 0.025, α::T = 0.33, ϵ::T = 10., θ::T = 0.7,
                   π_ss::T = 0., ϕ_r::T = 0.5,
                   ϕ_π::T = 1.3, ϕ_y::T = 0.25, ρ_β::T = 0.1,
                   ρ_l::T = 0.1, ρ_a::T = 0.9, ρ_r::T = 0.,
                   σ_β::T = 0.01, σ_l::T = 0.01, σ_a::T = 0.01, σ_r::T = 0.01,
                   N_approx::Int = 1) where {T <: Real}

    @assert N_approx > 0 "N_approx must be at least 1."

    ## Create Indexing dictionaries.

    # Note that for the exogenous shock
    # state variables, instead of e.g. η_L and η_A, I use η_l and η_a
    # since the uppercase variable will not appear in the jumps/states.
    S_init  = [:k₋₁, :v₋₁, :r₋₁, :output₋₁, :η_β, :η_l, :η_a, :η_r] # State Variables
    J_init  = [:output, :c, :l, :w, :r, :π, :q, :x, :rk, :ω, :mc,
               :s₁, :s₂, :v] # Jump variables
    E_init  = [:wage, :euler, :tobin, :cap_ret,
               :eq_mc, :kl_ratio, :eq_s₁, :eq_s₂,
               :phillips_curve, :price_dispersion,
               :mp, :output_market_clear, :production] # Equations
    SH_init = [:ε_β, :ε_l, :ε_a, :ε_r] # Exogenous shocks

    # Add approximations for forward-difference equations
    push!(E_init, :eq_omega)
    for var in [:q, :s₁, :s₂]
        inds = (var == :q) ? (1:N_approx) : (0:(N_approx - 1))
        push!(J_init, [Symbol(:d, var, "$(i)") for i in inds]...)
        push!(J_init, [Symbol(:p, var, "$(i)") for i in 1:N_approx]...)
        push!(E_init, [Symbol(:eq_d, var, "$(i)") for i in inds]...)
        push!(E_init, [Symbol(:eq_p, var, "$(i)") for i in 1:N_approx]...)
    end

    S  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(S_init))
    J  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(J_init))
    E  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(E_init))
    SH = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(SH_init))

    return NKCapital{T}(β, γ, φ, ν, χ, δ, α, ϵ, θ, π_ss, ϕ_r, ϕ_π, ϕ_y,
                        ρ_β, ρ_l, ρ_a, ρ_r, σ_β, σ_l, σ_a, σ_r,
                        N_approx, S, J, E, SH)
end

function nk_capital(m::NKCapital{T}) where {T <: Real}

    # Get parameters
    @unpack β, γ, φ, ν, χ, δ, α, ϵ, θ, π_ss, ϕ_r, ϕ_π, ϕ_y = m
    @unpack ρ_β, ρ_l, ρ_a, ρ_r, σ_β, σ_l, σ_a, σ_r = m
    r_ss = π_ss - log(β)
    X̄    = δ * χ / (χ + 1.)

    # Unpack indexing dictionaries
    @unpack N_approx, S, J, E, SH = m
    @unpack k₋₁, v₋₁, r₋₁, output₋₁, η_β, η_l, η_a, η_r = S
    @unpack output, c, l, w, r, π, q, x, rk, ω, mc, s₁, s₂, v = J
    @unpack wage, euler, tobin, cap_ret, eq_mc, kl_ratio, eq_s₁, eq_s₂ = E
    @unpack phillips_curve, price_dispersion, mp = E
    @unpack output_market_clear, production, eq_omega = E
    @unpack ε_β, ε_l, ε_a, ε_r = SH

    Nz = length(S)
    Ny = length(J)
    Nε = length(SH)

    ## Define nonlinear equations

    # Some helper functions
    _Φ(Xin, Kin)  = X̄ ^ (1. / χ) / (1. - 1. / χ) * (Xin / Kin) ^ (1. - 1. / χ) - X̄ / (χ * (χ - 1.))
    _Φ′(Xin, Kin) = X̄ ^ (1. / χ) * (Xin / Kin) ^ (- 1. / χ)
    Φ(z, y)  = _Φ(exp(y[x]), exp(z[k₋₁]))
    Φ′(z, y) = _Φ′(exp(y[x]), exp(z[k₋₁]))
    m_ξ(z, y) = log(β) - z[η_β] + γ * y[c]
    function m_fwd!(i, Γ₅, Γ₆)
        Γ₅[i, η_β] = 1.
        Γ₆[i, c]   = -γ
    end
    pstar(y) = log(ϵ / (ϵ - 1.)) + y[s₁] - y[s₂]

    function μ(F, z, y)
        F[k₋₁]      = log(1 + X̄ ^ (1. / χ) / (1. - 1. / χ) *
            (exp(y[x] - z[k₋₁])) ^ (1. - 1. / χ) -
            X̄ / (1. - 1. / χ)) + z[k₋₁]
        F[v₋₁]      = y[v]
        F[r₋₁]      = y[r]
        F[output₋₁] = y[output]
        F[η_β]      = ρ_β * z[η_β]
        F[η_l]      = ρ_l * z[η_l]
        F[η_a]      = ρ_a * z[η_a]
        F[η_r]      = ρ_r * z[η_r]
    end

    function ξ(F, z, y)
        F_type = eltype(F)

        ## Pre-evaluate (just once) some terms
        Φv     = Φ(z, y)
        Φ′v    = Φ′(z, y)
        pstarv = pstar(y)
        m_ξv   = m_ξ(z, y)

        ## Non-forward-difference equations
        F[wage]                = log(φ) + z[η_l] + ν * y[l] - (-γ * y[c] + y[w])
        F[euler]               = y[r] + m_ξv
        F[tobin]               = y[q] + log(Φ′v)
        F[eq_mc]               = (1. - α) * y[w] + α * y[rk] - z[η_a] -
            (1. - α) * log(1. - α) - α * log(α) - y[mc]
        F[kl_ratio]            = z[k₋₁] - y[l] - log(α / (1. - α)) - (y[w] - y[rk])
        F[phillips_curve]      = (1. - ϵ) * y[π] - log((1. - θ) * exp((1. - ϵ) * (pstarv + y[π])) + θ)
        F[price_dispersion]    = y[v] - ϵ * y[π] - log((1. - θ) * exp(-ϵ * (pstarv + y[π])) + θ * exp(z[v₋₁]))
        F[mp]                  = ϕ_r * z[r₋₁] + (1. - ϕ_r) .* (y[r] + ϕ_π * (y[π] - π_ss) +
                                                               ϕ_y * (y[output] - z[output₋₁])) + z[η_r] - y[r]
        F[output_market_clear] = y[output] - log(exp(y[c]) + exp(y[x]))
        F[production]          = z[η_a] + α * z[k₋₁] + (1. - α) * y[l] - y[v] - y[output]

        ## Forward-difference equations separately handled b/c recursions
        F[eq_omega] = 1. - δ + Φv - Φ′v * exp(y[x] - z[k₋₁]) - exp(y[ω])
        F[cap_ret]  = y[q] - log(sum([exp(y[J[Symbol("dq$(i)")]]) for i in 1:N_approx]) +
                                exp(y[J[Symbol("pq$(N_approx)")]]))
        F[eq_s₁]    = y[s₁] - log(sum([exp(y[J[Symbol("ds₁$(i)")]]) for i in 0:(N_approx - 1)]) +
                               exp(y[J[Symbol("ps₁$(N_approx)")]]))
        F[eq_s₂]    = y[s₂] - log(sum([exp(y[J[Symbol("ds₂$(i)")]]) for i in 0:(N_approx - 1)]) +
                               exp(y[J[Symbol("ps₂$(N_approx)")]]))

        # Set initial boundary conditions
        F[E[:eq_dq1]]  = -y[J[:dq1]] + m_ξv
        F[E[:eq_pq1]]  = -y[J[:pq1]] + m_ξv
        F[E[:eq_ds₁0]] = y[J[:ds₁0]] - y[mc] - y[output]
        F[E[:eq_ps₁1]] = log(θ) - y[J[:ps₁1]] + m_ξv
        F[E[:eq_ds₂0]] = y[J[:ds₂0]] - y[output]
        F[E[:eq_ps₂1]] = log(θ) - y[J[:ps₂1]] + m_ξv

        # Recursions for forward-difference equations
        for i in 2:N_approx
            F[E[Symbol("eq_dq$(i)")]]    = -y[J[Symbol("dq$(i)")]] + m_ξv
            F[E[Symbol("eq_pq$(i)")]]    = -y[J[Symbol("pq$(i)")]] + m_ξv
            F[E[Symbol("eq_ds₁$(i-1)")]] = log(θ) - y[J[Symbol("ds₁$(i-1)")]] + m_ξv
            F[E[Symbol("eq_ps₁$(i)")]]   = log(θ) - y[J[Symbol("ps₁$(i)")]]   + m_ξv
            F[E[Symbol("eq_ds₂$(i-1)")]] = log(θ) - y[J[Symbol("ds₂$(i-1)")]] + m_ξv
            F[E[Symbol("eq_ps₂$(i)")]]   = log(θ) - y[J[Symbol("ps₂$(i)")]]   + m_ξv
        end
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    Λ = zeros(T, Nz, Ny)

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Σ(F, z)
        F[η_β, ε_β] = σ_β
        F[η_l, ε_l] = σ_l
        F[η_a, ε_a] = σ_a
        F[η_r, ε_r] = σ_r
    end

    function ccgf(F, α, z)
        # F .= .5 * RiskAdjustedLinearizations.diag(α * α') # slower but this is the underlying math
        F .= vec(.5 * sum(α.^2, dims = 2)) # faster implementation
    end

    ## Forward-looking variables
    Γ₅ = zeros(T, Ny, Nz)
    Γ₆ = zeros(T, Ny, Ny)

    # Equations w/out SDF terms and are not forward-difference equations
    Γ₆[euler, π] = -one(T)

    # Equations with SDF terms but are not forward-difference equations
    m_fwd!(euler, Γ₅, Γ₆)

    # Forward difference equations: boundary conditions
    m_fwd!(E[:eq_dq1], Γ₅, Γ₆)
    Γ₆[E[:eq_dq1], rk] = one(T)

    m_fwd!(E[:eq_pq1], Γ₅, Γ₆)
    Γ₆[E[:eq_pq1], q] = one(T)
    Γ₆[E[:eq_pq1], ω] = one(T)

    m_fwd!(E[:eq_ps₁1], Γ₅, Γ₆)
    Γ₆[E[:eq_ps₁1], s₁] = one(T)

    m_fwd!(E[:eq_ps₂1], Γ₅, Γ₆)
    Γ₆[E[:eq_ps₂1], s₂] = one(T)

    # Forward difference equations: recursions
    for i in 2:N_approx
        m_fwd!(E[Symbol("eq_dq$(i)")], Γ₅, Γ₆)
        Γ₆[E[Symbol("eq_dq$(i)")], ω] = one(T)
        Γ₆[E[Symbol("eq_dq$(i)")], J[Symbol("dq$(i-1)")]] = one(T)

        m_fwd!(E[Symbol("eq_pq$(i)")], Γ₅, Γ₆)
        Γ₆[E[Symbol("eq_pq$(i)")], ω] = one(T)
        Γ₆[E[Symbol("eq_pq$(i)")], J[Symbol("pq$(i-1)")]] = one(T)

        m_fwd!(E[Symbol("eq_ds₁$(i-1)")], Γ₅, Γ₆)
        Γ₆[E[Symbol("eq_ds₁$(i-1)")], π] = convert(T, ϵ)
        Γ₆[E[Symbol("eq_ds₁$(i-1)")], J[Symbol("ds₁$(i-2)")]] = one(T)

        m_fwd!(E[Symbol("eq_ps₁$(i)")], Γ₅, Γ₆)
        Γ₆[E[Symbol("eq_ps₁$(i)")], π] = convert(T, ϵ)
        Γ₆[E[Symbol("eq_ps₁$(i)")], J[Symbol("ps₁$(i-1)")]] = one(T)

        m_fwd!(E[Symbol("eq_ds₂$(i-1)")], Γ₅, Γ₆)
        Γ₆[E[Symbol("eq_ds₁$(i-1)")], π] = convert(T, ϵ) - one(T)
        Γ₆[E[Symbol("eq_ds₁$(i-1)")], J[Symbol("ds₁$(i-2)")]] = one(T)

        m_fwd!(E[Symbol("eq_ps₂$(i)")], Γ₅, Γ₆)
        Γ₆[E[Symbol("eq_ps₂$(i)")], π] = convert(T, ϵ) - one(T)
        Γ₆[E[Symbol("eq_ps₂$(i)")], J[Symbol("ps₂$(i-1)")]] = one(T)
    end

    ## Mapping from states to jump variables
    Ψ = zeros(T, Ny, Nz)

    ## Deterministic steady state as initial guess
    z = Vector{T}(undef, Nz)
    y = Vector{T}(undef, Ny)

    # AR(1) start at 0
    η_β0 = 0.
    η_l0 = 0.
    η_a0 = 0.
    η_r0 = 0.

    # Variables known outright
    M0 = β
    Q0 = 1.
    RK0 = 1. / β + X̄ - 1.

    # Guesses
    L0 = .5548
    V0 = 1. # true if π_ss = 0, otherwise this is only a reasonable guess

    # Implied values given guesses
    C0_fnct = Cin -> Cin[1] + X̄ * (α / (1. - α) * φ * L0 ^ ν / Cin[1] ^ (-γ) / RK0 * L0) -
        (α / (1. - α) * φ * L0 ^ ν / Cin[1] ^ (-γ) / RK0) ^ α * L0 / V0
    C0_guess = NaN
    for theguess in .5:.5:10.
        try
            C0_fnct([theguess])
            C0_guess = theguess
        catch e
        end
    end
    C0 = nlsolve(C0_fnct, [C0_guess]).zero[1]
    W0 = φ * L0 ^ ν / C0 ^ (-γ)
    MC0 = (1. / (1. - α)) ^ (1. - α) * (1. / α) ^ α * W0 ^ (1. - α) * RK0 ^ α
    K0  = α / (1. - α) * W0 / RK0 * L0
    X0  = X̄ * K0
    Y0  = K0 ^ α * L0 ^ (1. - α) / V0
    S₁0  = MC0 * Y0 / (1. - θ * exp(π_ss) ^ ϵ)
    S₂0  = Y0 / (1. - θ * exp(π_ss) ^ (ϵ - 1.))
    Π0  = exp(π_ss)
    R0  = exp(r_ss)
    Ω0  = 1. - δ + _Φ(X0, K0) - _Φ′(X0, K0) * X0 / K0
    z .= [convert(T, x) for x in log.([K0, V0, R0, Y0, exp.([η_β0, η_l0, η_a0, η_r0])...])]
    y[1:14] = [convert(T, x) for x in log.([Y0, C0, L0, W0, R0, Π0, Q0, X0, RK0, Ω0, MC0, S₁0, S₂0, V0])]

    y[J[:dq1]] = convert(T, log(M0 * RK0))
    y[J[:pq1]] = convert(T, log(Ω0 * M0 * Q0))
    y[J[:ds₁0]] = convert(T, log(MC0 * Y0))
    y[J[:ps₁1]] = convert(T, log(θ * M0 * Π0^ϵ * S₁0))
    y[J[:ds₂0]] = convert(T, log(Y0))
    y[J[:ps₂1]] = convert(T, log(θ * M0 * Π0^(ϵ - 1.) * S₂0))

    for i in 2:N_approx
        y[J[Symbol("dq$(i)")]] = convert(T, log(M0) + log(Ω0) + y[J[Symbol("dq$(i-1)")]])
        y[J[Symbol("pq$(i)")]] = convert(T, log(M0) + log(Ω0) + y[J[Symbol("pq$(i-1)")]])
        y[J[Symbol("ds₁$(i-1)")]] = convert(T, log(θ) + log(M0) + ϵ * π_ss + y[J[Symbol("ds₁$(i-2)")]])
        y[J[Symbol("ps₁$(i)")]] = convert(T, log(θ) + log(M0) + ϵ * π_ss + y[J[Symbol("ps₁$(i-1)")]])
        y[J[Symbol("ds₂$(i-1)")]] = convert(T, log(θ) + log(M0) + (ϵ - 1.) * π_ss + y[J[Symbol("ds₂$(i-2)")]])
        y[J[Symbol("ps₂$(i)")]] = convert(T, log(θ) + log(M0) + (ϵ - 1.) * π_ss + y[J[Symbol("ps₂$(i-1)")]])
    end

    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, vec(z), vec(y), Ψ, Nε)
end

nk_cₜ(m, zₜ) = exp(m.y[2] + (m.Ψ * (zₜ - m.z))[2])
nk_qₜ(m, zₜ) = exp(m.y[7] + (m.Ψ * (zₜ - m.z))[7])
nk_dqₜ(m, zₜ, i, J) = exp(m.y[J[Symbol("dq$(i)")]] + (m.Ψ * (zₜ - m.z))[J[Symbol("dq$(i)")]])
nk_pqₜ(m, zₜ, i, J) = exp(m.y[J[Symbol("pq$(i)")]] + (m.Ψ * (zₜ - m.z))[J[Symbol("pq$(i)")]])
nk_s₁ₜ(m, zₜ) = exp(m.y[12] + (m.Ψ * (zₜ - m.z))[12])
nk_s₂ₜ(m, zₜ) = exp(m.y[13] + (m.Ψ * (zₜ - m.z))[13])
nk_ds₁ₜ(m, zₜ, i, J) = exp(m.y[J[Symbol("ds₁$(i)")]] + (m.Ψ * (zₜ - m.z))[J[Symbol("ds₁$(i)")]])
nk_ps₁ₜ(m, zₜ, i, J) = exp(m.y[J[Symbol("ps₁$(i)")]] + (m.Ψ * (zₜ - m.z))[J[Symbol("ps₁$(i)")]])
nk_ds₂ₜ(m, zₜ, i, J) = exp(m.y[J[Symbol("ds₂$(i)")]] + (m.Ψ * (zₜ - m.z))[J[Symbol("ds₂$(i)")]])
nk_ps₂ₜ(m, zₜ, i, J) = exp(m.y[J[Symbol("ps₂$(i)")]] + (m.Ψ * (zₜ - m.z))[J[Symbol("ps₂$(i)")]])

# Evaluates Euler equation errors in log terms
function nk_log_euler(m, zₜ, εₜ₊₁, Cₜ; β::T = .99, γ::T = 3.8,
                      J::AbstractDict = NKCapital().J, S::AbstractDict = NKCapital().S) where {T <: Real}
    yₜ = m.y + m.Ψ * (zₜ - m.z)
    zₜ₊₁, yₜ₊₁ = simulate(m, εₜ₊₁, zₜ)
    return log(β) - γ * (yₜ₊₁[J[:c]] - log(Cₜ)) +
        zₜ₊₁[S[:η_β]] - zₜ[S[:η_β]] + yₜ[J[:r]] - yₜ₊₁[J[:π]]
end

function nk_log_dq(m, zₜ, εₜ₊₁, DQₜ; β::T = .99, γ::T = 3.8,
                   i::Int = 1, J::AbstractDict = NKCapital().J, S::AbstractDict = NKCapital().S) where {T <: Real}
    yₜ = m.y + m.Ψ * (zₜ - m.z)
    zₜ₊₁, yₜ₊₁ = simulate(m, εₜ₊₁, zₜ)
    mₜ₊₁ = log(β) - γ * (yₜ₊₁[J[:c]] - yₜ[J[:c]]) +
        zₜ₊₁[S[:η_β]] - zₜ[S[:η_β]]
    @show DQₜ
    if i == 1
        return mₜ₊₁ + yₜ₊₁[J[:rk]] - log(DQₜ)
    else
        return yₜ₊₁[J[:ω]] + mₜ₊₁ + yₜ₊₁[J[Symbol("dq$(i-1)")]] - log(DQₜ)
    end
end

function nk_log_pq(m, zₜ, εₜ₊₁, PQₜ; β::T = .99, γ::T = 3.8,
                   i::Int = 1, J::AbstractDict = NKCapital().J, S::AbstractDict = NKCapital().S) where {T <: Real}
    yₜ = m.y + m.Ψ * (zₜ - m.z)
    zₜ₊₁, yₜ₊₁ = simulate(m, εₜ₊₁, zₜ)
    mₜ₊₁ = log(β) - γ * (yₜ₊₁[J[:c]] - yₜ[J[:c]]) +
        zₜ₊₁[S[:η_β]] - zₜ[S[:η_β]]
    if i == 1
        return yₜ₊₁[J[:ω]] + mₜ₊₁ + yₜ₊₁[J[:q]] - log(PQₜ)
    else
        return yₜ₊₁[J[:ω]] + mₜ₊₁ + yₜ₊₁[J[Symbol("pq$(i-1)")]] - log(PQₜ)
    end
end

function nk_log_ds₁(m, zₜ, εₜ₊₁, DS₁ₜ; β::T = .99, γ::T = 3.8,
                    θ::T = 0.7, ϵ::T = 10., i::Int = 0,
                    J::AbstractDict = NKCapital().J, S::AbstractDict = NKCapital().S) where {T <: Real}
    yₜ = m.y + m.Ψ * (zₜ - m.z)
    zₜ₊₁, yₜ₊₁ = simulate(m, εₜ₊₁, zₜ)
    mₜ₊₁ = log(β) - γ * (yₜ₊₁[J[:c]] - yₜ[J[:c]]) +
        zₜ₊₁[S[:η_β]] - zₜ[S[:η_β]]
    if i == 0
        return yₜ[J[:mc]] + yₜ[J[:output]]
    else
        return log(θ) + mₜ₊₁ + ϵ * yₜ₊₁[J[:π]] + yₜ₊₁[J[Symbol("ds₁$(i-1)")]] - log(DS₁ₜ)
    end
end

function nk_log_ps₁(m, zₜ, εₜ₊₁, PS₁ₜ; β::T = .99, γ::T = 3.8,
                    θ::T = 0.7, ϵ::T = 10., i::Int = 0,
                    J::AbstractDict = NKCapital().J, S::AbstractDict = NKCapital().S) where {T <: Real}
    yₜ = m.y + m.Ψ * (zₜ - m.z)
    zₜ₊₁, yₜ₊₁ = simulate(m, εₜ₊₁, zₜ)
    mₜ₊₁ = log(β) - γ * (yₜ₊₁[J[:c]] - yₜ[J[:c]]) +
        zₜ₊₁[S[:η_β]] - zₜ[S[:η_β]]
    if i == 1
        return log(θ) + mₜ₊₁ + ϵ * yₜ₊₁[J[:π]] + yₜ₊₁[J[:s₂]] - log(PS₁ₜ)
    else
        return log(θ) + mₜ₊₁ + ϵ * yₜ₊₁[J[:π]] + yₜ₊₁[J[Symbol("ps₁$(i-1)")]] - log(PS₁ₜ)
    end
end

function nk_log_ds₂(m, zₜ, εₜ₊₁, DS₂ₜ; β::T = .99, γ::T = 3.8,
                    θ::T = 0.7, ϵ::T = 10., i::Int = 0,
                    J::AbstractDict = NKCapital().J, S::AbstractDict = NKCapital().S) where {T <: Real}
    yₜ = m.y + m.Ψ * (zₜ - m.z)
    zₜ₊₁, yₜ₊₁ = simulate(m, εₜ₊₁, zₜ)
    mₜ₊₁ = log(β) - γ * (yₜ₊₁[J[:c]] - yₜ[J[:c]]) +
        zₜ₊₁[S[:η_β]] - zₜ[S[:η_β]]
    if i == 0
        return yₜ[J[:output]]
    else
        return log(θ) + mₜ₊₁ + (ϵ - 1.) * yₜ₊₁[J[:π]] + yₜ₊₁[J[Symbol("ds₂$(i-1)")]] - log(DS₂ₜ)
    end
end

function nk_log_ps₂(m, zₜ, εₜ₊₁, PS₂ₜ; β::T = .99, γ::T = 3.8,
                    θ::T = 0.7, ϵ::T = 10., i::Int = 0,
                    J::AbstractDict = NKCapital().J, S::AbstractDict = NKCapital().S) where {T <: Real}
    yₜ = m.y + m.Ψ * (zₜ - m.z)
    zₜ₊₁, yₜ₊₁ = simulate(m, εₜ₊₁, zₜ)
    mₜ₊₁ = log(β) - γ * (yₜ₊₁[J[:c]] - yₜ[J[:c]]) +
        zₜ₊₁[S[:η_β]] - zₜ[S[:η_β]]
    if i == 1
        return log(θ) + mₜ₊₁ + (ϵ - 1.) * yₜ₊₁[J[:π]] + yₜ₊₁[J[:s₂]] - log(PS₂ₜ)
    else
        return log(θ) + mₜ₊₁ + (ϵ - 1.) * yₜ₊₁[J[:π]] + yₜ₊₁[J[Symbol("ps₂$(i-1)")]] - log(PS₂ₜ)
    end
end

# Evaluates n-period ahead Euler equation errors in log terms

# Calculate Euler equation via quadrature
std_norm_mean = zeros(4)
std_norm_sig  = ones(4)
nk_𝔼_quadrature(f::Function) = gausshermite_expectation(f, std_norm_mean, std_norm_sig, 5)
