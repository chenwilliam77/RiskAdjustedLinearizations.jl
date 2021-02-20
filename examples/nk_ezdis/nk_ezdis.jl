using UnPack, OrderedCollections, ForwardDiff, JLD2, NLsolve

mutable struct NKEZDisaster{T <: Real, S, N}
    β::T
    γ::T
    ψ::T
    ν::T
    ν̅::T
    χ::T
    δ::T
    α::T
    ϵ::T
    θ::T
    π_ss::T
    ϕ_r::T
    ϕ_π::T
    ϕ_y::T
    χ_y::T
    ρ_β::T
    ρ_l::T
    ρ_r::T
    σ_β::T
    σ_l::T
    σ_r::T
    μ_a::T
    σ_a::T
    κ_a::T
    disaster_occur_spec::Symbol
    disaster_intensity_spec::Symbol
    disaster_para::NamedTuple{S, NTuple{N, T}}
    N_approx::NamedTuple{(:q, :s₁, :s₂, :ω), NTuple{4, Int}}
    S::OrderedDict{Symbol, Int}
    J::OrderedDict{Symbol, Int}
    E::OrderedDict{Symbol, Int}
    SH::OrderedDict{Symbol, Int}
end

# Absent a better way, I assume (1) each specification of disaster risk
# has a unique name and (2) disaster_para has correctly named parameters,
# given the specification's name. To see the process implied by
# `disaster_occur_spec` and `disaster_intensity_spec`, see
# the functions `infer_ccgf` and `infer_X̅` at the end of this file.
function NKEZDisaster(disaster_occur_spec::Symbol = :PoissonNormalMixture,
                      disaster_intensity_spec::Symbol = :CoxIngersollRoss,
                      disaster_para::NamedTuple{S1, NTuple{N1, T}} =
                      (σ_k = .01, ρ_p = .08^(1. / 4.), p = .0355 / 4., σ_p = .0114 / 4. / (.02 / sqrt(4.)) / sqrt(.0355 / 4.));
                      β::T = .99, γ::T = 3.8, ψ::T = 1. / .75, ν::T = 1., ν̅ = 0.72,
                      χ::T = 4., δ::T = 0.025, α::T = 0.33, ϵ::T = 10., θ::T = 0.7,
                      π_ss::T = 0., ϕ_r::T = 0.5, ϕ_π::T = 1.3, ϕ_y::T = 0.25,
                      χ_y::T = 1.6, ρ_β::T = 0.1, ρ_l::T = 0.1,, ρ_r::T = 0.,
                      σ_β::T = sqrt((log(β) / 4.)^2 * (1. - ρ_β^2)),
                      σ_l::T = 0.01, σ_r::T = 0.01, μ_a::T = 0.0125,
                      σ_a::T = 0.01, κ_a::T = 1.,
                      N_approx::NamedTuple{(:q, :s₁, :s₂, :ω), NTuple{4, Int}} =
                      (q = 1, s₁ = 1, s₂ = 1, ω = 1)) where {T <: Real, S1, N1}

    @assert all(N_approx[k] > 0 for k in keys(N_approx)) "N_approx must be at least 1 for all variables."

    ## Create Indexing dictionaries.

    # Note that for the exogenous shock
    # state variables, instead of e.g. η_L and η_A, I use η_l and η_a
    # since the uppercase variable will not appear in the jumps/states.
    S_init  = [:k₋₁, :logΔ₋₁, :r₋₁, :output₋₁, :η_β, :η_l, :η_r, :a, :η_k] # State Variables
    J_init  = [:output, :c, :l, :v, :ce, :ω, :ℓ, :β̅, :w, :r, :π, :q, :x,
               :rk, :rq, :mc, :s₁, :s₂, :logΔ] # Jump variables
    E_init  = [:value_fnct, :certainty_equiv, :ez_fwd_diff,
               :eq_β̅, :wage, :labor_disutility, :euler, :cap_ret,
               :eq_mc, :kl_ratio, :eq_s₁, :eq_s₂,
               :tobin,, :eq_rq, :phillips_curve, :price_dispersion,
               :mp, :output_market_clear, :production] # Equations
    SH_init = [:ε_β, :ε_l, :ε_r, :ε_a, :ε_k, :ε_p] # Exogenous shocks

    # Add approximations for forward-difference equations
    for var in [:q, :s₁, :s₂, :ω]
        inds = (var == :q) ? (1:N_approx[var]) : (0:(N_approx[var] - 1))
        push!(J_init, [Symbol(:d, var, "$(i)") for i in inds]...)
        push!(J_init, [Symbol(:p, var, "$(i)") for i in 1:N_approx[var]]...)
        push!(E_init, [Symbol(:eq_d, var, "$(i)") for i in inds]...)
        push!(E_init, [Symbol(:eq_p, var, "$(i)") for i in 1:N_approx[var]]...)
    end

    # Specify random process(es) for whether a disaster occurs or not
    if disaster_occur_spec in [:PoissonNormalMixture, :Bernoulli]
        # Nothing need to be added
    end

    # Specify random process(es) for "intensity" (size or frequency) of a disaster.
    if disaster_intensity_spec in [:CoxIngersollRoss, :TwoStateMarkovChain,
                                   :TruncatedCoxIngersollRoss]
        push!(S_init, :p)
    elseif disaster_intensity_spec in [:LogAR1]
        push!(S_init, :logp)
    end

    S  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(S_init))
    J  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(J_init))
    E  = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(E_init))
    SH = OrderedDict{Symbol, Int}(k => i for (i, k) in enumerate(SH_init))

    return NKEZDisaster(β, γ, ψ, ν,, ν̄,  χ, δ, α, ϵ, θ, π_ss, ϕ_r, ϕ_π, ϕ_y,
                        χ_y, ρ_β, ρ_l, ρ_r, σ_β, σ_l, σ_r, μ_a, σ_a, κ_a,
                        disaster_spec, disaster_para,
                        N_approx, S, J, E, SH)
end

function nk_ez_disaster(m::NKEZDisaster{T, SNK, NNK}) where {T <: Real, SNK, NNK}

    # Get parameters
    @unpack β, γ, ψ, ν, ν̅, χ, δ, α, ϵ, θ, π_ss, ϕ_r, ϕ_π, ϕ_y = m
    @unpack χ_y, ρ_β, ρ_l, ρ_r, σ_β, σ_l, σ_r, μ_a, σ_a, κ_a = m
    @unpack disaster_occur_spec, disaster_intensity_spec, disaster_para = m
    r_ss = infer_r_ss(m)
    X̅    = infer_X̅(m)
    𝔼η_k = infer_𝔼η_k(m)

    # Unpack indexing dictionaries
    @unpack N_approx, S, J, E, SH = m
    @unpack k₋₁, logΔ₋₁, r₋₁, output₋₁, η_β, η_l, η_r, a, η_k = S
    @unpack output, c, l, v, ce, ω, ℓ, β̅, w, r = J
    @unpack π, q, x, rk, rq, mc, s₁, s₂, logΔ  = J
    @unpack value_fnct, certainty_equiv, ez_fwd_diff = E
    @unpack eq_β̅, wage, labor_disutility, euler, cap_ret, eq_mc = E
    @unpack kl_ratio, eq_s₁, eq_s₂, tobin, eq_rq = E
    @unpack phillips_curve, price_dispersion, mp = E
    @unpack output_market_clear, production = E
    @unpack ε_β, ε_l, ε_r, ε_a, ε_k, ε_p = SH

    if disaster_intensity_spec in [:CoxIngersollRoss, :TwoStateMarkovChain,
                                   :TruncatedCoxIngersollRoss]
        p = m.S[:p]
        disaster_intensity_var = p
    elseif disaster_intensity_spec in [:LogAR1]
        logp = m.S[:logp]
        disaster_intensity_var = logp
    end

    Nz = length(S)
    Ny = length(J)
    Nε = length(SH)

    ## Define nonlinear equations

    # Some helper functions
    _Φ(Xin, Kin)  = X̅ ^ (1. / χ) / (1. - 1. / χ) * (Xin / Kin) ^ (1. - 1. / χ) - X̅ / (χ * (χ - 1.))
    _Φ′(Xin, Kin) = X̅ ^ (1. / χ) * (Xin / Kin) ^ (- 1. / χ)
    Φ(z, y)  = _Φ(exp(y[x]), exp(z[η_k] + z[k₋₁]))
    Φ′(z, y) = _Φ′(exp(y[x]), exp(z[η_k] + z[k₋₁]))
    m_ξ(z, y) = z[η_β] + log(β) - y[β̅] + γ * y[c] -
        (1. - γ) * y[ℓ] - (ψ - γ) * y[ce] - γ * μ_a
    μ_y_bgp(z, y) = μ_a + κ_a * 𝔼η_k # calculate growth rate of output along balanced growth path
    function m_fwd!(i, Γ₅, Γ₆)
        Γ₅[i, β̅] = 1.
        Γ₅[i, a] = -γ
        Γ₆[i, c] = -γ
        Γ₆[i, ℓ] = (1. - γ)
        Γ₆[i, v] = (ψ  - γ)
    end
    pstar(y) = log(ϵ / (ϵ - 1.)) + y[s₁] - y[s₂]
    μ_η_k    = infer_μ_disaster_occur(m)
    μ_disi   = infer_μ_disaster_intensity(m)

    function μ(F, z, y)
        # Expected value of η_k conditional on time t
        μ_η_k_v     = μ_η_k(z, y)

        F[k₋₁]      = log(1. - δ + Φ(z, y)) + z[η_k] + z[k₋₁]
        F[v₋₁]      = y[v]
        F[r₋₁]      = y[r]
        F[output₋₁] = y[output]
        F[η_β]      = ρ_β * z[η_β]
        F[η_l]      = ρ_l * z[η_l]
        F[η_r]      = ρ_r * z[η_r]
        F[a]        = κ_a * μ_η_k_v
        F[η_k]      = μ_η_k_v
        F[disaster_intensity_var] = μ_disi(z, y)
    end

    function ξ(F, z, y)
        F_type = eltype(F)

        ## Pre-evaluate (just once) some terms
        Φv     = Φ(z, y)
        Φ′v    = Φ′(z, y)
        pstarv = pstar(y)
        m_ξv   = m_ξ(z, y)

        ## Non-forward-difference equations
        F[value_fnct]          = 1. / (1. - ψ) * (y[β̅] + y[ω]) - y[v]
        F[certainty_equiv]     = 1. / (1. - ψ) * (y[β̅] - (z[η_β] + log(β)) + log(exp(y[ω]) - 1.)) - y[ce]
        F[wage]                = log(ψ) + z[η_l] + log(ν̅) + y[c] + ν * y[l] - (1. - ψ) / ψ * y[ℓ] - y[w]
        F[labor_disutility]    = ψ / (1. - ψ) * log(1. + (ψ - 1.) * exp(z[η_l]) * ν̅ *
                                                    exp((1. + ν) * y[l]) / (1. + ν)) - y[ℓ]
        F[euler]               = y[r] + m_ξv
        F[eq_mc]               = (1. - α) * (y[w] - log(1. - α)) + α * (y[rk] - log(α)) - y[mc]
        F[kl_ratio]            = log(α) - log(1. - α) + y[w] - y[rk] - (z[η_k] + z[k₋₁] - y[l])
        F[tobin]               = log(Φ′v) + y[q]
        F[eq_rq]               = log(1. - δ + Φv - Φ′v * exp(y[x] - (z[η_k] + z[k₋₁]))) - y[rq]
        F[phillips_curve]      = (1. - ϵ) * y[π] - log((1. - θ) * exp((1. - ϵ) * (pstarv + y[π])) + θ)
        F[price_dispersion]    = y[logΔ] - ϵ * y[π] - log((1. - θ) * exp(-ϵ * (pstarv + y[π])) + θ * exp(z[logΔ₋₁]))

        F[mp]                  = (1. - ϕ_r) * r_ss + ϕ_r * z[r₋₁] +
            (1. - ϕ_r) .* (ϕ_π * (y[π] - π_ss) + ϕ_y *
                           (y[output] - z[output₋₁] + (μ_a + z[a] - mp_μ_y_bgp(z, y)))) + z[η_r] - y[r]
        F[output_market_clear] = y[output] - log(exp(y[c]) + exp(y[x]))
        F[production]          = log(exp(α * z[k₋₁] + (1. - α) * y[l]) - χ_y) - y[logΔ] - y[output]
        F[eq_β̅]                = log(1. - exp(z[η_β])) - y[β̅]

        ## Forward-difference equations separately handled b/c recursions
        F[cap_ret]     = y[q]  - log(sum([exp(y[J[Symbol("dq$(i)")]]) for i in 1:N_approx[:q]]) +
                                     exp(y[J[Symbol("pq$(N_approx[:q])")]]))
        F[eq_s₁]       = y[s₁] - log(sum([exp(y[J[Symbol("ds₁$(i)")]]) for i in 0:(N_approx[:s₁] - 1)]) +
                                      exp(y[J[Symbol("ps₁$(N_approx[:s₁])")]]))
        F[eq_s₂]       = y[s₂] - log(sum([exp(y[J[Symbol("ds₂$(i)")]]) for i in 0:(N_approx[:s₂] - 1)]) +
                                      exp(y[J[Symbol("ps₂$(N_approx[:s₂])")]]))
        F[ez_fwd_diff] = y[ω]  - log(sum([exp(y[J[Symbol("dω$(i)")]]) for i in 0:(N_approx[:ω] - 1)]) +
                                     exp(y[J[Symbol("pω$(N_approx[:ω])")]]))

        # Set initial boundary conditions
        F[E[:eq_dq1]]  = -y[J[:dq1]] + m_ξv
        F[E[:eq_pq1]]  = -y[J[:pq1]] + m_ξv
        F[E[:eq_ds₁0]] = y[J[:ds₁0]] - y[mc] - y[output]
        F[E[:eq_ps₁1]] = μ_a + log(θ) - y[J[:ps₁1]] + m_ξv
        F[E[:eq_ds₂0]] = y[J[:ds₂0]] - y[output]
        F[E[:eq_ps₂1]] = μ_a + log(θ) - y[J[:ps₂1]] + m_ξv
        F[E[:eq_dω0]]  = y[J[:dω0]]
        F[E[:eq_pω1]]  = μ_a - y[c] - y[J[:pω1]] + m_ξv

        # Recursions for forward-difference equations
        for i in 2:N_approx[:q]
            F[E[Symbol("eq_dq$(i)")]]    = -y[J[Symbol("dq$(i)")]] + m_ξv
            F[E[Symbol("eq_pq$(i)")]]    = -y[J[Symbol("pq$(i)")]] + m_ξv
        end
        for i in 2:N_approx[:s₁]
            F[E[Symbol("eq_ds₁$(i-1)")]] = μ_a + log(θ) - y[J[Symbol("ds₁$(i-1)")]] + m_ξv
            F[E[Symbol("eq_ps₁$(i)")]]   = μ_a + log(θ) - y[J[Symbol("ps₁$(i)")]]   + m_ξv
        end
        for i in 2:N_approx[:s₂]
            F[E[Symbol("eq_ds₂$(i-1)")]] = μ_a + log(θ) - y[J[Symbol("ds₂$(i-1)")]] + m_ξv
            F[E[Symbol("eq_ps₂$(i)")]]   = μ_a + log(θ) - y[J[Symbol("ps₂$(i)")]]   + m_ξv
        end
        for i in 2:N_approx[:ω]
            F[E[Symbol("eq_dω$(i-1)")]] = μ_a - y[c] - y[J[Symbol("dω$(i-1)")]] + m_ξv
            F[E[Symbol("eq_pω$(i)")]]   = μ_a - y[c] - y[J[Symbol("pω$(i)")]]   + m_ξv
        end
    end

    # The cache is initialized as zeros so we only need to fill non-zero elements
    Λ = zeros(T, Nz, Ny)

    # The cache is initialized as zeros so we only need to fill non-zero elements
    function Σ(F, z)
        F_type = eltype(F)

        # AR(1) processes
        F[η_β, ε_β] = σ_β
        F[η_l, ε_l] = σ_l
        F[η_r, ε_r] = σ_r

        # Productivity process
        F[a, ε_a] = σ_a
        F[a, ε_k] = κ_a

        # Disaster risk
        F[η_k, ε_k] = one(F_type)
        F[disaster_intensity_var, ε_p] = Σ_disi(z)
    end

    ccgf = infer_ccgf(m)

    ## Forward-looking variables
    Γ₅ = zeros(T, Ny, Nz)
    Γ₆ = zeros(T, Ny, Ny)

    # Equations w/out SDF terms and are not forward-difference equations
    Γ₆[euler, π] = -one(T)

    # Equations with SDF terms but are not forward-difference equations
    m_fwd!(euler, Γ₅, Γ₆)

    # Forward difference equations: boundary conditions
    m_fwd!(E[:eq_dq1], Γ₅, Γ₆)
    Γ₅[E[:eq_dq1], η_k] = one(T)
    Γ₆[E[:eq_dq1], rk]  = one(T)

    m_fwd!(E[:eq_pq1], Γ₅, Γ₆)
    Γ₅[E[:eq_pq1], η_k] = one(T)
    Γ₆[E[:eq_pq1], q]   = one(T)
    Γ₆[E[:eq_pq1], rq]  = one(T)

    m_fwd!(E[:eq_ps₁1], Γ₅, Γ₆)
    Γ₅[E[:eq_ps₁1], a]  = one(T)
    Γ₆[E[:eq_ps₁1], π]  = convert(T, ϵ)
    Γ₆[E[:eq_ps₁1], s₁] = one(T)

    m_fwd!(E[:eq_ps₂1], Γ₅, Γ₆)
    Γ₅[E[:eq_ps₂1], a]  = one(T)
    Γ₆[E[:eq_ps₂1], π]  = convert(T, ϵ - 1.)
    Γ₆[E[:eq_ps₂1], s₂] = one(T)

    m_fwd!(E[:eq_pω1], Γ₅, Γ₆)
    Γ₆[E[:eq_pω1], c] = one(T)
    Γ₅[E[:eq_pω₁], a] = one(T)
    Γ₆[E[:eq_pω1], ω] = one(T)

    # Forward difference equations: recursions
    for i in 2:N_approx[:q]
        m_fwd!(E[Symbol("eq_dq$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_dq$(i)")], η_k] = one(T)
        Γ₆[E[Symbol("eq_dq$(i)")], rq] = one(T)
        Γ₆[E[Symbol("eq_dq$(i)")], J[Symbol("dq$(i-1)")]] = one(T)

        m_fwd!(E[Symbol("eq_pq$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_pq$(i)")], η_k] = one(T)
        Γ₆[E[Symbol("eq_pq$(i)")], rq] = one(T)
        Γ₆[E[Symbol("eq_pq$(i)")], J[Symbol("pq$(i-1)")]] = one(T)
    end

    for i in 2:N_approx[:s₁]
        m_fwd!(E[Symbol("eq_ds₁$(i-1)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_ds₁$(i-1)")], a] = one(T)
        Γ₆[E[Symbol("eq_ds₁$(i-1)")], π] = convert(T, ϵ)
        Γ₆[E[Symbol("eq_ds₁$(i-1)")], J[Symbol("ds₁$(i-2)")]] = one(T)

        m_fwd!(E[Symbol("eq_ps₁$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_ps₁$(i)")], a] = one(T)
        Γ₆[E[Symbol("eq_ps₁$(i)")], π] = convert(T, ϵ)
        Γ₆[E[Symbol("eq_ps₁$(i)")], J[Symbol("ps₁$(i-1)")]] = one(T)
    end

    for i in 2:N_approx[:s₂]
        m_fwd!(E[Symbol("eq_ds₂$(i-1)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_ds₂$(i-1)")], a] = one(T)
        Γ₆[E[Symbol("eq_ds₂$(i-1)")], π] = convert(T, ϵ) - one(T)
        Γ₆[E[Symbol("eq_ds₂$(i-1)")], J[Symbol("ds₂$(i-2)")]] = one(T)

        m_fwd!(E[Symbol("eq_ps₂$(i)")], Γ₅, Γ₆)
        Γ₅[E[Symbol("eq_ps₂$(i)")], a] = one(T)
        Γ₆[E[Symbol("eq_ps₂$(i)")], π] = convert(T, ϵ) - one(T)
        Γ₆[E[Symbol("eq_ps₂$(i)")], J[Symbol("ps₂$(i-1)")]] = one(T)
    end

    for i in 2:N_approx[:ω]
        m_fwd!(E[Symbol("eq_dω$(i-1)")], Γ₅, Γ₆)
        Γ₆[E[Symbol("eq_dω$(i-1)")], c] = one(T)
        Γ₅[E[Symbol("eq_dω$(i-1)")], a] = one(T)
        Γ₆[E[Symbol("eq_dω$(i-1)")], J[Symbol("dω$(i-2)")]] = one(T)

        m_fwd!(E[Symbol("eq_pω$(i)")], Γ₅, Γ₆)
        Γ₆[E[Symbol("eq_pω$(i)")], c] = one(T)
        Γ₅[E[Symbol("eq_pω$(i)")], a] = one(T)
        Γ₆[E[Symbol("eq_pω$(i)")], J[Symbol("pω$(i-1)")]] = one(T)
    end

    ## Mapping from states to jump variables
    Ψ = zeros(T, Ny, Nz)

    ## Deterministic steady state as initial guess
    z, y = create_deterministic_ss_guess(m)

    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, vec(z), vec(y), Ψ, Nε)
end

function create_deterministic_ss_guess(m::NKEZDisaster{T, SNK, NNK}) where {T <: Real, SNK, NNK}

    ## Set up

    # Get parameters
    @unpack β, γ, ψ, ν, ν̅, χ, δ, α, ϵ, θ, π_ss, ϕ_r, ϕ_π, ϕ_y = m
    @unpack χ_y, ρ_β, ρ_l, ρ_r, σ_β, σ_l, σ_r, μ_a, σ_a, κ_a = m
    @unpack disaster_occur_spec, disaster_intensity_spec, disaster_para = m
    r_ss = infer_r_ss(m)
    X̅    = infer_X̅(m)
    𝔼η_k = infer_𝔼η_k(m)

    # Unpack indexing dictionaries
    @unpack N_approx, S, J, E, SH = m
    @unpack k₋₁, logΔ₋₁, r₋₁, output₋₁, η_β, η_l, η_r, a, η_k = S
    @unpack output, c, l, v, ce, ω, ℓ, β̅, w, r = J
    @unpack π, q, x, rk, rq, mc, s₁, s₂, logΔ  = J

    ## Create guesses for deterministic steady state
    z = Vector{T}(undef, Nz)
    y = Vector{T}(undef, Ny)

    # AR(1) start at 0
    η_β0 = 0.
    η_l0 = 0.
    η_r0 = 0.

    # Disaster shock assumed to occur deterministically
    # and equals the unconditional expected value
    η_k0 = 𝔼η_k
    A0   = exp(κ_a * η_k0)

    # Variables known outright
    Ω0  = 1. / (1. - (β * A0 * exp(μ_a)) ^ (1. - ψ))
    V0  = ((1. - β) * Ω0) ^ (1. / (1. - ψ))
    𝒞ℰ0 = ((1. - β) / β * (Ω0 - 1.)) ^ (1. / (1. - ψ))
    M0  = β * (β * Ω0 / (Ω0 - 1.)) ^ ((ψ - γ) / (1. - ψ)) * (A0 * exp(μ_a)) ^ (-γ)
    R0  = exp(r_ss)
    Q0  = 1.
    Rq0 = 1 / η_k0 - X̅
    Rk0 = 1. / (M * exp(η_k0)) - Rq0
    expβ̅ = 1. - exp(η_β0) * β

    # Guesses
    L0 = .5548
    Δ0 = 1. # true if π_ss = 0, otherwise this is only a reasonable guess
    ℒ0 = (1. + (ψ - 1.) * exp(η_l0) * ν̅ * L0^(1. + ν) / (1. + ν))^(ψ / (1. - ψ))

    # Implied values given guesses
    C0_fnct = Cin -> Cin[1] + X̅ * (α / (1. - α) * ψ * ν̅ * C0 * L0^ν / ℒ0 / RK0 * L0) -
        ((α / (1. - α) * * ψ * ν̅ * C0 * L0^ν / ℒ0 / RK0) ^ α * L0 - χ_y) / Δ0
    C0_guess = NaN
    for theguess in .5:.5:10.
        try
            C0_fnct([theguess])
            C0_guess = theguess
        catch e
        end
    end
    C0 = nlsolve(C0_fnct, [C0_guess]).zero[1]
    W0 = ψ * exp(η_l0) * ν̅ * C0 * L0^ν / ℒ0^((1. - ψ) / ψ)
    MC0 = (1. / (1. - α)) ^ (1. - α) * (1. / α) ^ α * W0 ^ (1. - α) * RK0 ^ α
    K0  = (α / (1. - α) * W0 / RK0 * L0) / η_k0
    X0  = X̅ * η_k0 * K0
    Y0  = ((η_k0 * K0) ^ α * L0 ^ (1. - α) - χ_y) / Δ0
    Π0  = exp(π_ss)
    S₁0 = MC0 * Y0 / (1. - exp(μ_a) * θ * M0 * A0 * Π0 ^ ϵ)
    S₂0 = Y0 / (1. - exp(μ_a) * θ * M0 * A0 * Π0 ^ (ϵ - 1.))
    z .= [convert(T, x) for x in log.([K0, Δ0, R0, Y0, exp.([η_β0, η_l0, η_r0, log(A0), η_k0])...])]
    y[1:19] = [convert(T, x) for x in log.([Y0, C0, L0, V0, 𝒞ℰ0, Ω0, ℒ0, expβ̅, W0, R0, Π0, Q0, X0, Rk0, Rq0,
                                            MC0, S₁0, S₂0, Δ0])]

    y[J[:dq1]] = convert(T, log(M0 * Rk0))
    y[J[:pq1]] = convert(T, log(Rq0 * M0 * Q0))
    y[J[:ds₁0]] = convert(T, log(MC0 * Y0))
    y[J[:ps₁1]] = convert(T, log(exp(μ_a) * θ * M0 * A0 * Π0^ϵ * S₁0))
    y[J[:ds₂0]] = convert(T, log(Y0))
    y[J[:ps₂1]] = convert(T, log(exp(μ_a) * θ * M0 * A0 * Π0^(ϵ - 1.) * S₂0))

    # NEED TO ADD GUESSES FOR omega

    for i in 2:N_approx
        y[J[Symbol("dq$(i)")]] = convert(T, log(M0) + η_k0 + log(Rq0) + y[J[Symbol("dq$(i-1)")]])
        y[J[Symbol("pq$(i)")]] = convert(T, log(M0) + η_k0 + log(Rq0) + y[J[Symbol("pq$(i-1)")]])
        y[J[Symbol("ds₁$(i-1)")]] = convert(T, μ_a + log(θ) + log(M0) + log(A0) + ϵ * π_ss + y[J[Symbol("ds₁$(i-2)")]])
        y[J[Symbol("ps₁$(i)")]] = convert(T, μ_a + log(θ) + log(M0) + log(A0) + ϵ * π_ss + y[J[Symbol("ps₁$(i-1)")]])
        y[J[Symbol("ds₂$(i-1)")]] = convert(T, μ_a + log(θ) + log(M0) + log(A0) + (ϵ - 1.) * π_ss + y[J[Symbol("ds₂$(i-2)")]])
        y[J[Symbol("ps₂$(i)")]] = convert(T, μ_a + log(θ) + log(M0) + log(A0) + (ϵ - 1.) * π_ss + y[J[Symbol("ps₂$(i-1)")]])
    end
end

# Infer the value of η_k in the stochastic steady state
function infer_𝔼η_k(m::NKEZDisaster)
    d = m.disaster_para
    𝔼η_k = if m.disaster_occur_spec == :PoissonNormalMixture
        # η_{k, t} ∼ N(-jₜ, jₜ σ_k^2), jₜ ∼ Poisson(pₜ₋₁)
        # ⇒ 𝔼[ η_{k, t} ] = 𝔼[ 𝔼[η_{k, t} ∣ jₜ] ] = 𝔼[ -jₜ ]
        # = -𝔼[ 𝔼[ jₜ ∣ pₜ₋₁] ] = -𝔼[ pₜ₋₁ ].
        if m.disaster_intensity_spec == :CoxIngersollRoss
            # pₜ₋₁ ∼ discretized CIR process w/unconditional mean p
            # -𝔼[ pₜ₋₁ ] = -p
            -d[:p]
        elseif m.disaster_intensity_spec == :TwoStateMarkovChain
            # pₜ₋₁ ∼ Markov Chain with states p_ and p̅;
            # (respective) persistence probabilities ρ_ and ρ̅
            # -𝔼[ pₜ₋₁ ] = -(ergodic mean)
            -((1. - d[:ρ̅ₚ]) * d[:p_] + (1. - d[:ρ_ₚ] * d[:p̅])) / (2. - (d[:ρ_ₚ] + d[:ρ̅ₚ]))
        end
    elseif m.disaster_occur_spec == :Bernoulli
        # η_{k, t} ∼ Bernoulli(pₜ₋₁) taking values η_ w/probability pₜ₋₁ and zero otherwise.
        # ⇒ 𝔼[ η_{k, t} ] = 𝔼[ 𝔼[ η_{k, t} ∣ pₜ₋₁ ] ] = η_ 𝔼[ pₜ₋₁]
        if m.disaster_intensity_spec == :CoxIngersollRoss
            # pₜ₋₁ ∼ discretized CIR process w/unconditional mean p
            # η_ 𝔼[ pₜ₋₁ ] = η_ p
            d[:η_] * d[:p]
        elseif m.disaster_intensity_spec == :TwoStateMarkovChain
            # pₜ₋₁ ∼ Markov Chain with states p_ and p̅;
            # (respective) persistence probabilities ρ_ and ρ̅
            # η_ 𝔼[ pₜ₋₁ ] = η_ (ergodic mean)
            d[:η_] * ((1. - d[:ρ̅ₚ]) * d[:p_] + (1. - d[:ρ_ₚ] * d[:p̅])) / (2. - (d[:ρ_ₚ] + d[:ρ̅ₚ]))
        end
    end

    return 𝔼η_k
end

# Infer steady state investment rate given the disaster shock specification
function infer_X̅(m::NKEZDisaster)
    return m.χ / (m.χ + 1.) * (1. / exp(infer_𝔼η_k(m)) + m.δ - 1.)
end

# Figure out the steady state interest rate
# given EZ preferences and the disaster shock specification
function infer_r_ss(m::NKEZDisaster)

    # a = σ_a ε_a + κ_a * η_k
    # ⇒ in stochastic steady state, a = κ_a 𝔼[ η_k ]
    𝔼η_k = infer_𝔼η_k(m)
    A    = exp(m.κ_a * 𝔼η_k)

    # Stochastic steady state is the expected state,
    # conditional on shocks always equaling zero.
    Ω̃    = 1. / (1. - (m.β * A * exp(m.μ_a))^(1. - m.ψ))
    M    = m.β * (m.β * Ω̃ / (Ω̃ - 1.))^((ψ - γ) / (1. - ψ))
    return m.π_ss - log(M)
end

# Figure out the ccgf given the disaster shock specification
function infer_ccgf(m::NKEZDisaster)
    function ccgf(F, α, z)
        # F .= .5 * RiskAdjustedLinearizations.diag(α * α') # slower but this is the underlying math
        F .= vec(.5 * sum(α.^2, dims = 2)) # faster implementation
    end
end

# Infer state transition equations for
# the disaster shock occurrence η_k
function infer_μ_disaster_occur(m::NKEZDisaster)
    d = m.disaster_para

    # Define expected disaster shock proportion to pₜ b/c conditionally linear in pₜ
    𝔼ₜη_k_div_pₜ = if m.disaster_occur_spec == :PoissonNormalMixture
        # η_{k, t} ∼ N(-jₜ, jₜ σ_k^2), jₜ ∼ Poisson(pₜ₋₁)
        # ⇒ 𝔼ₜ[ η_{k, t + 1} ] = 𝔼ₜ[ 𝔼[η_{k, t + 1} ∣ jₜ₊₁] ] = 𝔼ₜ[ -jₜ₊₁ ]
        # = -𝔼ₜ[ 𝔼[ jₜ₊₁ ∣ pₜ] ] = -𝔼ₜ[ pₜ ] = -pₜ.
        -1.
    elseif m.disaster_occur_spec == :Bernoulli
        # η_{k, t} ∼ Bernoulli(pₜ₋₁) taking values η_ w/probability pₜ₋₁ and zero otherwise.
        # ⇒ 𝔼ₜ[ η_{k, t + 1} ] = 𝔼ₜ[ 𝔼[ η_{k, t + 1} ∣ pₜ ] ] = η_ 𝔼ₜ[ pₜ] = η_ pₜ
        d[:η_]
    end

    𝔼ₜη_k = if m.disaster_intensity_spec in [:CoxIngersollRoss, :TwoStateMarkovChain, :TruncatedCoxIngersollRoss]
        state_i = m.S[:p]
        _𝔼ₜη_k_linear(z, y) = 𝔼ₜη_k_div_pₜ * z[state_i]
    elseif m.disaster_intensity_spec in [:LogAR1]
        state_i = m.S[:logp]
        _𝔼ₜη_k_loglinear(z, y) = 𝔼ₜη_k_div_pₜ * exp(z[state_i])
    end

    return 𝔼ₜη_k
end

# Infer state transition equations for
# the disaster shock intensity p
function infer_μ_disaster_intensity(m::NKEZDisaster)
    d = m.disaster_para
    mdisi = m.disaster_intensity_spec

    μ_p = if mdisi == :CoxIngersollRoss
        state_i = m.S[:p]
        @inline _μ_p_cir(z, y) = (1. - d[:ρ_p]) * d[:p] + d[:ρ_p] * z[state_i]
    elseif mdisi == :TwoStateMarkovChain
        state_i = m.S[:p]
        @inline function _μ_p_2mc(z, y)
            if z[state_i] == d[:p̅]
                d[:ρ̅_p] * d[:p̅] + (1. - d[:ρ̅_p]) * d[:p_]
            else
                d[:ρ_p] * d[:p_] + (1. - d[:ρ_p]) * d[:p̅]
            end
        end
    elseif mdisi == :TruncatedCoxIngersollRoss
        error("TruncatedCoxIngersollRoss not implemented yet")
    elseif mdisi == :LogAR1
        state_i = m.S[:logp]
        @inline _μ_p_logar1(z, y) = (1 - d[:ρ_p]) * log(d[:p]) + d[:ρ_p] * z[state_i]
    end

    return μ_p
end

function infer_Σ_disaster_intensity(m::NKEZDisaster)
    d = m.disaster_para
    mdisi = m.disaster_intensity_spec

    Σ_p = if mdisi in [:CoxIngersollRoss, :TruncatedCoxIngersollRoss]
        state_i = m.S[:p]
        @inline _Σ_p_cir(z) = sqrt(z[state_i]) * d[:σ_p]
    elseif mdisi == :TwoStateMarkovChain
        @inline _Σ_p_2mc(z) = one(eltype(z))
    elseif mdisi == :LogAR1
        state_i = m.S[:logp]
        @inline _Σ_p_logar1(z) = d[:σ_p]
    end

    return Σ_p
end

# Infer the desired CCGF function
function infer_ccgf(m::NKEZDisaster)
    d = m.disaster_para
    S = m.S
    SH = m.SH
    not_dis_keys = setdiff(collect(keys(SH)), [:ε_k, :ε_p])
    not_dis_inds = [SH[i] for i in not_dis_keys]
    ccgf = if m.disaster_occur_spec == :PoissonNormalMixture
        # apply Poisson mgf to C_2(A) = -A + σ_j^2 A^2 / 2
        # Poisson mgf w/intensity pₜ₋₁ is exp((exp(s) - 1) pₜ₋₁)
        # and then subtract s * E_t[\eta_{k, t + 1}]

        if m.disaster_intensity_spec == :CoxIngersollRoss
            function _ccgf_poissonnormalmixture_cir(F, A, z)
                F  .= sum(A[:, vcat(not_dis_inds, SH[:ε_p])].^2, dims = 2) .* .5 # Gaussian parts
                A_k = @view A[:, SH[:ε_k]]
                F .+= ((exp.(-A_k + A_k.^2 .* (d[:σ_k] ^ 2 / 2.)) .- 1.) + A_k) .* z[S[:p]] # ε_k
            end
        elseif m.disaster_intensity_spec == :TwoStateMarkovChain
            function _ccgf_poissonnormalmixture_2smc(F, A, z)
                F .= sum((@view A[:, not_dis_inds]).^2, dims = 2) .* .5 # Gaussian parts
                A_k = @view A[:, SH[:ε_k]]
                F .+= ((exp.(-A_k + A_k.^2 .* (d[:σ_k] ^ 2 / 2.)) .- 1.) + A_k) .* z[S[:p]] # ε_k

                # ε_p
                A_p = @view A[:, SH[:ε_p]]
                if z[S[:p]] == d[:p̅]
                    F .+= log.((1. - d[:ρ̅_p]) .* exp.(A_p * d[:p_]) + d[:ρ̅_p] .* exp.(A_p * d[:p̅])) -
                        A_p .* (d[:ρ̅_p] * d[:p̅] + (1. - d[:ρ̅_p]) * d[:p_])
                else
                    F .+= log.((1. - d[:ρ_p]) .* exp.(A_p .* d[:p̅]) + d[:ρ_p] .* exp.(A_p .* d[:p_])) -
                        A_p .* (d[:ρ_p] * d[:p_] + (1. - d[:ρ_p]) * d[:p̅])
                end
            end
        elseif m.disaster_intensity_spec == :LogAR1
            function _ccgf_poissonnormalmixture_logar1(F, A, z)
                F .= sum(A[:, vcat(not_dis_inds, SH[:ε_p])].^2, dims = 2) .* .5 # Gaussian parts
                A_k = @view A[:, SH[:ε_k]]
                F .+= ((exp.(-A_k + A_k.^2 .* (d[:σ_k] ^ 2 / 2.)) .- 1.) + A_k) .* exp(z[S[:logp]]) # ε_k
            end
        end
    elseif m.disaster_occur_spec == :Bernoulli
        if m.disaster_intensity_spec == :CoxIngersollRoss
            function _ccgf_bernoulli_cir(F, A, z)
                F .= sum(A[:, vcat(not_dis_inds, SH[:ε_p])].^2, dims = 2) .* .5 # Gaussian parts
                A_k = @view A[:, SH[:ε_k]]
                F .+= log((1. - z[S[:p]] + z[S[:p]]) .* exp.(A_k * d[:η_])) .-
                    A_k .* (d[:η_] * z[S[:p]]) # ε_k
            end
        elseif m.disaster_intensity_spec == :TwoStateMarkovChain
            function _ccgf_bernoulli_2smc(F, A, z)
                F .= sum((@view A[:, not_dis_inds]).^2, dims = 2) .* .5 # Gaussian parts
                A_k = @view A[:, SH[:ε_k]]
                F .+= log.((1. - z[S[:p]] + z[S[:p]]) .* exp.(A_k * d[:η_])) .-
                    A_k .* (d[:η_] * z[S[:p]]) # ε_k

                # ε_p
                A_p = @view A[:, SH[:ε_p]]
                if z[S[:p]] == d[:p̅]
                    F .+= log.((1. - d[:ρ̅_p]) .* exp.(A_p * d[:p_]) + d[:ρ̅_p] .* exp.(A_p * d[:p̅])) -
                        A_p .* (d[:ρ̅_p] * d[:p̅] + (1. - d[:ρ̅_p]) * d[:p_])
                else
                    F .+= log.((1. - d[:ρ_p]) .* exp.(A_p * d[:p̅]) + d[:ρ_p] .* exp.(A_p * d[:p_])) -
                        A_p .* (d[:ρ_p] * d[:p_] + (1. - d[:ρ_p]) * d[:p̅])
                end
            end
        elseif m.disaster_intensity_spec == :LogAR1
            function _ccgf_bernoulli_logar1(F, A, z)
                F .= sum(A[:, vcat(not_dis_inds, SH[:ε_p])].^2, dims = 2) .* .5 # Gaussian parts
                A_k = @view A[:, SH[:ε_k]]
                F .+= log((1. - exp(z[S[:logp]]) + exp(z[S[:logp]])) .* exp.(A_k * d[:η_])) .-
                    A_k .* (d[:η_] * exp(z[S[:logp]])) # ε_k
            end
        end
    end

    if isnothing(ccgf)
        error("Either the specification of the disaster shock's occurrence ($(m.disaster_occur_spec)) or intensity " *
              "$(m.disaster_intensity_spec) is not recognized.")
    else
        return ccgf
    end
end
