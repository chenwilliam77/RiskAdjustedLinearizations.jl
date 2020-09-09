using ForwardDiff, UnPack, LinearAlgebra
using DiffEqBase: DiffCache, get_tmp, dualcache

mutable struct RALΛ{L <: Function, LC}
    Λ::L
    cache::LC
end

function RALΛ(Λ::Function, z::C1, matrix_type::DataType, dims::Tuple{Int, Int}) where {C1 <: AbstractVector{<: Number}}
    cache = matrix_type(undef, 0, 0) # Create empty matrix first, just to check if Λ is in place or not
    if applicable(Λ, cache, z)
        cache = matrix_type(undef, dims)
        Λnew = function _Λ_ip(cache::LCN, z::C1N) where {LCN <: DiffCache, C1N <: AbstractVector{<: Number}}
            Λ(get_tmp(cache, z), z)
            return get_tmp(cache, z)
        end
        return RALΛ(Λnew, dualcache(cache, Val{length(z)}))
    else
        function _Λ_oop(cache::LCN, z::C1N) where {LCN <: Nothing, C1N <: AbstractVector{<: Number}}
            return Λ(z)
        end
        return RALΛ(Λnew, nothing)
    end
end

function RALΛ(Λin::LC, z::C1) where {LC <: AbstractMatrix{<: Number}, C1 <: AbstractVector{<: Number}}
    Λ(cache::LCN, z::C1N) where {LCN <: AbstractMatrix{<: Number}, C1N <: AbstractVector{<: Number}} = cache
    return RALΛ{Function, LC}(Λ, Λin)
end

function (ralλ::RALΛ)(z::C1) where {C1 <: AbstractVector{<: Number}}
    return ralλ.Λ(ralλ.cache, z)
end

mutable struct RALΣ{S <: Function, SC}
    Σ::S
    cache::SC
end

function RALΣ(Σ::Function, z::C1, matrix_type::DataType, dims::Tuple{Int, Int}) where {C1 <: AbstractVector{<: Number}}
    cache = matrix_type(undef, 0, 0)
    if applicable(Σ, cache, z)
        cache = matrix_type(undef, dims)
        Σnew = function _Σ_ip(cache::SCN, z::C1N) where {SCN <: DiffCache, C1N <: AbstractVector{<: Number}}
            du = get_tmp(cache, z)
            Σ(du, z)
            return du
        end
        return RALΣ(Σnew, dualcache(cache, Val{length(z)}))
    else
        Σnew = function _Σ_oop(cache::SCN, z::C1N) where {SCN <: Nothing, C1N <: AbstractVector{<: Number}}
            return Σ(z)
        end
        return RALΣ(Σnew, nothing)
    end
end

function RALΣ(Σin::SC, z::C1) where {SC <: AbstractMatrix{<: Number}, C1 <: AbstractVector{<: Number}}
    Σ(cache::SCN, z::C1N) where {SCN <: AbstractMatrix{<: Number}, C1N <: AbstractVector{<: Number}} = cache
    return RALΣ{Function, SC}(Σ, Σin)
end

function (ralσ::RALΣ)(z::C1) where {C1 <: AbstractVector{<: Number}}
    return ralσ.Σ(ralσ.cache, z)
end

mutable struct RALNonlinearSystem{M <: Function, L <: RALΛ, S <: RALΣ, X <: Function, V <: Function,
                                  VC1 <: AbstractVector{<: Number}, VC2 <: AbstractVector{<: Number}, VC3 <: AbstractVector{<: Number}}
    μ::M         # Functions
    Λ::L         # no type assertion for L b/c it can be Function or Matrix of zeros
    Σ::S         # no type assertion for S b/c it can be Function or constant Matrix
    ξ::X
    𝒱::V
    μ_sss::VC1    # Stochastic steady state values, for caching
    ξ_sss::VC2
    𝒱_sss::VC3
    inplace::NamedTuple{(:μ, :ξ, :𝒱), NTuple{3, Bool}}
end

function RALNonlinearSystem(μ::M, Λ::L, Σ::S, ξ::X, 𝒱::V, μ_sss::VC1, ξ_sss::VC2, 𝒱_sss::VC3,
                            z::C1, y::C1, Ψ::C2, Γ₅::JC5, Γ₆::JC6) where {M <: Function, L <: RALΛ, S <: RALΣ, X <: Function, V <: Function,
                                                                          VC1 <: AbstractVector{<: Number}, VC2 <: AbstractVector{<: Number},
                                                                          VC3 <: AbstractVector{<: Number},
                                                                          C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number},
                                                                          JC5 <: AbstractMatrix{<: Number}, JC6 <: AbstractMatrix{<: Number}}

    inplace = (μ = applicable(μ, μ_sss, z, y), ξ = applicable(ξ, ξ_sss, z, y), 𝒱 = applicable(𝒱, 𝒱_sss, z, Ψ, Γ₅, Γ₆))

    return RALNonlinearSystem{M, L, S, X, V, VC1, VC2, VC3}(μ, Λ, Σ, ξ, 𝒱, μ_sss, ξ_sss, 𝒱_sss, inplace)
end

function update!(m::RALNonlinearSystem, z::C1, y::C1, Ψ::C2, Γ₅::JC5, Γ₆::JC6;
                 select::Vector{Symbol} = Symbol[:μ, :ξ, :𝒱]) where {C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number},
                                                                           JC5 <: AbstractMatrix{<: Number}, JC6 <: AbstractMatrix{<: Number}}

    if :μ in select
        if m.inplace[:μ]
            m.μ(m.μ_sss, z, y)
        else
            m.μ_sss .= m.μ(z, y)
        end
    end

    if :ξ in select
        if m.inplace[:ξ]
            m.ξ(m.ξ_sss, z, y)
        else
            m.ξ_sss .= m.ξ(z, y)
        end
    end

    if :𝒱 in select
        if m.inplace[:𝒱]
            m.𝒱(m.𝒱_sss, z, Ψ, Γ₅, Γ₆)
        else
            m.𝒱_sss .= m.𝒱(z, Ψ, Γ₅, Γ₆)
        end
    end

    m
end

mutable struct RALLinearizedSystem{Mz <: Function, My <: Function, Xz <: Function, Xy <: Function, J <: Function,
                                   JC1 <: AbstractMatrix{<: Number}, JC2 <: AbstractMatrix{<: Number},
                                   JC3 <: AbstractMatrix{<: Number}, JC4 <: AbstractMatrix{<: Number},
                                   JC5 <: AbstractMatrix{<: Number}, JC6 <: AbstractMatrix{<: Number},
                                   JC7 <: AbstractMatrix{<: Number}}
    μz::Mz     # Functions
    μy::My
    ξz::Xz
    ξy::Xy
    J𝒱::J
    Γ₁::JC1    # Jacobians, for caching
    Γ₂::JC2
    Γ₃::JC3
    Γ₄::JC4
    Γ₅::JC5
    Γ₆::JC6
    JV::JC7
    inplace::NamedTuple{(:μz, :μy, :ξz, :ξy, :J𝒱), NTuple{5, Bool}}
end

function RALLinearizedSystem(μz::Mz, μy::My, ξz::Xz, ξy::Xy, J𝒱::J,
                             Γ₁::JC1, Γ₂::JC2, Γ₃::JC3, Γ₄::JC4, Γ₅::JC5, Γ₆::JC6,
                             JV::JC7, z::C1, y::C1, Ψ::C2,
                             μ_sss::VC1, ξ_sss::VC2, 𝒱_sss::VC3) where {Mz <: Function, My <: Function, Xz <: Function,
                                                                        Xy <: Function, J <: Function,
                                                                        JC1 <: AbstractMatrix{<: Number}, JC2 <: AbstractMatrix{<: Number},
                                                                        JC3 <: AbstractMatrix{<: Number}, JC4 <: AbstractMatrix{<: Number},
                                                                        JC5 <: AbstractMatrix{<: Number}, JC6 <: AbstractMatrix{<: Number},
                                                                        JC7 <: AbstractMatrix{<: Number},
                                                                        C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number},
                                                                        VC1 <: AbstractVector{<: Number}, VC2 <: AbstractVector{<: Number},
                                                                        VC3 <: AbstractVector{<: Number},}

    inplace = (μz = applicable(μz, Γ₁, z, y, μ_sss), μy = applicable(μy, Γ₂, z, y, μ_sss), ξz = applicable(ξz, Γ₃, z, y, ξ_sss),
               ξy = applicable(ξy, Γ₄, z, y, ξ_sss), J𝒱 = applicable(J𝒱, JV, z, Ψ, Γ₅, Γ₆, 𝒱_sss))

    return RALLinearizedSystem(μz, μy, ξz, ξy, J𝒱, Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, JV, inplace)
end

function update!(m::RALLinearizedSystem, z::C1, y::C1, Ψ::C2,
                 μ_sss::VC1, ξ_sss::VC2, 𝒱_sss::VC3; select::Vector{Symbol} =
                 Symbol[:Γ₁, :Γ₂, :Γ₃, :Γ₄, :JV]) where {C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number},
                                                         VC1 <: AbstractVector{<: Number}, VC2 <: AbstractVector{<: Number},
                                                         VC3 <: AbstractVector{<: Number}}

    if :Γ₁ in select
        if m.inplace[:μz]
            m.μz(m.Γ₁, z, y, μ_sss)
        else
            m.μz(m.Γ₁, z, y)
        end
    end

    if :Γ₂ in select
        if m.inplace[:μy]
            m.μy(m.Γ₂, z, y, μ_sss)
        else
            m.μy(m.Γ₂, z, y)
        end
    end

    if :Γ₃ in select
        if m.inplace[:ξz]
            m.ξz(m.Γ₃, z, y, ξ_sss)
        else
            m.ξz(m.Γ₃, z, y)
        end
    end

    if :Γ₄ in select
        if m.inplace[:ξy]
            m.ξy(m.Γ₄, z, y, ξ_sss)
        else
            m.ξy(m.Γ₄, z, y)
        end
    end

    if :JV in select
        if m.inplace[:J𝒱]
            m.J𝒱(m.JV, z, Ψ, m.Γ₅, m.Γ₆, 𝒱_sss)
        else
            m.J𝒱(m.JV, z, Ψ, m.Γ₅, m.Γ₆)
        end
    end

    m
end

abstract type AbstractRiskAdjustedLinearization end

"""
    RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, 𝒱, Nz, Ny, Nε)

Creates a first-order perturbation around the stochastic steady state ``(z, y)`` of a discrete-time dynamic model.

(TODO: Move more of the formality to documentation, and make this shorter and concise, w/out explanation of matrix equations)
The affine approximation of the model is
``math
\\begin{aligned}
    \\mathbb{E}[z_{t + 1}] & = \\mu(z, y) + \\Gamma_1(z_t - z) + \\Gamma_2(y_t - y)\\\\
    0                      & = \\xi(z, y) + \\Gamma_3(z_t - z) + \\Gamma_4(y_t - y) + \\Gamma_5 \\mathbb{E}_t z_{t + 1} + \\Gamma_6 \\mathbb{E}_t y_{t + 1} + \\mathscr{V}(z) + J\\mathscr{V}(z)(z_t  - z),
\\end{aligned}
``

where ``\\Gamma_1, \\Gamma_2`` are the Jacobians of ``\\mu`` with respect to ``z_t`` and ``y_t``, respectively;
``\\Gamma_3, \\Gamma_4`` are the Jacobians of ``\\xi`` with respect to ``z_t`` and ``y_t``, respectively;
``\\Gamma_5, \\Gamma_6`` are constant matrices; ``\\mathscr{V}(z)`` is the model's entropy;
``J\\mathscr{V}(z)`` is the Jacobian of the entropy;

and the state variables ``z_t`` and jump variables ``y_t`` follow
``math
\\begin{aligned}
    z_{t + 1} & = z + \\Gamma_1(z_t - z) + \\Gamma_2(y_t - y) + (I_{n_z} - \\Lambda(z_t) \\Psi)^{-1}\\Sigma(z_t)\\varepsilon_{t + 1},\\\\
    y_t       & = y + \\Psi(z_t - z)
\\end{aligned}
``

The unknowns ``(z, y, \\Psi)`` solve the system of equations
``math
\\begin{aligned}
0 & = \\mu(z, y) - z,\\\\
0 & = \\xi(z, y) + \\Gamma_5 z + \\Gamma_6 y + \\mathscr{V}(z),\\\\
0 & = \\Gamma_3 + \\Gamma_4 \\Psi + (\\Gamma_5 + \\Gamma_6 \\Psi)(\\Gamma_1 + \\Gamma_2 \\Psi) + J\\mathscr{V}(z).
\\end{aligned}
``
(TODO: Move the nonlinear model statement to documentation)
The true nonlinear equations defining model are assumed to take the form

``math
\\begin{aligned}
    z_{t + 1} & = \\mu(z_t, y_t) + \\Lambda(z_t)(y_{t + 1} - \\mathbb{E}_t y_{t + 1}) + \\Sigma(z_t) \\varepsilon_{t + 1},\\\\
    0 & = \\log\\mathbb{E}_t[\\exp(\\xi(z_t, y_t) + \\Gamma_5 z_{t + 1} + \\Gamma_6 y_{t + 1})].
\\end{aligned}
``

The vectors ``z_t\\in \\mathbb{R}^{n_z}`` and ``y_t \\in \\mathbb{R}^{n_y}`` are the state and jump variables, respectively.
The first vector equation comprise the model's expectational equations, which are typically
the first-order conditions for the jump variables from agents' optimization problem.
The second vector equation comprise the transition equations of the state variables. The exogenous shocks
``\\varepsilon\\in\\mathbb{R}^{n_\\varepsilon}`` form a martingale difference sequence whose distribution
is described by the differentiable, conditional cumulant generating function (ccgf)

``math
\\begin{aligned}
\\kappa[\\alpha(z_t) \\mid z_t] = \\log\\mathbb{E}_t[\\exp(\\alpha(z_t)' \\varepsilon_{t + 1})],\\quad \text{ for any differentiable map }\\alpha::\\mathbb{R}^{n_z}\\rightarrow\\mathbb{R}^{n_\\varepsilon}.
\\end{aligned}
``

The functions
``math
\\begin{aligned}
\\xi:\\mathbb{R}^{2n_y + 2n_z}\\rightarrow \\mathbb{R}^{n_y},& \\quad \\mu:\\mathbb{R}^{n_y + n_z}\\rightarrow \\mathbb{R}^{n_z},\\\\
\\Lambda::\\mathbb{R}^{n_z} \\rightarrow \\mathbb{R}^{n_z \\times n_y}, & \\quad \\Sigma::\\mathbb{R}^{n_z}\\rightarrow \\mathbb{R}^{n_z\\times n_\\varepsilon}
\\end{aligned}
are differentiable. The first two functions characterize the effects of time ``t`` variables on the expectational and
state transition equations. The function ``\\Lambda`` characterizes heteroskedastic endogenous risk that depends on
innovations in jump variables while the function ``\\Sigma`` characterizes exogenous risk.

Refer to Lopz et al. (2018) "Risk-Adjusted Linearizations of Dynamic Equilibrium Models" for details.
"""
mutable struct RiskAdjustedLinearization{A <: RALNonlinearSystem, B <: RALLinearizedSystem,
                                         C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}} <: AbstractRiskAdjustedLinearization
    nonlinear::A
    linearization::B
    z::C1        # Coefficients
    y::C1
    Ψ::C2
    Nz::Int      # Dimensions
    Ny::Int
    Nε::Int
end
# TODO
# 1.UPDATE THE PRINTING, maybe just write out "risk-adjusted linearization with dimensions ()"
#
# 2. Test update! functions for the various blocks as well as access functions for RiskAdjustedLinearization
#
# 3. Check inplace inference is correct, check construction of each block plus main block
#=
TODO: Finish this once the final struct is completed
# A series of lower level constructors
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, 𝒱::V, μz::Mz, μy::My, ξz::Xz, ξy::Xy, J𝒱::J,
                                   μ_sss::AbstractVector{T}, ξ_sss::AbstractVector{T}, 𝒱_sss::AbstractVector{T},
                                   Γ₁::AbstractMatrix{T}, Γ₂::AbstractMatrix{T}, Γ₃::AbstractMatrix{T}
                                   Γ₄::AbstractMatrix{T}, Γ₅::AbstractMatrix{T}, Γ₆::AbstractMatrix{T},
                                   JV::AbstractMatrix{T}, z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int = -1) where {T <: Number, M <: Function, L,
                                                        S, X <: Function, V <: Function,
                                                        Mz <: Function, My <: Function, Xz <: Function,
                                                        Xy <: Function, J <: Function}

    Nz = length(z)
    Ny = length(y)
    if Nε < 0
        Nε = size(Σ(z), 2)
    end

    return RiskAdjustedLinearization{T, M, L, S, X, V, J}(μ, Λ, Σ, ξ, 𝒱, J𝒱, μ_sss, ξ_sss, 𝒱_sss,
                                                          Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆,
                                                          JV, z, y, Ψ, Nz, Ny, Nε)
end


function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, 𝒱::V, μz::Mz, μy::My, ξz::Xz, ξy::Xy, J𝒱::J,
                                   Γ₁::AbstractMatrix{T}, Γ₂::AbstractMatrix{T}, Γ₃::AbstractMatrix{T}
                                   Γ₄::AbstractMatrix{T}, Γ₅::AbstractMatrix{T}, Γ₆::AbstractMatrix{T},
                                   JV::AbstractMatrix{T}, z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int = -1) where {T <: Number, M <: Function, L,
                                                        S, X <: Function, V <: Function,
                                                        Mz <: Function, My <: Function, Xz <: Function,
                                                        Xy <: Function, J <: Function}
    Nz = length(z)
    Ny = length(y)
    if Nε < 0
        Nε = size(Σ(z), 2)
    end

    # Cache stochastic steady state vectors
    μ_sss, ξ_sss, 𝒱_sss = _cache_sss_vectors(z, y)

    return RiskAdjustedLinearization{T, M, L, S, X, V, J}(μ, Λ, Σ, ξ, 𝒱, J𝒱, μ_sss, ξ_sss, 𝒱_sss,
                                                          Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆,
                                                          JV, z, y, Ψ, Nz, Ny, Nε)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, 𝒱::V, μz::Mz, μy::My, ξz::Xz, ξy::Xy, J𝒱::J,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int = -1) where {T <: Number, M <: Function, L,
                                                        S, X <: Function, V <: Function,
                                                        Mz <: Function, My <: Function, Xz <: Function,
                                                        Xy <: Function, J <: Function}
    # Get dimensions
    Nz = length(z)
    Ny = length(y)
    if Nε < 0
        Nε = size(Σ(z), 2)
    end

    # Cache stochastic steady state vectors
    μ_sss, ξ_sss, 𝒱_sss = _cache_sss_vectors(z, y)

    # Cache stochastic steady state Jacobians
    Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, JV = _cache_jacobians(Ψ, Nz, Ny)

    return RiskAdjustedLinearization{T, M, L, S, X, V, J}(μ, Λ, Σ, ξ, 𝒱, J𝒱, μ_sss, ξ_sss, 𝒱_sss,
                                                          Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆,
                                                          JV, z, y, Ψ, Nz, Ny, Nε)
end
=#
function RiskAdjustedLinearization(nonlinear::A, linearization::B, z::C1, y::C1, Ψ::C2,
                                   Nz::Int, Ny::Int, Nε::Int;
                                   check_inputs::Bool = true) where {A <: RALNonlinearSystem, B <: RALLinearizedSystem,
                                                                     C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}}

    # Make sure inputs are well-formed
    if check_inputs
        _check_inputs(nonlinear, linearization, z, y, Ψ)
    end

    return RiskAdjustedLinearization{A, B, C1, C2}(nonlinear, linearization, z, y, Ψ, Nz, Ny, Nε)
end

# Constructor that uses ForwardDiff to calculate Jacobian functions
# NOTE THAT here we pass in the ccgf, rather than 𝒱
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_type::DataType = Vector{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: Function, L <: RALΛ, S <: RALΣ,
                                                                               X <: Function,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}

    # Cache stochastic steady state vectors
    μ_sss, ξ_sss, 𝒱_sss = _cache_sss_vectors(z, y)

    # Cache stochastic steady state Jacobians
    Γ₁, Γ₂, Γ₃, Γ₄, JV = _cache_jacobians(Ψ, Nz, Ny, jacobian_type)

    # Use cached Jacobians to create Jacobian functions for μ, ξ
    if applicable(μ, z, y) # Check if μ is in place or not
        μz = (F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(x, y), z) # not in place
        μy = (F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(z, x), y)
    else # in place
        μz = (F, z, y, μ_sss) -> ForwardDiff.jacobian!(F, (G, x) -> μ(G, x, y), μ_sss, z)
        μy = (F, z, y, μ_sss) -> ForwardDiff.jacobian!(F, (G, x) -> μ(G, z, x), μ_sss, y)
    end

    if applicable(ξ, z, y) # Check if ξ is in place or not
        ξz = (F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(x, y), z) # not in place
        ξy = (F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(z, x), y)
    else # in place
        ξz = (F, z, y, ξ_sss) -> ForwardDiff.jacobian!(F, (G, x) -> ξ(G, x, y), ξ_sss, z)
        ξy = (F, z, y, ξ_sss) -> ForwardDiff.jacobian!(F, (G, x) -> ξ(G, z, x), ξ_sss, y)
    end

    # Create 𝒱 and its Jacobian J𝒱
    if applicable(ccgf, Γ₅, z) # Check if ccgf is in place or not
        𝒱 = function _𝒱(F, z, Ψ, Γ₅, Γ₆)
            F .= ccgf((Γ₅ + Γ₆ * Ψ) * ((I - Λ(z) * Ψ) \ Σ(z)), z)
        end
    else # in place
        𝒱 = (F, z, Ψ, Γ₅, Γ₆) -> ccgf(F, (Γ₅ + Γ₆ * Ψ) * ((I - Λ(z) * Ψ) \ Σ(z)), z)
    end
    J𝒱 = function _J𝒱(F, z, Ψ, Γ₅, Γ₆, 𝒱_sss)
        ForwardDiff.jacobian!(F, (G, x) -> 𝒱(G, x, Ψ, Γ₅, Γ₆), 𝒱_sss, z)
    end

    # Form underlying RAL blocks
    nonlinear_system  = RALNonlinearSystem(μ, Λ, Σ, ξ, 𝒱, μ_sss, ξ_sss, 𝒱_sss, z, y, Ψ, Γ₅, Γ₆)
    linearized_system = RALLinearizedSystem(μz, μy, ξz, ξy, J𝒱, Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, JV, z, y, Ψ, μ_sss, ξ_sss, 𝒱_sss)

    return RiskAdjustedLinearization(nonlinear_system, linearized_system, z, y, Ψ, Nz, Ny, Nε)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int; sss_vector_type::DataType = Vector{T}, sss_matrix_type::DataType = Matrix{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: Function, L <: Function, S <: Function,
                                                                               X <: Function,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}
    # Get dimensions
    Nz = length(z)
    Ny = length(y)
    if Nε < 0
        error("Nε cannot be negative")
    end

    # Create wrappers enabling caching for Λ and Σ
    Λ = RALΛ(Λ, z, sss_matrix_type, (Nz, Ny))
    Σ = RALΣ(Σ, z, sss_matrix_type, (Nz, Nε))

    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_type = sss_vector_type,
                                     jacobian_type = jacobian_type)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int = -1; sss_vector_type::DataType = Vector{T}, sss_matrix_type::DataType = Matrix{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: Function, L <: AbstractMatrix{<: Number}, S <: Function,
                                                                               X <: Function,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}
    # Get dimensions
    Nz = length(z)
    Ny = length(y)
    if Nε < 0
        error("Nε cannot be negative")
    end

    # Create wrappers enabling caching for Λ and Σ
    Λ = RALΛ(Λ, z)
    Σ = RALΣ(Σ, z, sss_matrix_type, (Nz, Nε))

    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_type = sss_vector_type,
                                     jacobian_type = jacobian_type)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int = -1; sss_vector_type::DataType = Vector{T}, sss_matrix_type::DataType = Matrix{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: Function, L <: Function, S <: AbstractMatrix{<: Number},
                                                                               X <: Function,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}
    # Get dimensions
    Nz = length(z)
    Ny = length(y)
    if Nε < 0
        error("Nε cannot be negative")
    end

    # Create wrappers enabling caching for Λ and Σ
    Λ = RALΛ(Λ, z, sss_matrix_type, (Nz, Ny))
    Σ = RALΣ(Σ, z)

    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_type = sss_vector_type,
                                     jacobian_type = jacobian_type)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int = -1; sss_vector_type::DataType = Vector{T}, sss_matrix_type::DataType = Matrix{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: Function,
                                                                               L <: AbstractMatrix{<: Number}, S <: AbstractMatrix{<: Number},
                                                                               X <: Function,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}
    # Get dimensions
    Nz = length(z)
    Ny = length(y)
    if Nε < 0
        error("Nε cannot be negative")
    end

    # Create wrappers enabling caching for Λ and Σ
    Λ = RALΛ(Λ, z)
    Σ = RALΣ(Σ, z)

    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_type = sss_vector_type,
                                     jacobian_type = jacobian_type)
end

function _cache_jacobians(Ψ::AbstractMatrix{T}, Nz::Int, Ny::Int, mat_type::DataType) where {T <: Number}

    Γ₁ = mat_type(undef, Nz, Nz)
    Γ₂ = mat_type(undef, Nz, Ny)
    Γ₃ = similar(Ψ)
    Γ₄ = mat_type(undef, Ny, Ny)
    JV = similar(Ψ)

    return Γ₁, Γ₂, Γ₃, Γ₄, JV
end

function _cache_sss_vectors(z::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number, L, S}

    μ_sss = similar(z)
    ξ_sss = similar(y)
    𝒱_sss = similar(y)

   return μ_sss, ξ_sss, 𝒱_sss
end

function _check_inputs(nonlinear::A, linearization::B, z::C1, y::C1, Ψ::C2) where {A <: RALNonlinearSystem, B <: RALLinearizedSystem,
                                                                                   C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}}

    # Get contents of nonlinear and linearization blocks
    @unpack μ, ξ, 𝒱, μ_sss, ξ_sss, 𝒱_sss = nonlinear
    @unpack μz, μy, ξz, ξy, J𝒱, Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, JV = linearization

    @assert applicable(μ, z, y) ||
        applicable(μ, μ_sss, z, y) "The function μ must take either the form " *
        "μ(z, y) or the in-place equivalent μ(F, z, y)"

    @assert applicable(ξ, z, y) ||
        applicable(ξ, ξ_sss, z, y) "The function μ must take either the form " *
        "ξ(z, y) or the in-place equivalent ξ(F, z, y)"

    @assert applicable(𝒱, z, Ψ, Γ₅, Γ₆) ||
        applicable(𝒱, y, z, Ψ, Γ₅, Γ₆) "The function 𝒱 must take either the form " *
        "𝒱(z, Ψ, Γ₅, Γ₆) or the in-place equivalent 𝒱(F, z, Ψ, Γ₅, Γ₆)"

    @assert applicable(μz, Γ₁, z, y) ||
        applicable(μz, Γ₁, z, y, μ_sss) "The function μz must take either the form " *
        "μz(F, z, y) or μz(F, z, y, μ_sss)"

    @assert applicable(μy, Γ₂, z, y) ||
        applicable(μy, Γ₂, z, y, μ_sss) "The function μy must take either the form " *
        "μy(F, z, y) or μy(F, z, y, μ_sss)"

    @assert applicable(ξz, Γ₃, z, y) ||
        applicable(ξz, Γ₃, z, y, ξ_sss) "The function ξz must take either the form " *
        "ξz(F, z, y) or ξz(F, z, y, ξ_sss)"

    @assert applicable(ξy, Γ₄, z, y) ||
        applicable(ξy, Γ₄, z, y, ξ_sss) "The function ξy must take either the form " *
        "ξy(F, z, y) or ξy(F, z, y, ξ_sss)"

    @assert applicable(J𝒱, z, Ψ, Γ₅, Γ₆) ||
        applicable(J𝒱, JV, z, Ψ, Γ₅, Γ₆, 𝒱_sss) "The function J𝒱 must take either the form " *
        "J𝒱(F, z, Ψ, Γ₅, Γ₆) or J𝒱(F, z, Ψ, Γ₅, Γ₆, 𝒱_sss)"
end

## Methods for using RiskAdjustedLinearization
@inline Γ₁(m::RiskAdjustedLinearization) = m.linearization.Γ₁
@inline Γ₂(m::RiskAdjustedLinearization) = m.linearization.Γ₂
@inline Γ₃(m::RiskAdjustedLinearization) = m.linearization.Γ₃
@inline Γ₄(m::RiskAdjustedLinearization) = m.linearization.Γ₄
@inline Γ₅(m::RiskAdjustedLinearization) = m.linearization.Γ₅
@inline Γ₆(m::RiskAdjustedLinearization) = m.linearization.Γ₆
@inline JV(m::RiskAdjustedLinearization) = m.linearization.JV
@inline getvalues(m::RiskAdjustedLinearization) = (m.z, m.y, m.Ψ)
@inline getvecvalues(m::RiskAdjustedLinearization) = vcat(m.z, m.y, vec(m.Ψ))
@inline nonlinear_system(m::RiskAdjustedLinearization) = m.nonlinear
@inline linearized_system(m::RiskAdjustedLinearization) = m.linearization

function update!(m::RiskAdjustedLinearization)
    update!(nonlinear_system(m), m.z, m.y, m.Ψ, Γ₅(m), Γ₆(m))
    update!(linearized_system(m), m.z, m.y, m.Ψ, m.nonlinear.μ_sss, m.nonlinear.ξ_sss, m.nonlinear.𝒱_sss)
end


function update!(m::RiskAdjustedLinearization, z::C1, y::C1, Ψ::C2;
                 update_cache::Bool = true) where {C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}}

    # Update values of the affine approximation
    m.z .= z
    m.y .= y
    m.Ψ .= Ψ

    # Update the cached vectors and Jacobians
    if update_cache
        update!(m)
    end

    m
end

function Base.show(io::IO, m::RiskAdjustedLinearization)
    @printf io "Risk-Adjusted Linearization of an Economic Model\n"
    @printf io "No. of state variables:      %i\n" m.Nz
    @printf io "No. of jump variables:       %i\n" m.Ny
    @printf io "No. of exogenous shocks:     %i\n" m.Nε
end

function Base.show(io::IO, m::RALNonlinearSystem)
    @printf io "RALNonlinearSystem"
end

function Base.show(io::IO, m::RALLinearizedSystem)
    @printf io "RALLinearizedSystem"
end
