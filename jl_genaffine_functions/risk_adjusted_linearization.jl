using ForwardDiff

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
mutable struct RiskAdjustedLinearization{M <: Function, L, S,
                                         X <: Function, V <: Function,
                                         Mz <: Function, My <: Function, Xz <: Function, Xy <: Function, J <: Function,
                                         VC <: AbstractVector{<: Number}, JC <: AbstractMatrix{<: Number},
                                         C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}}
    μ::M         # Functions
    Λ::L         # no type assertion for L b/c it can be Function or Matrix of zeros
    Σ::S         # no type assertion for S b/c it can be Function or constant Matrix
    ξ::X
    𝒱::V
    μz::Mz
    μy::My
    ξz::Xz
    ξy::Xy
    J𝒱::J
    μ_sss::VC    # Stochastic steady state values, for caching
    ξ_sss::VC
    𝒱_sss::VC
    Γ₁::JC       # Jacobians, for caching
    Γ₂::JC
    Γ₃::JC
    Γ₄::JC
    Γ₅::JC
    Γ₆::JC
    JV::JC
    z::C1        # Coefficients
    y::C1
    Ψ::C2
    Nz::Int      # Dimensions
    Ny::Int
    Nε::Int
end

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

# Constructor that uses ForwardDiff to calculate Jacobian functions
# NOTE THAT here we pass in the ccgf, rather than 𝒱
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int = -1) where {T <: Number, M <: Function, L,
                                                        S, X <: Function, CF <: Function}
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
    if applicable(ccgf, z) # Check if ccgf is in place or not
        𝒱 = function _𝒱(F, z, Ψ, Γ₅, Γ₆)
            F .= ccgf((Γ₅ + Γ₆ * Ψ) * ((I - Λ(z) * Ψ) \ Σ(z)), z)
        end
    else # in place
        𝒱 = (F, z, Ψ, Γ₅, Γ₆) -> ccgf(F, (Γ₅ + Γ₆ * Ψ) * ((I - Λ(z) * Ψ) \ Σ(z)), z)
    end
    J𝒱 = function _J𝒱(F, z, Ψ, Γ₅, Γ₆, 𝒱_sss)
        ForwardDiff.jacobian!(F, (G, x) -> 𝒱(G, x, Ψ, Γ₅, Γ₆), 𝒱_sss, z)
    end

    _check_inputs(z, y, Ψ, Γ₅, Γ₆, μ_sss, ξ_sss, 𝒱_sss, μ, Λ, Σ, ξ, 𝒱, μz, μy, ξz, ξy, J𝒱)

    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, 𝒱, μz, μy, ξz, ξy, J𝒱, μ_sss, ξ_sss, 𝒱_sss,
                                     Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, JV, z, y, Ψ, Nz, Ny, Nε)
end

# Create a function that takes in a ccgf and creates the associated entropy function, given the right inputs

# Rewrite the solution code to operate on jacobians, etc.

function _cache_jacobians(Ψ::AbstractMatrix{T}, Nz::Int, Ny::Int) where {T <: Number}

    Ψtype = typeof(Ψ)
    Γ₁ = convert(Ψtype, Matrix{T}(undef, Nz, Nz))
    Γ₂ = convert(Ψtype, Matrix{T}(undef, Nz, Ny))
    Γ₃ = similar(Ψ)
    Γ₄ = convert(Ψtype, Matrix{T}(undef, Ny, Ny))
    Γ₅ = similar(Ψ)
    Γ₆ = similar(Γ₄)
    JV = similar(Ψ)

    return Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, JV
end

function _cache_sss_vectors(z::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number}

    μ_sss = similar(z)
    ξ_sss = similar(y)
    𝒱_sss = similar(y)

   return μ_sss, ξ_sss, 𝒱_sss
end

function _check_inputs(z::C1, y::C1, Ψ::C2, Γ₅::JC, Γ₆::JC,
                       μ_sss::VC, ξ_sss::VC, 𝒱_sss::VC,
                       μ::M, Λ::L, Σ::S, ξ::X, 𝒱::V, μz::Mz,
                       μy::My, ξz::Xz, ξy::Xy, J𝒱::J)  where {C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number},
                                                              VC <: AbstractVector{<: Number}, JC <: AbstractMatrix{<: Number},
                                                              M <: Function, L, S,
                                                              X <: Function, V <: Function, Mz <: Function, My <: Function,
                                                              Xz <: Function, Xy <: Function, J <: Function}

    @assert applicable(μ, z, y) ||
        applicable(μ, z, z, y) "The function μ must take either the form " *
        "μ(z, y) or the in-place equivalent μ(F, z, y)"

    @assert applicable(ξ, z, y) ||
        applicable(ξ, z, z, y) "The function μ must take either the form " *
        "ξ(z, y) or the in-place equivalent ξ(F, z, y)"

     @assert applicable(Λ, z) ||
         applicable(Λ, Ψ, z) "The function Λ must take either the form Λ(z) or the in-place equivalent Λ(F, z)"

     @assert applicable(Σ, z) ||
         applicable(Σ, Ψ, z) "The function Λ must take either the form Σ(z) or the in-place equivalent Σ(F, z)"

    @assert applicable(𝒱, z, Ψ, Γ₅, Γ₆) ||
        applicable(𝒱, y, z, Ψ, Γ₅, Γ₆) "The function 𝒱 must take either the form " *
        "𝒱(z, Ψ, Γ₅, Γ₆) or the in-place equivalent 𝒱(F, z, Ψ, Γ₅, Γ₆)"

    @assert applicable(μz, z, y) ||
        applicable(μz, Ψ, z, y, μ_sss) "The function μz must take either the form " *
        "μz(z, y) or the in-place equivalent μz(F, z, y, μ_sss)"

    @assert applicable(μy, z, y) ||
        applicable(μy, Ψ, z, y, μ_sss) "The function μy must take either the form " *
        "μy(z, y) or the in-place equivalent μy(F, z, y, μ_sss)"

    @assert applicable(ξz, z, y) ||
        applicable(ξz, Ψ, z, y, ξ_sss) "The function ξz must take either the form " *
        "ξz(z, y) or the in-place equivalent ξz(F, z, y, ξ_sss)"

    @assert applicable(ξy, z, y) ||
        applicable(ξy, Ψ, z, y, ξ_sss) "The function ξy must take either the form " *
        "ξy(z, y) or the in-place equivalent ξy(F, z, y, ξ_sss)"

    @assert applicable(J𝒱, z, Ψ, Γ₅, Γ₆) ||
        applicable(J𝒱, Ψ, z, Ψ, Γ₅, Γ₆, 𝒱_sss) "The function J𝒱 must take either the form " *
        "J𝒱(z, Ψ, Γ₅, Γ₆) or the in-place equivalent J𝒱(F, z, Ψ, Γ₅, Γ₆, 𝒱_sss)"
end
