"""
    RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, 𝒱, Nz, Ny, Nε)

Creates a first-order perturbation around the stochastic steady state ``(z, y)`` of a discrete-time dynamic model.

(TODO: Move more of the formality to documentation, and make this shorter and concise, w/out explanation of matrix equations)
The affine approximation of the model is
``math
\\begin{aligned}
    \\mathbb{E}[z_{t + 1}] & = \\mu(z, y) + \\Gamma_1(z_t - z) + \\Gamma_2(y_t - y)\\
    0                      & = \\xi(z, y) + \\Gamma_3(z_t - z) + \\Gamma_4(y_t - y) + \\Gamma_5 \\mathbb{E}_t z_{t + 1} + \\Gamma_6 \\mathbb{E}_t y_{t + 1} + \\mathscr{V}(z) + J\\mathscr{V}(z)(z_t  - z),
\\end{aligned}
``
where ``\\Gamma_1, \Gamma_2`` are the Jacobians of ``\\mu`` with respect to ``z_t`` and ``y_t``, respectively;
``\\Gamma_3, \Gamma_4`` are the Jacobians of ``\\xi`` with respect to ``z_t`` and ``y_t``, respectively;
``\\Gamma_5, \\Gamma_6`` are constant matrices; ``\\mathscr{V}(z)`` is the model's entropy;
``J\\mathscr{V}(z)`` is the Jacobian of the entropy;
and the state variables ``z_t`` and jump variables ``y_t`` follow
``math
\\begin{aligned}
    z_{t + 1} & = z + \\Gamma_1(z_t - z) + \\Gamma_2(y_t - y) + (I_{n_z} - \\Lambda(z_t) \\Psi)^{-1}\\Sigma(z_t)\\varepsilon_{t + 1},\\
    y_t       & = y + \\Psi(z_t - z)
\\end{aligned}
``

The unknowns ``(z, y, \\Psi)`` solve the system of equations
``math
\\begin{aligned}
0 & = \\mu(z, y) - z\\
0 & = \\xi(z, y) + \\Gamma_5 z + \\Gamma_6 y + \\mathscr{V}(z)\\
0 & = \\Gamma_3 + \\Gamma_4 \\Psi + (\\Gamma_5 + \\Gamma_6 \\Psi)(\\Gamma_1 + \\Gamma_2 \\Psi) + J\\mathscr{V}(z),
\\end{aligned}
``

(TODO: Move the nonlinear model statement to documentation)
The true nonlinear equations defining model are assumed to take the form

``math
\\begin{aligned}
    z_{t + 1} & = \\mu(z_t, y_t) + \\Lambda(z_t)(y_{t + 1} - \\mathbb{E}_t y_{t + 1}) + \\Sigma(z_t) \\varepsilon_{t + 1}.
    0 & = \\log\\mathbb{E}_t[\\exp(\\xi(z_t, y_t) + \\Gamma_5 z_{t + 1} + \\Gamma_6 y_{t + 1})], \\\\
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
\\xi:\\mathbb{R}^{2n_y + 2n_z}\\rightarrow \\mathbb{R}^{n_y},& \\quad \\mu:\\mathbb{R}^{n_y + n_z}\\rightarrow \\mathbb{R}^{n_z},\\
\\Lambda::\\mathbb{R}^{n_z} \\rightarrow \\mathbb{R}^{n_z \\times n_y}, & \quad \\Sigma::\\mathbb{R}^{n_z}\\rightarrow \\mathbb{R}^{n_z\times n_\\varepsilon}
\\end{aligned}
are differentiable. The first two functions characterize the effects of time ``t`` variables on the expectational and
state transition equations. The function ``\\Lambda`` characterizes heteroskedastic endogenous risk that depends on
innovaitons in jump variables while the function ``\\Sigma`` characterizes exogenous risk.

Refer to Lopz et al. (2018) "Risk-Adjusted Linearizations of Dynamic Equilibrium Models" for details.
"""

mutable struct RiskAdjustedLinearization{T, M, L, S, X, V, J} where {T <: Number, M <: Function, L <: Function, S <: Function, X <: Function, V <: Function, J <: Function}
    μ::G                     # Functions
    Λ::L
    Σ::S
    ξ::X
    𝒱::V
    J𝒱::J
    Γ₁::AbstractMatrix{T}    # Jacobians
    Γ₂::AbstractMatrix{T}
    Γ₃::AbstractMatrix{T}
    Γ₄::AbstractMatrix{T}
    Γ₅::AbstractMatrix{T}
    Γ₆::AbstractMatrix{T}
    JV::AbstractMatrix{T}
    z::AbstractVector{T}     # Coefficients
    y::AbstractVector{T}
    Ψ::AbstractMatrix{T}
    Nz::Int                  # Dimensions
    Ny::Int
    Nε::Int
end

# Create a constructor function that automatically implements autodiff

# Create a function that takes in a ccgf and creates the associated entropy function, given the right inputs

# Rewrite the solution code to operate on jacobians, etc.
