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
    RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nε)
    RiskAdjustedLinearization(nonlinear_system, linearized_system, z, y, Ψ, Nz, Ny, Nε)


Creates a first-order perturbation around the stochastic steady state of a discrete-time dynamic economic model.

The first method is the main constructor most users will want, while the second method is the default constructor.

### Inputs for First Method
- `μ::Function`: expected state transition function
- `ξ::Function`: nonlinear terms of the expectational equations
- `ccgf::Function`: conditional cumulant generating function of the exogenous shocks
- `Λ::Function` or `Λ::AbstractMatrix`: function or matrix mapping endogenous risk into state transition equations
- `Σ::Function` or `Σ::AbstractMatrix`: function or matrix mapping exogenous risk into state transition equations
- `Γ₅::AbstractMatrix{<: Number}`: coefficient matrix on one-period ahead expectation of state variables
- `Γ₆::AbstractMatrix{<: Number}`: coefficient matrix on one-period ahead expectation of jump variables
- `z::AbstractVector{<: Number}`: state variables in stochastic steady state
- `y::AbstractVector{<: Number}`: jump variables in stochastic steady state
- `Ψ::AbstractMatrix{<: Number}`: matrix linking deviations in states to deviations in jumps, i.e. ``y_t - y = \\Psi(z_t - z)``.
- `Nε::Int`: number of exogenous shocks

### Inputs for Second Method
- `nonlinear_system::RALNonlinearSystem`
- `linearized_system::RALLinearizedSystem`
- `z::AbstractVector{<: Number}`: state variables in stochastic steady state
- `y::AbstractVector{<: Number}`: jump variables in stochastic steady state
- `Ψ::AbstractMatrix{<: Number}`: matrix linking deviations in states to deviations in jumps, i.e. ``y_t - y = \\Psi(z_t - z)``.
- `Nz::Int`: number of state variables
- `Ny::Int`: number of jump variables
- `Nε::Int`: number of exogenous shocks
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
# This is typically the main constructor for most users.
# Note that here we pass in the ccgf, rather than 𝒱
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
