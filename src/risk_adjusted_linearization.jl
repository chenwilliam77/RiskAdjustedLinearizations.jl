# Subtypes used for the main RiskAdjustedLinearization type
mutable struct RALNonlinearSystem{L <: AbstractRALF, S <: AbstractRALF, V <: AbstractRALF}
    μ::RALF2
    Λ::L
    Σ::S
    ξ::RALF2
    𝒱::V
end

Λ_eltype(m::RALNonlinearSystem{L, S}) where {L, S} = L
Σ_eltype(m::RALNonlinearSystem{L, S}) where {L, S} = S

function update!(m::RALNonlinearSystem, z::C1, y::C1, Ψ::C2;
                 select::Vector{Symbol} = Symbol[:μ, :ξ, :𝒱]) where {C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}}

    if :μ in select
        m.μ(z, y)
    end

    if :ξ in select
        m.ξ(z, y)
    end

    if :𝒱 in select
        if isa(m.𝒱, RALF2)
            m.𝒱(z, Ψ)
        else
            m.𝒱(z, y, Ψ, z)
        end
    end

    m
end

mutable struct RALLinearizedSystem{JV <: AbstractRALF, JC5 <: AbstractMatrix{<: Number}, JC6 <: AbstractMatrix{<: Number}}
    μz::RALF2
    μy::RALF2
    ξz::RALF2
    ξy::RALF2
    J𝒱::JV
    Γ₅::JC5
    Γ₆::JC6
end

function update!(m::RALLinearizedSystem, z::C1, y::C1, Ψ::C2;
                 select::Vector{Symbol} =
                 Symbol[:Γ₁, :Γ₂, :Γ₃, :Γ₄, :JV]) where {C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}}

    if :Γ₁ in select
        m.μz(z, y)
    end

    if :Γ₂ in select
        m.μy(z, y)
    end

    if :Γ₃ in select
        m.ξz(z, y)
    end

    if :Γ₄ in select
        m.ξy(z, y)
    end

    if :JV in select
        if isa(m.J𝒱, RALF2)
            m.J𝒱(z, Ψ)
        else
            m.J𝒱(z, y, Ψ)
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
mutable struct RiskAdjustedLinearization{C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}} <: AbstractRiskAdjustedLinearization

    nonlinear::RALNonlinearSystem
    linearization::RALLinearizedSystem
    z::C1 # Coefficients, TODO: at some point, we may or may not want to make z, y, and Ψ also DiffCache types
    y::C1
    Ψ::C2
    Nz::Int      # Dimensions
    Ny::Int
    Nε::Int
end

# The following constructor is typically the main constructor for most users.
# It will call a lower-level constructor that uses automatic differentiation
# to calculate the Jacobian functions.
# Note that here we pass in the ccgf, rather than 𝒱
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nε::Int; sss_vector_type::DataType = Vector{T}, Λ_Σ_type::DataType = Matrix{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: Function, L, S,
                                                                               X <: Function,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}
    # Get dimensions
    Nz  = length(z)
    Ny  = length(y)
    Nzy = Nz + Ny
    if Nε < 0
        throw(BoundsError("Nε cannot be negative"))
    end

    # Create wrappers enabling caching for μ and ξ
    Nzchunk = ForwardDiff.pickchunksize(Nz)
    Nychunk = ForwardDiff.pickchunksize(Ny)
    _μ = RALF2(μ, z, y, sss_vector_type, (Nz, ), (max(min(Nzchunk, Nychunk), 2), Nzchunk, Nychunk))
    _ξ = RALF2(ξ, z, y, sss_vector_type, (Ny, ), (max(min(Nzchunk, Nychunk), 2), Nzchunk, Nychunk))

    # Apply dispatch on Λ and Σ to figure what they should be
    return RiskAdjustedLinearization(_μ, Λ, Σ, _ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_type = sss_vector_type,
                                     jacobian_type = jacobian_type)
end

# Constructor that uses ForwardDiff to calculate Jacobian functions.
# Users will not typically use this constructor, however, because it requires
# various functions of the RALNonlinearSystem and RALLinearizedSystem to already
# be wrapped with either an RALF1 or RALF2 type.
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_type::DataType = Vector{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: RALF2, L <: RALF1, S <: RALF1,
                                                                               X <: RALF2,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}

    # Use RALF2 wrapper to create Jacobian functions with caching for μ, ξ.
    # Use the tuple to select the correct Dual cache b/c μ is in place
    μz = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(x, y, (1, 2)), z), z, y,
               jacobian_type, (Nz, Nz))
    μy = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(z, x, (2, 3)), y), z, y,
               jacobian_type, (Nz, Ny))

    ξz = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(x, y, (1, 2)), z), z, y,
               jacobian_type, (Ny, Nz))
    ξy = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(z, x, (2, 3)), y), z, y,
               jacobian_type, (Ny, Ny))

    # Create RALF2 wrappers for 𝒱 and its Jacobian J𝒱
    if applicable(ccgf, Γ₅, z) # Check if ccgf is in place or not
        _𝒱 = function _𝒱_oop(F, z, Ψ)
            F .= ccgf((Γ₅ + Γ₆ * Ψ) * ((I - Λ(z) * Ψ) \ Σ(z)), z)
        end
    else # in place
        _𝒱 = (F, z, Ψ) -> ccgf(F, (Γ₅ + Γ₆ * Ψ) * ((I - Λ(z) * Ψ) \ Σ(z)), z)
    end
    Nzchunk = ForwardDiff.pickchunksize(Nz)
    Nychunk = ForwardDiff.pickchunksize(Ny)
    𝒱 = RALF2((F, z, Ψ) -> _𝒱(F, z, Ψ), z, Ψ, sss_vector_type, (Ny, ), (max(min(Nzchunk, Nychunk), 2), Nzchunk))

    _J𝒱(F, z, Ψ) = ForwardDiff.jacobian!(F, x -> 𝒱(x, Ψ, (1, 2)), z)
    J𝒱           = RALF2((F, z, Ψ) -> _J𝒱(F, z, Ψ), z, Ψ, jacobian_type, (Ny, Nz))

    # Form underlying RAL blocks
    nonlinear_system  = RALNonlinearSystem(μ, Λ, Σ, ξ, 𝒱)
    linearized_system = RALLinearizedSystem(μz, μy, ξz, ξy, J𝒱, Γ₅, Γ₆)

    return RiskAdjustedLinearization(nonlinear_system, linearized_system, z, y, Ψ, Nz, Ny, Nε)
end

# Handles case where Λ and Σ are RALF2
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_type::DataType = Vector{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: RALF2, L <: RALF2, S <: RALF2,
                                                                               X <: RALF2,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}

    # Use RALF2 wrapper to create Jacobian functions with caching for μ, ξ.
    # Use the tuple to select the correct Dual cache b/c μ is in place
    μz = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(x, y, (1, 2)), z), z, y,
               jacobian_type, (Nz, Nz))
    μy = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(z, x, (2, 3)), y), z, y,
               jacobian_type, (Nz, Ny))

    ξz = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(x, y, (1, 2)), z), z, y,
               jacobian_type, (Ny, Nz))
    ξy = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(z, x, (2, 3)), y), z, y,
               jacobian_type, (Ny, Ny))

    # Create RALF2 wrappers for 𝒱 and its Jacobian J𝒱
    if applicable(ccgf, Γ₅, z) # Check if ccgf is in place or not
        _𝒱 = function _𝒱_oop(F, z, y, Ψ, zₜ)
            yₜ = y + Ψ * (zₜ - z)
            F .= ccgf((Γ₅ + Γ₆ * Ψ) * ((I - Λ(zₜ, yₜ) * Ψ) \ Σ(zₜ, yₜ)), zₜ)
        end
    else # in place
        _𝒱 = function _𝒱_ip(F, z, y, Ψ, zₜ)
            yₜ = y + Ψ * (zₜ - z)
            ccgf(F, (Γ₅ + Γ₆ * Ψ) * ((I - Λ(zₜ, yₜ) * Ψ) \ Σ(zₜ, yₜ)), zₜ)
        end
    end
    Nzchunk = ForwardDiff.pickchunksize(Nz)
    Nychunk = ForwardDiff.pickchunksize(Ny)
    𝒱       = RALF4((F, z, y, Ψ, zₜ) -> _𝒱(F, z, y, Ψ, zₜ), z, y, Ψ, z, sss_vector_type, (Ny, ), (max(min(Nzchunk, Nychunk), 2), Nzchunk))

    _J𝒱(F, z, y, Ψ) = ForwardDiff.jacobian!(F, zₜ -> 𝒱(z, y, Ψ, zₜ, (4, 2)), z) # use zₜ argument to infer the cache
    J𝒱              = RALF3((F, z, y, Ψ) -> _J𝒱(F, z, y, Ψ), z, y, Ψ, jacobian_type, (Ny, Nz))

    # Form underlying RAL blocks
    nonlinear_system  = RALNonlinearSystem(μ, Λ, Σ, ξ, 𝒱)
    linearized_system = RALLinearizedSystem(μz, μy, ξz, ξy, J𝒱, Γ₅, Γ₆)

    return RiskAdjustedLinearization(nonlinear_system, linearized_system, z, y, Ψ, Nz, Ny, Nε)
end

# The following four constructors cover different common cases for the Λ and Σ functions.
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_type::DataType = Vector{T}, Λ_Σ_type::DataType = Matrix{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: RALF2, L <: Function, S <: Function,
                                                                               X <: RALF2,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}
    # Create wrappers enabling caching for Λ and Σ
    Nzchunk = ForwardDiff.pickchunksize(Nz)
    Nychunk = ForwardDiff.pickchunksize(Ny)
    if applicable(Λ, z, y)
        _Λ = RALF2(Λ, z, y, Λ_Σ_type, (Nz, Ny), (max(min(Nzchunk, Nychunk), 2), Nzchunk))
    else
        _Λ = RALF1(Λ, z, Λ_Σ_type, (Nz, Ny))
    end
    if applicable(Σ, z, y)
        _Σ = RALF2(Σ, z, y, Λ_Σ_type, (Nz, Nε), (max(min(Nzchunk, Nychunk), 2), Nzchunk))
    else
        _Σ = RALF1(Σ, z, Λ_Σ_type, (Nz, Nε))
    end

    return RiskAdjustedLinearization(μ, _Λ, _Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_type = sss_vector_type,
                                     jacobian_type = jacobian_type)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_type::DataType = Vector{T}, Λ_Σ_type::DataType = Matrix{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: RALF2, L <: AbstractMatrix{<: Number}, S <: Function,
                                                                               X <: RALF2,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}

    # Create wrappers enabling caching for Λ and Σ
    _Λ = RALF1(Λ)
    _Σ = RALF1(Σ, z, Λ_Σ_type, (Nz, Nε))

    return RiskAdjustedLinearization(μ, _Λ, _Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_type = sss_vector_type,
                                     jacobian_type = jacobian_type)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_type::DataType = Vector{T}, Λ_Σ_type::DataType = Matrix{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: RALF2, L <: Function, S <: AbstractMatrix{<: Number},
                                                                               X <: RALF2,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}

    # Create wrappers enabling caching for Λ and Σ
    Nzchunk = ForwardDiff.pickchunksize(Nz)
    Nychunk = ForwardDiff.pickchunksize(Ny)
    if applicable(Λ, z, y)
        _Λ = RALF2(Λ, z, y, Λ_Σ_type, (Nz, Ny), (max(min(Nzchunk, Nychunk), 2), Nzchunk))
    else
        _Λ = RALF1(Λ, z, Λ_Σ_type, (Nz, Ny))
    end
    _Σ = RALF1(Σ)

    return RiskAdjustedLinearization(μ, _Λ, _Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_type = sss_vector_type,
                                     jacobian_type = jacobian_type)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_type::DataType = Vector{T}, Λ_Σ_type::DataType = Matrix{T},
                                   jacobian_type::DataType = Matrix{T}) where {T <: Number, M <: RALF2,
                                                                               L <: AbstractMatrix{<: Number}, S <: AbstractMatrix{<: Number},
                                                                               X <: RALF2,
                                                                               JC5 <: AbstractMatrix{<: Number},
                                                                               JC6 <: AbstractMatrix{<: Number},
                                                                               CF <: Function}

    # Create wrappers enabling caching for Λ and Σ
    _Λ = RALF1(Λ)
    _Σ = RALF1(Σ)

    return RiskAdjustedLinearization(μ, _Λ, _Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_type = sss_vector_type,
                                     jacobian_type = jacobian_type)
end

## Print statements for RAL objects
function Base.show(io::IO, m::AbstractRiskAdjustedLinearization)
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

## Indexing for convenient access to steady state values
function Base.getindex(m::RiskAdjustedLinearization, sym::Symbol)
    if sym in [:μ_sss, :ξ_sss, :𝒱_sss, :Σ_sss, :Λ_sss]
        m.nonlinear[sym]
    elseif sym in [:Γ₁, :Γ₂, :Γ₃, :Γ₄, :Γ₅, :Γ₆, :JV]
        m.linearization[sym]
    else
        throw(KeyError("key $sym not found"))
    end
end

function Base.getindex(m::RALNonlinearSystem, sym::Symbol)
    if sym == :μ_sss
        isnothing(m.μ.cache) ? error("μ is out of place, so its stochastic steady state value is not cached.") : m.μ.cache.du
    elseif sym == :ξ_sss
        isnothing(m.ξ.cache) ? error("ξ is out of place, so its stochastic steady state value is not cached.") : m.ξ.cache.du
    elseif sym == :𝒱_sss
        m.𝒱.cache.du
    elseif sym == :Σ_sss
        if isnothing(m.Σ.cache)
            error("Λ is out of place, so its stochastic steady state value is not cached.")
        elseif isa(m.Σ.cache, DiffCache)
            m.Σ.cache.du
        else
            m.Σ.cache
        end
    elseif sym == :Λ_sss
        if isnothing(m.Λ.cache)
            error("Λ is out of place, so its stochastic steady state value is not cached.")
        elseif isa(m.Λ.cache, DiffCache)
            m.Λ.cache.du
        else
            m.Λ.cache
        end
    else
        throw(KeyError("key $sym not found"))
    end
end

function Base.getindex(m::RALLinearizedSystem, sym::Symbol)
    if sym == :Γ₁
        m.μz.cache.du
    elseif sym == :Γ₂
        m.μy.cache.du
    elseif sym == :Γ₃
        m.ξz.cache.du
    elseif sym == :Γ₄
        m.ξy.cache.du
    elseif sym == :Γ₅
        m.Γ₅
    elseif sym == :Γ₆
        m.Γ₆
    elseif sym == :JV
        m.J𝒱.cache.du
    else
        throw(KeyError("key $sym not found"))
    end
end

## Methods for using RiskAdjustedLinearization
@inline getvalues(m::RiskAdjustedLinearization) = (m.z, m.y, m.Ψ)
@inline getvecvalues(m::RiskAdjustedLinearization) = vcat(m.z, m.y, vec(m.Ψ))
@inline nonlinear_system(m::RiskAdjustedLinearization) = m.nonlinear
@inline linearized_system(m::RiskAdjustedLinearization) = m.linearization

@inline function update!(m::RiskAdjustedLinearization)
    update!(nonlinear_system(m), m.z, m.y, m.Ψ)
    update!(linearized_system(m), m.z, m.y, m.Ψ)
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
