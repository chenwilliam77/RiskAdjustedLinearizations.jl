# Subtypes used for the main RiskAdjustedLinearization type
mutable struct RALNonlinearSystem{L <: AbstractRALF, S <: AbstractRALF, V <: AbstractRALF}
    μ::RALF2
    Λ::L
    Σ::S
    ξ::RALF2
    𝒱::V
    ccgf::Function
end

Λ_eltype(m::RALNonlinearSystem{L, S}) where {L, S} = L
Σ_eltype(m::RALNonlinearSystem{L, S}) where {L, S} = S

function update!(m::RALNonlinearSystem{L, S, V}, z::C1, y::C1, Ψ::C2;
                 select::Vector{Symbol} = Symbol[:μ, :ξ, :𝒱]) where {L, S, V <: RALF2,
                                                                     C1 <: AbstractVector{<: Number},
                                                                     C2 <: AbstractMatrix{<: Number}}

    if :μ in select
        m.μ(z, y)
    end

    if :ξ in select
        m.ξ(z, y)
    end

    if :𝒱 in select
        m.𝒱(z, Ψ)
    end

    m
end

function update!(m::RALNonlinearSystem{L, S, V}, z::C1, y::C1, Ψ::C2;
                 select::Vector{Symbol} = Symbol[:μ, :ξ, :𝒱]) where {L, S, V <: RALF4,
                                                                     C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}}

    if :μ in select
        m.μ(z, y)
    end

    if :ξ in select
        m.ξ(z, y)
    end

    if :𝒱 in select
        m.𝒱(z, y, Ψ, z)
    end

    m
end

mutable struct RALLinearizedSystem{JC5 <: AbstractMatrix{<: Number},
                                   JC6 <: AbstractMatrix{<: Number}, SJC <: AbstractDict{Symbol, NamedTuple}}
    μz::RALF2
    μy::RALF2
    ξz::RALF2
    ξy::RALF2
    J𝒱::Union{RALF2, RALF3}
    Γ₅::JC5
    Γ₆::JC6
    sparse_jac_caches::SJC
end

function RALLinearizedSystem(μz::RALF2, μy::RALF2, ξz::RALF2, ξy::RALF2, J𝒱::AbstractRALF,
                             Γ₅::AbstractMatrix{<: Number}, Γ₆::AbstractMatrix{<: Number})
    RALLinearizedSystem(μz, μy, ξz, ξy, J𝒱, Γ₅, Γ₆, Dict{Symbol, NamedTuple}())
end

function update!(m::RALLinearizedSystem{JC5, JC6}, z::C1, y::C1, Ψ::C2;
                 select::Vector{Symbol} =
                 Symbol[:Γ₁, :Γ₂, :Γ₃, :Γ₄, :JV]) where {#JV <: RALF2,
                                                         JC5, JC6,
                                                         C1 <: AbstractVector{<: Number}, C2 <: AbstractMatrix{<: Number}}

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

### Keywords for First Method
- `sss_vector_cache_init::Function = dims -> Vector{T}(undef, dims)`: initializer for the cache of steady state vectors.
- `Λ_Σ_cache_init::Function = dims -> Matrix{T}(undef, dims)`:  initializer for the cache of `Λ` and `Σ`
- `jacobian_cache_init::Function = dims -> Matrix{T}(undef, dims)`: initializer for the cache of the Jacobians of `μ`, `ξ`, and `𝒱 `.
- `jump_dependent_shock_matrices::Bool = false`: if true, `Λ` and `Σ` are treated as `Λ(z, y)` and `Σ(z, y)`
    to allow dependence on jumps.
- `sparse_jacobian::Vector{Symbol} = Symbol[]`: pass the symbols `:μ`, `:ξ`, and/or `:𝒱 ` to declare that
    the Jacobians of these functions are sparse and should be differentiated using sparse methods from SparseDiffTools.jl
- `sparsity::AbstractDict = Dict{Symbol, Mtarix}()`: a dictionary for declaring the
    sparsity patterns of the Jacobians of `μ`, `ξ`, and `𝒱 `. The relevant keys are `:μz`, `:μy`, `:ξz`, `:ξy`, and `:J𝒱 `.
- `colorvec::AbstractDict = Dict{Symbol, Vector{Int}}()`: a dictionary for declaring the
    the matrix coloring vector. The relevant keys are `:μz`, `:μy`, `:ξz`, `:ξy`, and `:J𝒱 `.
- `sparsity_detection::Bool = false`: if true, use SparseDiffTools to determine the sparsity pattern.
    When false (default), the sparsity pattern is estimated by differentiating the Jacobian once
    with `ForwardDiff` and assuming any zeros in the calculated Jacobian are supposed to be zeros.

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
                                   Nε::Int; sss_vector_cache_init::Function = dims -> Vector{T}(undef, dims),
                                   Λ_Σ_cache_init::Function = dims -> Matrix{T}(undef, dims), jump_dependent_shock_matrices::Bool = false,
                                   jacobian_cache_init::Function = dims -> Matrix{T}(undef, dims),
                                   sparse_jacobian::Vector{Symbol} = Symbol[],
                                   sparsity::AbstractDict{Symbol, AbstractMatrix} = Dict{Symbol, AbstractMatrix}(),
                                   colorvec::AbstractDict = Dict{Symbol, Vector{Int}}(),
                                   sparsity_detection::Bool = false) where {T <: Number, M <: Function, L, S,
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
    _μ = RALF2(μ, z, y, sss_vector_cache_init(Nz), (max(min(Nzchunk, Nychunk), 2), Nzchunk, Nychunk))
    _ξ = RALF2(ξ, z, y, sss_vector_cache_init(Ny), (max(min(Nzchunk, Nychunk), 2), Nzchunk, Nychunk))

    # Apply dispatch on Λ and Σ to figure what they should be
    return RiskAdjustedLinearization(_μ, Λ, Σ, _ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_cache_init = sss_vector_cache_init,
                                     Λ_Σ_cache_init = Λ_Σ_cache_init,
                                     jump_dependent_shock_matrices = jump_dependent_shock_matrices,
                                     jacobian_cache_init = jacobian_cache_init,
                                     sparse_jacobian = sparse_jacobian, sparsity = sparsity,
                                     colorvec = colorvec, sparsity_detection = sparsity_detection)
end

# Constructor that uses ForwardDiff to calculate Jacobian functions.
# Users will not typically use this constructor, however, because it requires
# various functions of the RALNonlinearSystem and RALLinearizedSystem to already
# be wrapped with either an RALF1 or RALF2 type.
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_cache_init::Function = dims -> Vector{T}(undef, dims),
                                   jacobian_cache_init::Function = dims -> Matrix{T}(undef, dims),
                                   sparse_jacobian::Vector{Symbol} = Symbol[],
                                   sparsity::AbstractDict{Symbol, AbstractMatrix} = Dict{Symbol, AbstractMatrix}(),
                                   colorvec::AbstractDict = Dict{Symbol, Vector{Int}}(),
                                   sparsity_detection::Bool = false) where {T <: Number, M <: RALF2, L <: RALF1, S <: RALF1,
                                                                            X <: RALF2,
                                                                            JC5 <: AbstractMatrix{<: Number},
                                                                            JC6 <: AbstractMatrix{<: Number},
                                                                            CF <: Function}

    jac_cache = Dict{Symbol, NamedTuple}()

    # Use RALF2 wrapper to create Jacobian functions with caching for μ, ξ.
    # Use the tuple to select the correct Dual cache b/c μ is in place
    if :μ in sparse_jacobian
        μz, μy, jac_cache[:μz], jac_cache[:μy] =
            construct_μ_jacobian_function(μ, z, y;
                                          sparsity_z = haskey(sparsity, :μz) ? sparsity[:μz] : nothing,
                                          sparsity_y = haskey(sparsity, :μy) ? sparsity[:μy] : nothing,
                                          colorvec_z = haskey(sparsity, :μz) ? sparsity[:μz] : nothing,
                                          colorvec_y = haskey(sparsity, :μy) ? sparsity[:μy] : nothing,
                                          sparsity_detection = sparsity_detection)
    else
        μz = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(x, y, (1, 2)), z), z, y,
                   jacobian_cache_init((Nz, Nz)))
        μy = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(z, x, (2, 3)), y), z, y,
                   jacobian_cache_init((Nz, Ny)))
    end

    if :ξ in sparse_jacobian
        ξz, ξy, jac_cache[:ξz], jac_cache[:ξy] =
            construct_ξ_jacobian_function(ξ, z, y;
                                          sparsity_z = haskey(sparsity, :ξz) ? sparsity[:ξz] : nothing,
                                          sparsity_y = haskey(sparsity, :ξy) ? sparsity[:ξy] : nothing,
                                          colorvec_z = haskey(sparsity, :ξz) ? sparsity[:ξz] : nothing,
                                          colorvec_y = haskey(sparsity, :ξy) ? sparsity[:ξy] : nothing,
                                          sparsity_detection = sparsity_detection)
    else
        ξz = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(x, y, (1, 2)), z), z, y,
                   jacobian_cache_init((Ny, Nz)))
        ξy = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(z, x, (2, 3)), y), z, y,
                   jacobian_cache_init((Ny, Ny)))
    end

    # Create RALF2 wrappers for 𝒱 and its Jacobian J𝒱
    if applicable(ccgf, Γ₅, z) # Check if ccgf is in place or not
        _𝒱 = function _𝒱_oop(F, z, Ψ)
            Λ0 = Λ(z)
            Σ0 = Σ(z)
            if size(Λ0) != (Nz, Ny)
                Λ0 = reshape(Λ0, Nz, Ny)
            end
            if size(Σ0) != (Nz, Nε)
                Σ0 = reshape(Σ0, Nz, Nε)
            end
            F .= ccgf((Γ₅ + Γ₆ * Ψ) * ((I - (Λ0 * Ψ)) \ Σ0), z)
        end
    else # in place
        _𝒱 = function _𝒱_ip(F, z, Ψ)
            Λ0 = Λ(z)
            Σ0 = Σ(z)
            if size(Λ0) != (Nz, Ny)
                Λ0 = reshape(Λ0, Nz, Ny)
            end
            if size(Σ0) != (Nz, Nε)
                Σ0 = reshape(Σ0, Nz, Nε)
            end
            ccgf(F, (Γ₅ + Γ₆ * Ψ) * ((I - (Λ0 * Ψ)) \ Σ0), z)
        end
    end
    Nzchunk = ForwardDiff.pickchunksize(Nz)
    Nychunk = ForwardDiff.pickchunksize(Ny)
    𝒱 = RALF2((F, z, Ψ) -> _𝒱(F, z, Ψ), z, Ψ, sss_vector_cache_init(Ny), (max(min(Nzchunk, Nychunk), 2), Nzchunk))

    if :𝒱 in sparse_jacobian
        J𝒱, jac_cache[:J𝒱] = construct_𝒱_jacobian_function(𝒱, ccgf, Λ, Σ, Γ₅, Γ₆, z, Ψ;
                                                           sparsity = haskey(sparsity, :J𝒱) ? sparsity[:J𝒱] : nothing,
                                                           colorvec = haskey(colorvec, :J𝒱) ? colorvec[:J𝒱] : nothing,
                                                           sparsity_detection = sparsity_detection)
    else
        _J𝒱(F, z, Ψ) = ForwardDiff.jacobian!(F, x -> 𝒱(x, Ψ, (1, 2)), z)
        J𝒱           = RALF2((F, z, Ψ) -> _J𝒱(F, z, Ψ), z, Ψ, jacobian_cache_init((Ny, Nz)))
    end

    # Form underlying RAL blocks
    nonlinear_system  = RALNonlinearSystem(μ, Λ, Σ, ξ, 𝒱, ccgf)
    linearized_system = RALLinearizedSystem(μz, μy, ξz, ξy, J𝒱, Γ₅, Γ₆, jac_cache)

    return RiskAdjustedLinearization(nonlinear_system, linearized_system, z, y, Ψ, Nz, Ny, Nε)
end

# Handles case where Λ and Σ are RALF2
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_cache_init::Function = dims -> Vector{T}(undef, dims),
                                   jacobian_cache_init::Function = dims -> Matrix{T}(undef, dims),
                                   sparse_jacobian::Vector{Symbol} = Symbol[],
                                   sparsity::AbstractDict{Symbol, AbstractMatrix} = Dict{Symbol, AbstractMatrix}(),
                                   colorvec::AbstractDict = Dict{Symbol, Vector{Int}}(),
                                   sparsity_detection::Bool = false) where {T <: Number, M <: RALF2, L <: RALF2, S <: RALF2,
                                                                            X <: RALF2,
                                                                            JC5 <: AbstractMatrix{<: Number},
                                                                            JC6 <: AbstractMatrix{<: Number},
                                                                            CF <: Function}

    jac_cache = Dict{Symbol, NamedTuple}()

    # Use RALF2 wrapper to create Jacobian functions with caching for μ, ξ.
    # Use the tuple to select the correct Dual cache b/c μ is in place
    if :μ in sparse_jacobian
        μz, μy, jac_cache[:μz], jac_cache[:μy] =
            construct_μ_jacobian_function(μ, z, y;
                                          sparsity_z = haskey(sparsity, :μz) ? sparsity[:μz] : nothing,
                                          sparsity_y = haskey(sparsity, :μy) ? sparsity[:μy] : nothing,
                                          colorvec_z = haskey(sparsity, :μz) ? sparsity[:μz] : nothing,
                                          colorvec_y = haskey(sparsity, :μy) ? sparsity[:μy] : nothing,
                                          sparsity_detection = sparsity_detection)
    else
        μz = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(x, y, (1, 2)), z), z, y,
                   jacobian_cache_init((Nz, Nz)))
        μy = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> μ(z, x, (2, 3)), y), z, y,
                   jacobian_cache_init((Nz, Ny)))
    end

    if :ξ in sparse_jacobian
        ξz, ξy, jac_cache[:ξz], jac_cache[:ξy] =
            construct_ξ_jacobian_function(μ, z, y;
                                          sparsity_z = haskey(sparsity, :ξz) ? sparsity[:ξz] : nothing,
                                          sparsity_y = haskey(sparsity, :ξy) ? sparsity[:ξy] : nothing,
                                          colorvec_z = haskey(sparsity, :ξz) ? sparsity[:ξz] : nothing,
                                          colorvec_y = haskey(sparsity, :ξy) ? sparsity[:ξy] : nothing,
                                          sparsity_detection = sparsity_detection)
    else
        ξz = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(x, y, (1, 2)), z), z, y,
                   jacobian_cache_init((Ny, Nz)))
        ξy = RALF2((F, z, y) -> ForwardDiff.jacobian!(F, x -> ξ(z, x, (2, 3)), y), z, y,
                   jacobian_cache_init((Ny, Ny)))
    end

    # Create RALF2 wrappers for 𝒱 and its Jacobian J𝒱
    if applicable(ccgf, Γ₅, z) # Check if ccgf is in place or not
        _𝒱 = function _𝒱_oop(F, z, y, Ψ, zₜ)
            yₜ = y + Ψ * (zₜ - z)
            Λ0 = Λ(zₜ, yₜ)
            Σ0 = Σ(zₜ, yₜ)
            if size(Λ0) != (Nz, Ny)
                Λ0 = reshape(Λ0, Nz, Ny)
            end
            if size(Σ0) != (Nz, Nε)
                Σ0 = reshape(Σ0, Nz, Nε)
            end
            F .= ccgf((Γ₅ + Γ₆ * Ψ) * ((I - (Λ0 * Ψ)) \ Σ0), zₜ)
        end
    else # in place
        _𝒱 = function _𝒱_ip(F, z, y, Ψ, zₜ)
            yₜ = y + Ψ * (zₜ - z)
            Λ0 = Λ(zₜ, yₜ)
            Σ0 = Σ(zₜ, yₜ)
            if size(Λ0) != (Nz, Ny)
                Λ0 = reshape(Λ0, Nz, Ny)
            end
            if size(Σ0) != (Nz, Nε)
                Σ0 = reshape(Σ0, Nz, Nε)
            end
            ccgf(F, (Γ₅ + Γ₆ * Ψ) * ((I - (Λ0 * Ψ)) \ Σ0), zₜ)
        end
    end
    Nzchunk = ForwardDiff.pickchunksize(Nz)
    Nychunk = ForwardDiff.pickchunksize(Ny)
    𝒱       = RALF4((F, z, y, Ψ, zₜ) -> _𝒱(F, z, y, Ψ, zₜ), z, y, Ψ, z, sss_vector_cache_init(Ny),
                    (max(min(Nzchunk, Nychunk), 2), Nzchunk))

    if :𝒱 in sparse_jacobian
        J𝒱, jac_cache[:J𝒱] = construct_𝒱_jacobian_function(𝒱, ccgf, Λ, Σ, Γ₅, Γ₆, z, y, Ψ;
                                                           sparsity = haskey(sparsity, :J𝒱) ? sparsity[:J𝒱] : nothing,
                                                           colorvec = haskey(colorvec, :J𝒱) ? colorvec[:J𝒱] : nothing,
                                                           sparsity_detection = sparsity_detection)
    else
        _J𝒱(F, z, y, Ψ) = ForwardDiff.jacobian!(F, zₜ -> 𝒱(z, y, Ψ, zₜ, (4, 2)), z) # use zₜ argument to infer the cache
        J𝒱              = RALF3((F, z, y, Ψ) -> _J𝒱(F, z, y, Ψ), z, y, Ψ, jacobian_cache_init((Ny, Nz)))
    end

    # Form underlying RAL blocks
    nonlinear_system  = RALNonlinearSystem(μ, Λ, Σ, ξ, 𝒱, ccgf)
    linearized_system = RALLinearizedSystem(μz, μy, ξz, ξy, J𝒱, Γ₅, Γ₆, jac_cache)

    return RiskAdjustedLinearization(nonlinear_system, linearized_system, z, y, Ψ, Nz, Ny, Nε)
end

# The following four constructors cover different common cases for the Λ and Σ functions.
function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_cache_init::Function = dims -> Vector{T}(undef, dims),
                                   Λ_Σ_cache_init::Function = dims -> Matrix{T}(undef, dims), jump_dependent_shock_matrices::Bool = false,
                                   jacobian_cache_init::Function = dims -> Matrix{T}(undef, dims),
                                   sparse_jacobian::Vector{Symbol} = Symbol[],
                                   sparsity::AbstractDict = Dict{Symbol, Matrix}(),
                                   colorvec::AbstractDict = Dict{Symbol, Vector{Int}}(),
                                   sparsity_detection::Bool = false) where {T <: Number, M <: RALF2, L <: Function, S <: Function,
                                                                            X <: RALF2,
                                                                            JC5 <: AbstractMatrix{<: Number},
                                                                            JC6 <: AbstractMatrix{<: Number},
                                                                            CF <: Function}

    # Create wrappers enabling caching for Λ and Σ
    Nzchunk = ForwardDiff.pickchunksize(Nz)
    Nychunk = ForwardDiff.pickchunksize(Ny)
    if jump_dependent_shock_matrices
        _Λ = RALF2(Λ, z, y, Λ_Σ_cache_init((Nz, Ny)), (max(min(Nzchunk, Nychunk), 2), Nzchunk))
        _Σ = RALF2(Σ, z, y, Λ_Σ_cache_init((Nz, Nε)), (max(min(Nzchunk, Nychunk), 2), Nzchunk))
    else
        _Λ = RALF1(Λ, z, Λ_Σ_cache_init((Nz, Ny)))
        _Σ = RALF1(Σ, z, Λ_Σ_cache_init((Nz, Nε)))
    end

    return RiskAdjustedLinearization(μ, _Λ, _Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_cache_init = sss_vector_cache_init,
                                     jacobian_cache_init = jacobian_cache_init, sparse_jacobian = sparse_jacobian,
                                     sparsity = sparsity, sparsity_detection = sparsity_detection, colorvec = colorvec)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_cache_init::Function = dims -> Vector{T}(undef, dims),
                                   Λ_Σ_cache_init::Function = dims -> Matrix{T}(undef, dims), jump_dependent_shock_matrices::Bool = false,
                                   jacobian_cache_init::Function = dims -> Matrix{T}(undef, dims),
                                   sparse_jacobian::Vector{Symbol} = Symbol[],
                                   sparsity::AbstractDict = Dict{Symbol, Matrix}(),
                                   colorvec::AbstractDict = Dict{Symbol, Vector{Int}}(),
                                   sparsity_detection::Bool = false) where {T <: Number, M <: RALF2,
                                                                            L <: AbstractMatrix{<: Number}, S <: Function,
                                                                            X <: RALF2,
                                                                            JC5 <: AbstractMatrix{<: Number},
                                                                            JC6 <: AbstractMatrix{<: Number},
                                                                            CF <: Function}

    # Create wrappers enabling caching for Λ and Σ
    if jump_dependent_shock_matrices
        _Λ = RALF2(Λ)
        _Σ = RALF2(Σ, z, y, Λ_Σ_cache_init((Nz, Nε)), (max(min(Nzchunk, Nychunk), 2), Nzchunk))
    else
        _Λ = RALF1(Λ)
        _Σ = RALF1(Σ, z, Λ_Σ_cache_init((Nz, Nε)))
    end

    return RiskAdjustedLinearization(μ, _Λ, _Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_cache_init = sss_vector_cache_init,
                                     jacobian_cache_init = jacobian_cache_init, sparse_jacobian = sparse_jacobian,
                                     sparsity = sparsity, sparsity_detection = sparsity_detection, colorvec = colorvec)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_cache_init::Function = dims -> Vector{T}(undef, dims),
                                   Λ_Σ_cache_init::Function = dims -> Matrix{T}(undef, dims), jump_dependent_shock_matrices::Bool = false,
                                   jacobian_cache_init::Function = dims -> Matrix{T}(undef, dims),
                                   sparse_jacobian::Vector{Symbol} = Symbol[],
                                   sparsity::AbstractDict = Dict{Symbol, Matrix}(),
                                   colorvec::AbstractDict = Dict{Symbol, Vector{Int}}(),
                                   sparsity_detection::Bool = false) where {T <: Number, M <: RALF2, L <: Function, S <: AbstractMatrix{<: Number},
                                                                            X <: RALF2,
                                                                            JC5 <: AbstractMatrix{<: Number},
                                                                            JC6 <: AbstractMatrix{<: Number},
                                                                            CF <: Function}

    # Create wrappers enabling caching for Λ and Σ
    Nzchunk = ForwardDiff.pickchunksize(Nz)
    Nychunk = ForwardDiff.pickchunksize(Ny)
    if jump_dependent_shock_matrices
        _Λ = RALF2(Λ, z, y, Λ_Σ_cache_init((Nz, Ny)), (max(min(Nzchunk, Nychunk), 2), Nzchunk))
        _Σ = RALF2(Σ)
    else
        _Λ = RALF1(Λ, z, Λ_Σ_cache_init((Nz, Ny)))
        _Σ = RALF1(Σ)
    end

    return RiskAdjustedLinearization(μ, _Λ, _Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_cache_init = sss_vector_cache_init,
                                     jacobian_cache_init = jacobian_cache_init, sparse_jacobian = sparse_jacobian,
                                     sparsity = sparsity, sparsity_detection = sparsity_detection, colorvec = colorvec)
end

function RiskAdjustedLinearization(μ::M, Λ::L, Σ::S, ξ::X, Γ₅::JC5, Γ₆::JC6, ccgf::CF,
                                   z::AbstractVector{T}, y::AbstractVector{T}, Ψ::AbstractMatrix{T},
                                   Nz::Int, Ny::Int, Nε::Int; sss_vector_cache_init::Function = dims -> Vector{T}(undef, dims),
                                   Λ_Σ_cache_init::Function = dims -> Matrix{T}(undef, dims),
                                   jacobian_cache_init::Function = dims -> Matrix{T}(undef, dims),
                                   sparse_jacobian::Vector{Symbol} = Symbol[],
                                   sparsity::AbstractDict = Dict{Symbol, Matrix}(),
                                   sparsity_detection::Bool = false) where {T <: Number, M <: RALF2,
                                                                            L <: AbstractMatrix{<: Number}, S <: AbstractMatrix{<: Number},
                                                                            X <: RALF2,
                                                                            JC5 <: AbstractMatrix{<: Number},
                                                                            JC6 <: AbstractMatrix{<: Number},
                                                                            CF <: Function}

    # Create wrappers enabling caching for Λ and Σ
    _Λ = RALF1(Λ)
    _Σ = RALF1(Σ)

    return RiskAdjustedLinearization(μ, _Λ, _Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nz, Ny, Nε, sss_vector_cache_init = sss_vector_cache_init,
                                     jacobian_cache_init = jacobian_cache_init, sparse_jacobian = sparse_jacobian,
                                     sparsity = sparsity, sparsity_detection = sparsity_detection, colorvec = colorvec)
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
