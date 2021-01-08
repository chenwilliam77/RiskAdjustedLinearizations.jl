# Helper functions for exploiting sparsity in the Jacobians of μ, ξ, and 𝒱

"""
```
compute_sparsity_pattern(f::Function, x::AbstractVector{<: Number}, nrow::Int;
                         sparsity::Union{AbstractArray, Nothing} = nothing,
                         sparsity_detection::Bool = false)
```
calculates the sparsity pattern of the Jacobian of the functions μ, ξ, and 𝒱.

### Inputs
- `f`: is the function to be differentiated, e.g. `z -> 𝒱(z, Ψ, (1, 2))`
- `x`: the vector at which differentiation occurs
- `nrow`: specifies the number of rows of the Jacobian

### Keywords
- `sparsity`: sparsity pattern of the Jacobian
- `sparsity_detection`: if true, use SparsityDetection.jl to determine the sparsity pattern.
    If false, then the sparsity pattern is determined by using automatic differentiation
    to calculate a Jacobian and assuming any zeros will always be zero.
"""
function compute_sparsity_pattern(f::Function, x::AbstractVector{<: Number}, nrow::Int;
                                  sparsity::Union{AbstractArray, Nothing} = nothing,
                                  sparsity_detection::Bool = false)

    if isnothing(sparsity)
        sparsity = if sparsity_detection
            jacobian_sparsity(f, similar(x, nrow), x)
        else
            sparse(ForwardDiff.jacobian(f, x))
        end
    end

    if isempty(nonzeros(sparsity))
        # default to differentiating a dense matrix if all zeros
        return ones(size(sparsity)), 1:length(x)
    else
        return sparsity, matrix_colors(sparsity)
    end
end

"""
```
update_sparsity_pattern!(m::RiskAdjustedLinearization, function_name::Union{Symbol, Vector{Symbol}};
                         z::AbstractVector{<: Number} = m.z,
                         y::AbstractVector{<: Number} = m.y,
                         Ψ::AbstractVector{<: Number} = m.Ψ,
                         sparsity::AbstractDict{Symbol, AbstractMatrix} = Dict{Symbol, AbstractMatrix}(),
                         colorvec::AbstractDict{Symbol, <: AbstractVector{Int}} = Dict{Symbol, Vector{Int}}(),
                         sparsity_detection::Bool = false)
```
updates the Jacobians of μ, ξ, and/or 𝒱 in `m` with a new sparsity pattern. The Jacobians
to be updated are specified by `function_name`, e.g. `function_name = [:μ, :ξ, :𝒱]`.

If the keyword `sparsity` is empty, then the function attempts to determine the new sparsity pattern by computing
the Jacobian via automatic differentiation and assuming any zeros are always zero.
Keywords provide guesses for the coefficients ``(z, y, \\Psi)`` that are required
to calculate the Jacobians.

### Keywords
- `z`: state coefficients at steady state
- `y`: jump coefficients at steady state
- `Ψ`: coefficients for mapping from states to jumps
- `sparsity`: key-value pairs can be used to specify new sparsity patterns for the Jacobian functions
    `μz`, `μy`, `ξz`, `ξy`, and `J𝒱 `.
- `colorvec`: key-value pairs can be used to specify new matrix coloring vectors for the Jacobian functions
    `μz`, `μy`, `ξz`, `ξy`, and `J𝒱 `.
- `sparsity_detection`: use SparsityDetection.jl to determine the sparsity pattern.
"""
function update_sparsity_pattern!(m::RiskAdjustedLinearization, function_name::Symbol;
                                  z::AbstractVector{<: Number} = m.z,
                                  y::AbstractVector{<: Number} = m.y,
                                  Ψ::AbstractMatrix{<: Number} = m.Ψ,
                                  sparsity::AbstractDict{Symbol, <: AbstractMatrix} = Dict{Symbol, AbstractMatrix}(),
                                  colorvec::AbstractDict{Symbol, <: AbstractVector{Int}} = Dict{Symbol, Vector{Int}}(),
                                  sparsity_detection::Bool = false)
    return update_sparsity_pattern!(m, [function_name]; z = z, y = y, Ψ = Ψ,
                                    sparsity = sparsity, colorvec = colorvec,
                                    sparsity_detection = sparsity_detection)
end

function update_sparsity_pattern!(m::RiskAdjustedLinearization, function_names::Vector{Symbol};
                                  z::AbstractVector{<: Number} = m.z,
                                  y::AbstractVector{<: Number} = m.y,
                                  Ψ::AbstractMatrix{<: Number} = m.Ψ,
                                  sparsity::AbstractDict{Symbol, <: AbstractMatrix} = Dict{Symbol, AbstractMatrix}(),
                                  colorvec::AbstractDict{Symbol, <: AbstractVector{Int}} = Dict{Symbol, Vector{Int}}(),
                                  sparsity_detection::Bool = false)

    if :μ in function_names
        μz, μy, μz_jac_cache, μy_jac_cache =
            construct_μ_jacobian_function(m.nonlinear.μ, z, y;
                                          sparsity_z = haskey(sparsity, :μz) ? sparsity[:μz] : nothing,
                                          sparsity_y = haskey(sparsity, :μy) ? sparsity[:μy] : nothing,
                                          colorvec_z = haskey(sparsity, :μz) ? sparsity[:μz] : nothing,
                                          colorvec_y = haskey(sparsity, :μy) ? sparsity[:μy] : nothing,
                                          sparsity_detection = sparsity_detection)

        m.linearization.μz = μz
        m.linearization.μy = μy
        m.linearization.sparse_jac_caches[:μz] = μz_jac_cache
        m.linearization.sparse_jac_caches[:μy] = μy_jac_cache
    end

    if :ξ in function_names
        ξz, ξy, ξz_jac_cache, ξy_jac_cache =
            construct_ξ_jacobian_function(m.nonlinear.ξ, z, y;
                                          sparsity_z = haskey(sparsity, :ξz) ? sparsity[:ξz] : nothing,
                                          sparsity_y = haskey(sparsity, :ξy) ? sparsity[:ξy] : nothing,
                                          colorvec_z = haskey(sparsity, :ξz) ? sparsity[:ξz] : nothing,
                                          colorvec_y = haskey(sparsity, :ξy) ? sparsity[:ξy] : nothing,
                                          sparsity_detection = sparsity_detection)

        m.linearization.ξz = ξz
        m.linearization.ξy = ξy
        m.linearization.sparse_jac_caches[:ξz] = ξz_jac_cache
        m.linearization.sparse_jac_caches[:ξy] = ξy_jac_cache
    end

    if :𝒱 in function_names
        J𝒱, J𝒱_jac_cache = if isa(m.nonlinear.𝒱, RALF2)
            construct_𝒱_jacobian_function(m.nonlinear.𝒱, z, Ψ; sparsity = haskey(sparsity, :J𝒱) ? sparsity[:J𝒱] : nothing,
                                          colorvec = haskey(colorvec, :J𝒱) ? colorvec[:J𝒱] : nothing,
                                          sparsity_detection = sparsity_detection)
        else
            construct_𝒱_jacobian_function(m.nonlinear.𝒱, z, y, Ψ; sparsity = haskey(sparsity, :J𝒱) ? sparsity[:J𝒱] : nothing,
                                          colorvec = haskey(colorvec, :J𝒱) ? colorvec[:J𝒱] : nothing,
                                          sparsity_detection = sparsity_detection)
        end

        m.linearization.J𝒱 = J𝒱
        m.linearization.sparse_jac_caches[:J𝒱] = J𝒱_jac_cache
    end

    m
end

## Helper functions for constructing the Jacobian functions of μ, ξ, and 𝒱
function construct_μ_jacobian_function(μ::RALF2, z::AbstractVector{T}, y::AbstractVector{T};
                                       jacobian_type::DataType = Matrix{T},
                                       sparsity_z::Union{AbstractArray, Nothing} = nothing,
                                       sparsity_y::Union{AbstractArray, Nothing} = nothing,
                                       colorvec_z = nothing, colorvec_y = nothing,
                                       sparsity_detection::Bool = false) where {T <: Number}

    # Define (temporary) objective functions
    _f_μz = z -> μ(z, y, (1, 2))
    _f_μy = y -> μ(z, y, (2, 3))

    # Infer sparsity patterns and matrix coloring vector
    Nz = length(z)
    Ny = length(y)
    if isnothing(sparsity_z)
        sparsity_z, colorvec_z = compute_sparsity_pattern(_f_μz, z, Nz; sparsity_detection = sparsity_detection)
    elseif isnothing(colorvec_z)
        colorvec_z = matrix_colors(sparsity_z)
    end
    if isnothing(sparsity_y)
        sparsity_y, colorvec_y = compute_sparsity_pattern(_f_μy, y, Nz; sparsity_detection = sparsity_detection)
    elseif isnothing(colorvec_y)
        colorvec_y = matrix_colors(sparsity_y)
    end

    #=        # Create caches for the sparse Jacobian methods # This code is left here for when
    jac_cache_μz = ForwardColorJacCache(_f_μz, z, min(m.Nz, m.Ny); # Jacobians of μ and ξ are refactored
    sparsity = sparsity_μz, colorvec = colorvec_μz)
    jac_cache_μy = ForwardColorJacCache(_f_μy, y, min(m.Nz, m.Ny);
    sparsity = sparsity_μy, colorvec = colorvec_μy)=#

    # Create RALF2 objects. Note that we cannot pre-allocate the caches for
    # forwarddiff_color_jacobian! by using ForwardColorJacCache b/c the objective function
    # changes as z and y change. If Jacobians of μ and ξ are refactored to be done once,
    # then it'll be possible to cache.
    μ_dz = similar(z)
    μ_dy = similar(z)

    μz = RALF2((F, z, y) -> forwarddiff_color_jacobian!(F, x -> μ(x, y, (1, 2)), z, dx = μ_dz,
                                                        colorvec = colorvec_z, sparsity = sparsity_z),
               z, y, jacobian_type, (Nz, Nz))
    μy = RALF2((F, z, y) -> forwarddiff_color_jacobian!(F, x -> μ(z, x, (2, 3)), y, dx = μ_dy,
                                                        colorvec = colorvec_y, sparsity = sparsity_y),
               z, y, jacobian_type, (Nz, Ny))

    # Create mini-version of the Jacobian cache
    μz_jac_cache = (dx = μ_dz, sparsity = sparsity_z, colorvec = colorvec_z)
    μy_jac_cache = (dx = μ_dy, sparsity = sparsity_y, colorvec = colorvec_y)

    return μz, μy, μz_jac_cache, μy_jac_cache
end

function construct_ξ_jacobian_function(ξ::RALF2, z::AbstractVector{T}, y::AbstractVector{T};
                                       jacobian_type::DataType = Matrix{T},
                                       sparsity_z::Union{AbstractArray, Nothing} = nothing,
                                       sparsity_y::Union{AbstractArray, Nothing} = nothing,
                                       colorvec_z = nothing, colorvec_y = nothing,
                                       sparsity_detection::Bool = false) where {T <: Number}

    # Define (temporary) objective functions
    _f_ξz = z -> ξ(z, y, (1, 2))
    _f_ξy = y -> ξ(z, y, (2, 3))

    # Infer sparsity patterns and matrix coloring vector
    Nz = length(z)
    Ny = length(y)
    if isnothing(sparsity_z)
        sparsity_z, colorvec_z = compute_sparsity_pattern(_f_ξz, z, Ny; sparsity_detection = sparsity_detection)
    elseif isnothing(colorvec_z)
        colorvec_z = matrix_colors(sparsity_z)
    end
    if isnothing(sparsity_y)
        sparsity_y, colorvec_y = compute_sparsity_pattern(_f_ξy, y, Ny; sparsity_detection = sparsity_detection)
    elseif isnothing(colorvec_y)
        colorvec_y = matrix_colors(sparsity_y)
    end

    #=        # Create caches for the sparse Jacobian methods # This code is left here for when
    jac_cache_ξz = ForwardColorJacCache(_f_ξz, z, min(m.Nz, m.Ny);
    sparsity = sparsity_ξz, colorvec = colorvec_ξz)
    jac_cache_ξy = ForwardColorJacCache(_f_ξy, y, min(m.Nz, m.Ny);
    sparsity = sparsity_ξy, colorvec = colorvec_ξy)=#

    # Create RALF2 objects. Note that we cannot pre-allocate the caches for
    # forwarddiff_color_jacobian! by using ForwardColorJacCache b/c the objective function
    # changes as z and y change. If Jacobians of μ and ξ are refactored to be done once,
    # then it'll be possible to cache.
    ξ_dz = similar(y)
    ξ_dy = similar(y)

    ξz = RALF2((F, z, y) -> forwarddiff_color_jacobian!(F, x -> ξ(x, y, (1, 2)), z, dx = ξ_dz,
                                                        colorvec = colorvec_z, sparsity = sparsity_z),
               z, y, jacobian_type, (Ny, Nz))
    ξy = RALF2((F, z, y) -> forwarddiff_color_jacobian!(F, x -> ξ(z, x, (2, 3)), y, dx = ξ_dy,
                                                        colorvec = colorvec_y, sparsity = sparsity_y),
               z, y, jacobian_type, (Ny, Ny))

    # Create mini-version of the Jacobian cache
    ξz_jac_cache = (dx = ξ_dz, sparsity = sparsity_z, colorvec = colorvec_z)
    ξy_jac_cache = (dx = ξ_dy, sparsity = sparsity_y, colorvec = colorvec_y)

    return ξz, ξy, ξz_jac_cache, ξy_jac_cache
end

function construct_𝒱_jacobian_function(𝒱::RALF2, z::AbstractVector{T}, Ψ::AbstractMatrix{T};
                                       jacobian_type::DataType = Matrix{T},
                                       sparsity::Union{AbstractArray, Nothing} = nothing,
                                       colorvec = nothing, sparsity_detection::Bool = false) where {T <: Number}

    # Define (temporary) objective functions
    _f_𝒱z = z -> 𝒱(z, Ψ, (1, 2))

    # Infer sparsity patterns and matrix coloring vector
    if isnothing(sparsity)
        Nz = length(z)
        sparsity, colorvec = compute_sparsity_pattern(_f_𝒱z, z, size(Ψ, 1); sparsity_detection = sparsity_detection)
    elseif isnothing(colorvec)
        colorvec = matrix_colors(sparsity)
    end

    # Create RALF2 objects. Note that we cannot pre-allocate the caches for
    # forwarddiff_color_jacobian! by using ForwardColorJacCache b/c the objective function
    # changes as z and y change. If Jacobians of μ and ξ are refactored to be done once,
    # then it'll be possible to cache.
    𝒱_dz = similar(Ψ, size(Ψ, 1))
    J𝒱 = RALF2((F, z, Ψ) -> forwarddiff_color_jacobian!(F, x -> 𝒱(x, Ψ, (1, 2)), z, dx = 𝒱_dz,
                                                        colorvec = colorvec, sparsity = sparsity),
               z, Ψ, jacobian_type, size(Ψ))

    # Create mini-version of the Jacobian cache
    J𝒱_jac_cache = (dx = 𝒱_dz, sparsity = sparsity, colorvec = colorvec)

    return J𝒱, J𝒱_jac_cache
end

function construct_𝒱_jacobian_function(𝒱::RALF4, z::AbstractVector{T}, y::AbstractVector{T},
                                       Ψ::AbstractMatrix{T};
                                       matrix_type::DataType = Matrix{T},
                                       sparsity::Union{AbstractArray, Nothing} = nothing,
                                       colorvec = nothing, sparsity_detection::Bool = false) where {T <: Number}

    # Define (temporary) objective functions
    _f_𝒱z = zₜ -> 𝒱(zₜ, y, Ψ, zₜ, (4, 2))

    # Infer sparsity patterns and matrix coloring vector
    if isnothing(sparsity)
        sparsity, colorvec = compute_sparsity_pattern(_f_𝒱z, z, size(Ψ, 1); sparsity_detection = sparsity_detection)
    elseif isnothing(colorvec)
        colorvec = matrix_colors(sparsity)
    end

    # Create RALF2 objects. Note that we cannot pre-allocate the caches for
    # forwarddiff_color_jacobian! by using ForwardColorJacCache b/c the objective function
    # changes as z and y change. If Jacobians of μ and ξ are refactored to be done once,
    # then it'll be possible to cache.
    𝒱_dz = similar(Ψ, size(Ψ, 1))

    J𝒱 = RALF3((F, z, y, Ψ) -> forwarddiff_color_jacobian!(F, zₜ -> 𝒱(z, y, Ψ, zₜ, (4, 2)), z, dx = 𝒱_dz,
                                                           colorvec = colorvec, sparsity = sparsity),
               z, y, Ψ, jacobian_type, size(Ψ))

    # Create mini-version of the Jacobian cache
    J𝒱_jac_cache = (dx = 𝒱_dz, sparsity = sparsity, colorvec = colorvec)

    return J𝒱, J𝒱_jac_cache
end

# Helper functions for exploiting sparsity in calls to nlsolve

## Helper function for compute_sparsity_pattern
function infer_objective_function(m::RiskAdjustedLinearization, algorithm::Symbol; q::Float64 = .1)

    f = if algorithm == :deterministic
        (F, x) -> _deterministic_equations(F, x, m)
    elseif algorithm == :relaxation
        (F, x) -> _relaxation_equations(F, x, m, m.Ψ, m[:𝒱_sss])
    elseif algorithm == :homotopy
        if Λ_eltype(m.nonlinear) <: RALF1 && Σ_eltype(m.nonlinear) <: RALF1
            (F, x) -> _homotopy_equations1(F, x, m, q)
        else
            (F, x) -> _homotopy_equations2(F, x, m, q)
        end
    end

    return f
end

"""
```
compute_sparsity_pattern(m::RiskAdjustedLinearization, algorithm::Symbol; q::Float64 = .1,
                         sparsity::Union{AbstractArray, Nothing} = nothing,
                         sparsity_detection::Bool = false)
```
calculates the sparsity pattern and matrix coloring vector of the Jacobian
of the nonlinear system of equations for either the deterministic or
stochastic steady state, depending on which `algorithm` is called.

### Keywords
- `q`: step size for homotopy. Should satisfy `0 < q < 1` and is only required to ensure
    that the sparsity pattern is correctly determined when `algorithm = :homotopy`
    and thus the dependence of the entropy `𝒱` on the coefficients `(z, y, Ψ)` matters.
- `sparsity`: sparsity pattern of the Jacobian of the nonlinear system of equations
- `sparsity_detection`: if true, use SparsityDetection.jl to determine the sparsity pattern.
    If false, then the sparsity pattern is determined by using finite differences
    to calculate a Jacobian and assuming any zeros will always be zero.
"""
function compute_sparsity_pattern(m::RiskAdjustedLinearization, algorithm::Symbol; q::Float64 = .1,
                                  sparsity::Union{AbstractArray, Nothing} = nothing,
                                  sparsity_detection::Bool = false)
    @assert algorithm in [:deterministic, :relaxation, :homotopy] "The algorithm must be :deterministic, :relaxation, or :homotopy"
    @assert 1 > q > 0 "The step size q must satisfy 0 < q < 1."

    f = infer_objective_function(m, algorithm; q = q)

    input = algorithm == :homotopy ? vcat(m.z, m.y, vec(m.Ψ)) : vcat(m.z, m.y)
    if isnothing(sparsity)
        sparsity = if sparsity_detection
            jacobian_sparsity(f, similar(input), input)
        else
            jac = similar(input, length(input), length(input))
            FiniteDiff.finite_difference_jacobian!(jac, f, input)
            sparse(jac)
        end
    end
    colorvec = matrix_colors(sparsity)

    return sparsity, colorvec
end

"""
```
preallocate_jac_cache(m::RiskAdjustedLinearization, algorithm::Symbol; q::Float64 = .1,
                      sparsity::Union{AbstractArray, Nothing} = nothing,
                      sparsity_detection::Bool = false)
```
pre-allocates the cache for the Jacobian of the nonlinear system of equations
for either the deterministic or stochastic steady state, depending on which
`algorithm` is called.

### Keywords
- `q`: step size for homotopy. Should satisfy `0 < q < 1` and is only required to ensure
    that the sparsity pattern is correctly determined when `algorithm = :homotopy`
    and thus the dependence of the entropy `𝒱` on the coefficients `(z, y, Ψ)` matters.
- `sparsity`: the sparsity pattern of the Jacobian of the nonlinear system of equations
- `sparsity_detection`: if true, use SparsityDetection.jl to determine the sparsity pattern.
    If false, then the sparsity pattern is determined by using finite differences
    to calculate a Jacobian and assuming any zeros will always be zero.
"""
function preallocate_jac_cache(m::RiskAdjustedLinearization, algorithm::Symbol; q::Float64 = .1,
                               sparsity::Union{AbstractArray, Nothing} = nothing,
                               sparsity_detection::Bool = false)

    sparsity, colorvec = compute_sparsity_pattern(m, algorithm; q = q,
                                                  sparsity = sparsity, sparsity_detection = sparsity_detection)
    input = algorithm == :homotopy ? vcat(m.z, m.y, vec(m.Ψ)) : vcat(m.z, m.y)

    return FiniteDiff.JacobianCache(input, colorvec = colorvec, sparsity = sparsity)
end

function construct_sparse_jacobian_function(m::RiskAdjustedLinearization, f::Function,
                                            algorithm::Symbol, autodiff::Symbol;
                                            sparsity::Union{AbstractArray, Nothing} = nothing,
                                            colorvec = nothing, jac_cache = nothing,
                                            sparsity_detection::Bool = false)

    if isnothing(jac_cache)
        # Create Jacobian function that does not assume the existence of a cache

        if isnothing(sparsity) # No sparsity pattern provided, so need to make one
            sparsity, colorvec = compute_sparsity_pattern(m, algorithm; sparsity = sparsity,
                                                          sparsity_detection = sparsity_detection)
        elseif isnothing(colorvec) # Sparsity pattern, but no colorvec, so apply matrix_colors
            colorvec = matrix_colors(sparsity)
        end

        nlsolve_jacobian! = if autodiff == :forward
            (F, x) -> forwarddiff_color_jacobian!(F, f, x, # homotopy doesn't work with autodiff, so assuming
                                                  ForwardColorJacCache(f, x, min(m.Nz, m.Ny); # only using deterministic/relaxation,
                                                                       colorvec = colorvec, sparsity = sparsity)) # hence the chunk size
        else
            (F, x) -> FiniteDiff.finite_difference_jacobian!(F, f, x; colorvec = colorvec,
                                                             sparsity = sparsity)
        end

        return nlsolve_jacobian!, sparsity
    else
        # Create Jacobian function that assumes the existence of a cache

        nlsolve_jacobian! = if autodiff == :forward
            (F, x) -> forwarddiff_color_jacobian!(F, f, x, jac_cache)
        else
            (F, x) -> FiniteDiff.finite_difference_jacobian!(F, f, x, jac_cache)
        end

        return nlsolve_jacobian!, jac_cache.sparsity
    end
end
