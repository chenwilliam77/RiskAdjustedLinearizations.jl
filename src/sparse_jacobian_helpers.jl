# Helper functions for exploiting sparsity in the Jacobians of μ, ξ, and 𝒱

"""
```
compute_sparsity_pattern(f::Function, x::AbstractVector{<: Number};
                         sparsity::Union{AbstractArray, Nothing} = nothing,
                         sparsity_detection::Bool = false)
```
calculates the sparsity pattern of the Jacobian of the functions μ, ξ, and 𝒱.

### Keywords
- `sparsity`: sparsity pattern of the Jacobian
- `sparsity_detection`: if true, use SparsityDetection.jl to determine the sparsity pattern.
    If false, then the sparsity pattern is determined by using automatic differentiation
    to calculate a Jacobian and assuming any zeros will always be zero.
"""
function compute_sparsity_pattern(f::Function, x::AbstractVector{<: Number};
                                  sparsity::Union{AbstractArray, Nothing} = nothing,
                                  sparsity_detection::Bool = false)

    if isnothing(sparsity)
        sparsity = if sparsity_detection
            jacobian_sparsity(f, similar(input), input)
        else
            sparse(ForwardDiff.jacobian(f, input))
        end
    end
    colorvec = matrix_colors(sparsity)

    return sparsity, colorvec
end

function construct_μ_jacobian_function(μ::RALF2, z::AbstractVector{<: Number}, y::AbstractVector{<: Number};
                                       sparsity_z::Union{AbstractArray, Nothing} = nothing,
                                       sparsity_y::Union{AbstractArray, Nothing} = nothing,
                                       colorvec_z = nothing, colorvec_y = nothing,
                                       sparsity_detection::Bool = false)

    # Define (temporary) objective functions
    _f_μz = (F, z) -> μ(z, y, (1, 2))
    _f_μy = (F, y) -> μ(z, y, (2, 3))

    # Infer sparsity patterns and matrix coloring vector
    sparsity_z, colorvec_z = compute_sparsity_pattern(_f_μz, z; sparsity_detection = sparsity_detection)
    sparsity_y, colorvec_y = compute_sparsity_pattern(_f_μy, y; sparsity_detection = sparsity_detection)

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

function construct_ξ_jacobian_function(ξ::RALF2, z::AbstractVector{<: Number}, y::AbstractVector{<: Number};
                                       sparsity_z::Union{AbstractArray, Nothing} = nothing,
                                       sparsity_y::Union{AbstractArray, Nothing} = nothing,
                                       colorvec_z = nothing, colorvec_y = nothing,
                                       sparsity_detection::Bool = false)

    # Define (temporary) objective functions
    _f_ξz = (F, z) -> ξ(z, y, (1, 2))
    _f_ξy = (F, y) -> ξ(z, y, (2, 3))

    # Infer sparsity patterns and matrix coloring vector
    sparsity_z, colorvec_z = compute_sparsity_pattern(_f_ξz, z; sparsity_detection = sparsity_detection)
    sparsity_y, colorvec_y = compute_sparsity_pattern(_f_ξy, y; sparsity_detection = sparsity_detection)

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

function construct_𝒱_jacobian_function(𝒱::RALF2, z::AbstractVector{<: Number}, Ψ::AbstractMatrix{<: Number};
                                       sparsity::Union{AbstractArray, Nothing} = nothing,
                                       colorvec = nothing, sparsity_detection::Bool = false)

    # Define (temporary) objective functions
    _f_𝒱z = (F, z) -> 𝒱(z, Ψ, (1, 2))

    # Infer sparsity patterns and matrix coloring vector
    if isnothing(sparsity)
        sparsity, colorvec = compute_sparsity_pattern(_f_𝒱z, z; sparsity_detection = sparsity_detection)
    elseif isnothing(colorvec)
        colorvec = matrix_colors(sparsity)
    end

    # Create RALF2 objects. Note that we cannot pre-allocate the caches for
    # forwarddiff_color_jacobian! by using ForwardColorJacCache b/c the objective function
    # changes as z and y change. If Jacobians of μ and ξ are refactored to be done once,
    # then it'll be possible to cache.
    𝒱_dz = similar(Ψ, size(Ψ, 1))

    J𝒱 = RALF2((F, z, y) -> forwarddiff_color_jacobian!(F, x -> 𝒱(x, Ψ, (1, 2)), z, dx = 𝒱_dz,
                                                        colorvec = colorvec, sparsity = sparsity),
               z, Ψ, jacobian_type, (Ny, Nz))

    # Create mini-version of the Jacobian cache
    J𝒱_jac_cache = (dx = 𝒱_dz, sparsity = sparsity, colorvec = colorvec)

    return J𝒱, J𝒱_jac_cache
end

function construct_𝒱_jacobian_function(𝒱::RALF4, z::AbstractVector{<: Number}, y::AbstractVector{<: Number},
                                       Ψ::AbstractMatrix{<: Number};
                                       sparsity::Union{AbstractArray, Nothing} = nothing,
                                       colorvec = nothing, sparsity_detection::Bool = false)

    # Define (temporary) objective functions
    _f_𝒱z = (F, zₜ) -> 𝒱(zₜ, y, Ψ, zₜ, (4, 2))

    # Infer sparsity patterns and matrix coloring vector
    if isnothing(sparsity)
        sparsity, colorvec = compute_sparsity_pattern(_f_𝒱z, z; sparsity_detection = sparsity_detection)
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
               z, y, Ψ, jacobian_type, (Ny, Nz))

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
