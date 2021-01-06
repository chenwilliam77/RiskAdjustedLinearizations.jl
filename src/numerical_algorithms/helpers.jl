# Helper functions for exploiting sparsity in calls to nlsolve

## Helper function for compute_sparsity_pattern
function infer_objective_function(m::RiskAdjustedLinearization, algorithm::Symbol; q::Float64 = .1)

    f = if algorithm == :deterministic
        (F, x) -> _my_deterministic_equations(F, x, m)
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
calculates the sparsity pattern of the Jacobian of the nonlinear system of equations
for either the deterministic or stochastic steady state, depending on which
`algorithm` is called.

### Keywords
- `q`: step size for homotopy. Should satisfy `0 < q < 1` and is only required to ensure
    that the sparsity pattern is correctly determined when `algorithm = :homotopy`
    and thus the dependence of the entropy `𝒱` on the coefficients `(z, y, Ψ)` matters.
- `sparsity`: the sparsity patteren of the Jacobian of the nonlinear system of equations
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
- `sparsity`: the sparsity patteren of the Jacobian of the nonlinear system of equations
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
