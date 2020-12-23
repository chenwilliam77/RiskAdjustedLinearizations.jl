function standard_normal_gausshermite(n::Int)
    ϵᵢ, wᵢ = gausshermite(n) # approximates exp(-x²)
    ϵᵢ   .*= sqrt(2.)        # Normalize ϵᵢ and wᵢ nodes to approximate standard normal
    wᵢ   ./= sqrt(π)

    return ϵᵢ, wᵢ
end

"""
```
gausshermite_expectation(f, μ, σ, n = 10)
gausshermite_expectation(f, μ, Σ, n = 10)
gausshermite_expectation(f, μ, Σ, ns)
```

calculates the expectation of a function of a Gaussian random variable/vector.
The first method evalulates ``\\mathbb{E}[f(X)]`` where ``X \\sim N(\\mu, \\sigma)``,
while the other two methods evaluate ``\\mathbb{E}[f(X)]`` where
``X \\sim \\mathcal{N}(\\mu, \\Sigma)`` and ``\\Sigma`` is diagonal.
The latter two methods differ in that the first assumes the same number of
quadrature points in every dimension while the second does not.

### Inputs
- `f::Function`: some function of a random variable. If `f(x) = x`, then
    `gausshermite_expectation(f, μ, σ)` calculates the mean of ``N(\\mu, \\sigma)``
    using 10-point Gauss-Hermite quadrature.
- `μ::Number` or `μ::AbstractVector`: mean of the Gaussian random variable/vector.
- `σ::Number`: standard deviation of the Gaussian random variable.
- `Σ::AbstractVector`: diagonal of the variance-covariance matrix of
     the Gaussian random vector.
- `n::Int`: number of quadrature points to use
- `ns::AbstractVector{Int}` or `ns::NTuple{N, Int} where N`: number of quadrature points to use
    in each dimension of the Gaussian random vector.
"""
function gausshermite_expectation(f::Function, μ::Number, σ::Number, n::Int = 10)
    ϵᵢ, wᵢ = gausshermite(n)
    ϵᵢ   .*= sqrt(2.)  # Normalize ϵᵢ and wᵢ nodes to approximate standard normal
    # wᵢ   ./= sqrt(π) # This step done later to reduce number of computations

    if μ ≈ 0.
        return sum([wᵢ[i] * f(ϵᵢ[i] * σ) for i in 1:n]) / sqrt(π)
    else
        return sum([wᵢ[i] * f(ϵᵢ[i] * σ + μ) for i in 1:n]) / sqrt(π)
    end

end

function gausshermite_expectation(f::Function, μ::AbstractVector{S},
                                  Σ::AbstractVector{<: Number}, n::Int = 10) where {S <: Number}

    d = length(μ)
    @assert length(Σ) == d "The length of μ and Σ must be the same."
    ϵ, w = gausshermite(n)
    ϵ   .*= sqrt(2.)  # Normalize ϵ and w nodes to approximate standard normal
    # w   ./= sqrt(π) # This step done later to reduce number of computations

    # Evaluate over the tensor grid
    feval = Array{S}(undef, (n for i in 1:d)...)
    allCI = CartesianIndices(feval)
    if all(μ .≈ 0.)
        @simd for CI in allCI
            feval[CI] = f([ϵ[i] for i in Tuple(CI)] .* Σ)
        end
    else
        @simd for CI in allCI
            feval[CI] = f([ϵ[i] for i in Tuple(CI)] .* Σ + μ)
        end
    end

    for n_dim in 1:(d - 1)
        # Iteratively integrate out each dimension, i.e. law of iterated expectations
        iter = CartesianIndices(tuple(Tuple(1:n for i in 1:(d - n_dim))...,
                                      Tuple(1:1 for i in 1:n_dim)...)) # Create CartesianIndices for all remaining dimensions
        # ((1:n for i in 1:(d - n_dim + 1))..., (1 for i in 1:(n_dim - 1))...) creates a Tuple of 1:n for the dimensions
        # that are not to be integrated out and uses 1s for the remaining dimensions. We want to use each dimension of feval
        # from 1 to (d - n_dim) (inclusive). So on the first iteration, the tuple should be (1:n, 1:n).
        # We then assign it to the dimensions of feval from 1 to (d - n_dim - 1) (inclusive) to avoid allocations
        feval[iter] .= dropdims(sum(mapslices(fᵢ -> fᵢ .* w, (@view feval[((1:n for i in 1:(d - n_dim + 1))...,
                                                                           (1 for i in 1:(n_dim - 1))...)...]),
                                              dims = (d - n_dim) + 1), dims = (d - n_dim) + 1), dims = (d - n_dim) + 1)
    end

    # Handle final integration on its own
    return sum(w .* (@view feval[:, (1 for i in 1:(d - 1))...])) / π^(d / 2)
end

function gausshermite_expectation(f::Function, μ::AbstractVector{S},
                                  Σ::AbstractVector{<: Number}, ns::AbstractVector{Int}) where {S <: Number}

    d = length(μ)
    @assert length(Σ) == d "The length of μ and Σ must be the same."
    ϵ = Dict{Int, Vector{S}}()
    w = Dict{Int, Vector{S}}()
    for i in 1:d
        ϵ[i], w[i] = gausshermite(ns[i])
        ϵ[i]     .*= sqrt(2.)  # Normalize ϵ and w nodes to approximate standard normal
        # w[i]   ./= sqrt(π) # This step done later to reduce number of computations
    end

    # Evaluate over the tensor grid
    feval = Array{S}(undef, (n for n in ns)...)
    allCI = CartesianIndices(feval)
    if all(μ .≈ 0.)
        @simd for CI in allCI
            feval[CI] = f([ϵ[n_dim][gridᵢ] for (n_dim, gridᵢ) in enumerate(Tuple(CI))] .* Σ)
        end
    else
        @simd for CI in allCI
            feval[CI] = f([ϵ[n_dim][gridᵢ] for (n_dim, gridᵢ) in enumerate(Tuple(CI))] .* Σ + μ)
        end
    end

    # Iteratively integrate out each dimension, i.e. law of iterated expectations
    for n_dim in 1:(d - 1)
        iter = CartesianIndices(tuple(Tuple(1:ns[i] for i in 1:(d - n_dim))...,
                                      Tuple(1:1 for i in 1:n_dim)...))
        feval[iter, 1] .= dropdims(sum(mapslices(fᵢ -> fᵢ .* w[d - n_dim + 1], (@view feval[((1:ns[i] for i in 1:(d - n_dim + 1))...,
                                                                                             (1 for i in 1:(n_dim - 1))...)...]),
                                                 dims = (d - n_dim) + 1), dims = (d - n_dim) + 1), dims = (d - n_dim) + 1)
    end

    # Handle final integration on its own
    return sum(w[1] .* (@view feval[:, (1 for i in 1:(d - 1))...])) / π^(d / 2)
end

function gausshermite_expectation(f::Function, μ::AbstractVector{S},
                                  Σ::AbstractVector{<: Number}, ns::NTuple{N, Int}) where {S<: Number, N}

    d = length(μ)
    @assert length(Σ) == d "The length of μ and Σ must be the same."
    ϵ = Dict{Int, Vector{S}}()
    w = Dict{Int, Vector{S}}()
    for i in 1:d
        ϵ[i], w[i] = gausshermite(ns[i])
        ϵ[i]     .*= sqrt(2.)  # Normalize ϵ and w nodes to approximate standard normal
        # w[i]   ./= sqrt(π) # This step done later to reduce number of computations
    end

    # Evaluate over the tensor grid
    feval = Array{S}(undef, (n for n in ns)...)
    allCI = CartesianIndices(feval)
    if all(μ .≈ 0.)
        @simd for CI in allCI
            feval[CI] = f([ϵ[n_dim][gridᵢ] for (n_dim, gridᵢ) in enumerate(Tuple(CI))] .* Σ)
        end
    else
        @simd for CI in allCI
            feval[CI] = f([ϵ[n_dim][gridᵢ] for (n_dim, gridᵢ) in enumerate(Tuple(CI))] .* Σ + μ)
        end
    end

    # Iteratively integrate out each dimension, i.e. law of iterated expectations
    for n_dim in 1:(d - 1)
        iter = CartesianIndices(tuple(Tuple(1:ns[i] for i in 1:(d - n_dim))...,
                                      Tuple(1:1 for i in 1:n_dim)...))
        feval[iter, 1] .= dropdims(sum(mapslices(fᵢ -> fᵢ .* w[d - n_dim + 1], (@view feval[((1:ns[i] for i in 1:(d - n_dim + 1))...,
                                                                                             (1 for i in 1:(n_dim - 1))...)...]),
                                                 dims = (d - n_dim) + 1), dims = (d - n_dim) + 1), dims = (d - n_dim) + 1)
    end

    # Handle final integration on its own
    return sum(w[1] .* (@view feval[:, (1 for i in 1:(d - 1))...])) / π^(d / 2)
end
