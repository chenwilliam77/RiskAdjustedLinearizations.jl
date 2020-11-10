# It is assumed consumption is a jump variable and the user can provide
# a quadrature function to facilitate an expectation calculation
function standard_normal_gausshermite(n::Int)
    ϵᵢ, wᵢ = gausshermite(n) # approximates exp(-x²)
    ϵᵢ   .*= sqrt(2.)        # Normalize ϵᵢ and wᵢ nodes to approximate standard normal
    wᵢ   ./= sqrt(π)

    return ϵᵢ, wᵢ
end

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
    feval = Array{S}(undef, (n for i in 1:d))
    allCI = CartesianIndices(feval)
    if all(μ .≈ 0.)
        @simd for CI in allCI
            feval[CI] = f([ϵ[i] for i in CI] .* Σ)
        end
    else
        @simd for CI in allCI
            feval[CI] = f([ϵ[i] for i in CI] .* Σ + μ)
        end
    end

    for n_dim in 1:(d - 1)
        # Iteratively integrate out each dimension, i.e. law of iterated expectations
        iter = CartesianIndices((1:n for i in 1:(d - n_dim))) # Create CartesianIndices for all remaining dimensions
        # ((1:n for i in 1:(d - n_dim + 1))..., (1 for i in 1:(n_dim - 1))...) creates a Tuple of 1:n for the dimensions
        # that are not to be integrated out and uses 1s for the remaining dimensions. We want to use each dimension of feval
        # from 1 to (d - n_dim) (inclusive). So on the first iteration, the tuple should be (1:n, 1:n).
        # We then assign it to the dimensions of feval from 1 to (d - n_dim - 1) (inclusive) to avoid allocations
        feval[iter, 1] .= sum(mapslices(fᵢ -> fᵢ .* w, (@view feval[((1:n for i in 1:(d - n_dim + 1))...,
                                           (1 for i in 1:(n_dim - 1))...)...]),
                                        dims = (d - n_dim) + 1), dims = (d - n_dim) + 1)
    end

    # Handle final integration on its own
    return sum(w .* (@view feval[:, (1 for i in 1:(d - 1))...])) / sqrt(π)
end

function gausshermite_expectation(f::Function, μ::AbstractVector{S},
                                  Σ::AbstractVector{<: Number}, ns::AbstractVector{Int} = fill(10, length(μ))) where {S <: Number}

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
    feval = Array{S}(undef, (n for n in ns))
    allCI = CartesianIndices(feval)
    if all(μ .≈ 0.)
        @simd for CI in allCI
            feval[CI] = f([ϵ[n_dim][gridᵢ] for (n_dim, gridᵢ) in enumerate(CI)] .* Σ)
        end
    else
        @simd for CI in allCI
            feval[CI] = f([ϵ[n_dim][gridᵢ] for (n_dim, gridᵢ) in enumerate(CI)] .* Σ + μ)
        end
    end

    # Iteratively integrate out each dimension, i.e. law of iterated expectations
    for n_dim in 1:(d - 1)
        iter = CartesianIndices((1:ns[i] for i in 1:(d - n_dim)))
        feval[iter, 1] .= sum(mapslices(fᵢ -> fᵢ .* w[d - n_dim + 1], (@view feval[((1:ns[i] for i in 1:(d - n_dim + 1))...,
                                                                                    (1 for i in 1:(n_dim - 1))...)...]),
                                        dims = (d - n_dim) + 1), dims = (d - n_dim) + 1)
    end

    # Handle final integration on its own
    return sum(w[1] .* (@view feval[:, (1 for i in 1:(d - 1))...])) / sqrt(π)
end

function gausshermite_expectation(f::Function, μ::AbstractVector{<: Number},
                                  Σ::AbstractVector{<: Number},
                                  ns::NTuple{Int, N} = (10 for i in 1: length(μ))) where N

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
    feval = Array{S}(undef, (n for n in ns))
    allCI = CartesianIndices(feval)
    if all(μ .≈ 0.)
        @simd for CI in allCI
            feval[CI] = f([ϵ[n_dim][gridᵢ] for (n_dim, gridᵢ) in enumerate(CI)] .* Σ)
        end
    else
        @simd for CI in allCI
            feval[CI] = f([ϵ[n_dim][gridᵢ] for (n_dim, gridᵢ) in enumerate(CI)] .* Σ + μ)
        end
    end

    # Iteratively integrate out each dimension, i.e. law of iterated expectations
    for n_dim in 1:(d - 1)
        iter = CartesianIndices((1:ns[i] for i in 1:(d - n_dim)))
        feval[iter, 1] .= sum(mapslices(fᵢ -> fᵢ .* w[d - n_dim + 1], (@view feval[((1:ns[i] for i in 1:(d - n_dim + 1))...,
                                                                                    (1 for i in 1:(n_dim - 1))...)...]),
                                        dims = (d - n_dim) + 1), dims = (d - n_dim) + 1)
    end

    # Handle final integration on its own
    return sum(w[1] .* (@view feval[:, (1 for i in 1:(d - 1))...])) / sqrt(π)
end
