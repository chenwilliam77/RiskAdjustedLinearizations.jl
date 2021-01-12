# This script times the results from using sparse arrays for caching; sparse Jacobians
# of μ, ξ, and 𝒱 ; and sparse Jacobians for calls to nlsolve.
using RiskAdjustedLinearizations, LinearAlgebra, SparseArrays
using BenchmarkTools, Test, SparseDiffTools

# Settings
define_functions = true
time_methods     = true
algorithm        = :relaxation # Note only relaxation works for sparse differentiation.
N_approx         = 10          # Number of periods ahead used for forward-difference equations

if define_functions
    include(joinpath(dirname(@__FILE__), "..", "nk_with_capital", "nk_with_capital.jl"))
end

# Set up

## Instantiate object
m_nk = NKCapital(; N_approx = N_approx) # create parameters
m = nk_capital(m_nk; sparse_arrays = true, sparse_jacobian = [:μ, :ξ])
zinit = deepcopy(m.z)
yinit = deepcopy(m.y)
Ψinit = deepcopy(m.Ψ)

## Solve for steady state once and update sparsity pattern
solve!(m; algorithm = algorithm, verbose = :none)

sparsity = Dict()
colorvec = Dict()
sparsity[:J𝒱] = sparse(m[:JV])
colorvec[:J𝒱] = isempty(sparsity[:J𝒱].nzval) ? ones(Int64, size(sparsity[:J𝒱], 2)) : matrix_colors(sparsity[:J𝒱])
update_sparsity_pattern!(m, [:𝒱]; sparsity = sparsity, colorvec = colorvec)

## Solve w/sparse array caching; sparse differentiation of Jacobians of
## μ, ξ, and 𝒱 ; and sparse differentiation of the objective functions in `nlsolve`
jac_cache = preallocate_jac_cache(m, algorithm)
update!(m, zinit, yinit, Ψinit)
solve!(m; algorithm = algorithm, sparse_jacobian = true, jac_cache = jac_cache)

if time_methods
    m_dense = nk_capital(m_nk)

    @info "Timing solve! with varying degrees of sparsiy"

    println("Dense Array Caches and Dense Jacobians")
    @btime begin
        update!(m_dense, zinit, yinit, Ψinit)
        solve!(m_dense; algorithm = algorithm, verbose = :none)
    end
    # ~ 2.48 s

    println("Sparse Array Caches and Sparse Jacobians for Equilibrium Functions")
    @btime begin
        update!(m, zinit, yinit, Ψinit)
        solve!(m; algorithm = algorithm, verbose = :none)
    end
    # ~ 2.37 s

    println("Sparse Jacobians for nlsolve")
    @btime begin
        update!(m_dense, zinit, yinit, Ψinit)
        solve!(m_dense; algorithm = algorithm, sparse_jacobian = true,
               jac_cache = jac_cache, verbose = :none)
    end
    # ~ 0.85 s

    println("Sparse Array Caches, Sparse Jacobians for Equilibrium Functions, and Sparse Jacobians for nlsolve")
    @btime begin
        update!(m, zinit, yinit, Ψinit)
        solve!(m; algorithm = algorithm, sparse_jacobian = true, jac_cache = jac_cache, verbose = :none)
    end
    # ~ 0.9s

    @test m_dense.z ≈ m.z
    @test m_dense.y ≈ m.y
    @test m_dense.Ψ ≈ m.Ψ
end
