# This script shows how to compute risk-adjusted linearizations using sparse arrays
# for caches and sparse differentiation for the Jacobians of μ, ξ, and 𝒱 .
using RiskAdjustedLinearizations, LinearAlgebra, SparseArrays
using BenchmarkTools, Test, SparseDiffTools

# Settings
define_functions = true
time_methods     = true
algorithm        = :relaxation # Note that while both methods work with sparse array caches, only relaxation works for sparse differentiation.
N_approx         = 10          # Number of periods ahead used for forward-difference equations

if define_functions
    include(joinpath(dirname(@__FILE__), "..", "nk_with_capital", "nk_with_capital.jl"))
end

# Set up
m_nk = NKCapital(; N_approx = N_approx) # create parameters

# Sparse arrays for caches
m = nk_capital(m_nk; sparse_arrays = true)

## The keyword sparse_arrays tells the nk_capital function
## to make Γ₅, Γ₆ sparse arrays and to add the keyword argument
## `Λ_Σ_cache_init = dims -> spzeros(dims...)`
## when calling the constructor, i.e.
## RiskAdjustedLinearization(...; Λ_Σ_cache_init = ...)
solve!(m; algorithm = algorithm, verbose = :none)

# Risk-adjusted linearization with sparse differentiation
# for the Jacobians of μ, ξ, and 𝒱 in addition to sparse caches.

## The first approach directly uses
## the constructor of a RiskAdjustedLinearization
## to determine the sparsity pattern.
## Since the initial guess for Ψ is a matrix of zeros,
## we will only use sparse differentiation for μ and ξ at first.
## Within nk_capital, the keyword `sparse_jacobian = [:μ, :ξ]` is passed
## to the constructor, i.e.
## RiskAdjustedLinearization(...; sparse_jacobian = ...)
m_sparsejac = nk_capital(m_nk; sparse_arrays = true, sparse_jacobian = [:μ, :ξ])

## Check the caches for the Jacobians of μ and ξ are actually sparse
@test issparse(m_sparsejac[:Γ₁])
@test issparse(m_sparsejac[:Γ₂])
@test issparse(m_sparsejac[:Γ₃])
@test issparse(m_sparsejac[:Γ₄])

## Now solve the model! Note that sparse differentiation
## can be fragile sometimes and result in NaNs or undefined
## numbers appearing during calls to `nlsolve`. Re-running
## `solve!` repeatedly or reconstructing `m_sparsejac` again
## will usually lead to a successful run.
solve!(m_sparsejac; algorithm = algorithm)
@test norm(steady_state_errors(m_sparsejac), Inf) < 1e-8

## The second approach calls `update_sparsity_pattern!`
## on an existing `RiskAdjustedLinearization`

### Create dictionaries for specifying the sparsity pattern
### Here, we will tell `m_sparsejac`
### to now use sparse differentiation for J𝒱
### and to use a new sparsity pattern for the
### Jacobians of μ and ξ.
sparsity = Dict()
colorvec = Dict()
sparsity[:J𝒱] = sparse(m[:JV])
@test isempty(sparsity[:J𝒱].nzval) # However, in the case of the current NK model, the entropy actually does not depend on z.
colorvec[:J𝒱] = ones(Int64, size(sparsity[:J𝒱], 2)) # Instead, we just pass in a coloring vector of this form
jac_to_sym = (μz = :Γ₁, μy = :Γ₂, ξz = :Γ₃, ξy = :Γ₄)
for k in [:μz, :μy, :ξz, :ξy]
    sparsity[k] = m_sparsejac[jac_to_sym[k]]
    colorvec[k] = matrix_colors(sparsity[k])
end

@test !issparse(m_sparsejac[:JV]) # JV is not sparse currently

update_sparsity_pattern!(m_sparsejac, [:μ, :ξ]; sparsity = sparsity) # Don't have to provide the matrix coloring vector (note 𝒱 is not included
                                                                      # b/c calling matrix_color on its sparsity pattern will error)
update_sparsity_pattern!(m_sparsejac, [:μ, :ξ, :𝒱]; sparsity = sparsity, # But if the coloring vector already exists,
                         colorvec = colorvec)                            # then you may as well pass that information, too.

@test issparse(m_sparsejac[:JV]) # Now JV is sparse

### If you ever need to look at the sparsity pattern or coloring vector,
### you can call `linearized_system(m_sparsejac).sparse_jac_caches` or
### `m_sparsejac.linearization.sparse_jac_caches`, which is a
### `NamedTuple` whose values include the sparsity pattern and coloring vector,
### as well as a cache used by forwarddiff_color_jacobian!
for k in [:μz, :μy, :ξz, :ξy, :J𝒱]
    @test issparse(m_sparsejac.linearization.sparse_jac_caches[k][:sparsity])
end

### Now solve again with sparse Jacobian of 𝒱, too!
solve!(m_sparsejac, m_sparsejac.z .* 1.01, m_sparsejac.y .* 1.01, m_sparsejac.Ψ .* 1.01; algorithm = algorithm)
@test norm(steady_state_errors(m_sparsejac), Inf) < 1e-8

if time_methods
    m_dense = nk_capital(m_nk)
    zinit = deepcopy(m_dense.z)
    yinit = deepcopy(m_dense.y)
    Ψinit = deepcopy(m_dense.Ψ)

    solve!(m; algorithm = :deterministic, verbose = :none)
    zdet = deepcopy(m.z)
    ydet = deepcopy(m.y)
    Ψdet = deepcopy(m.Ψ)

    println("Deterministic steady state with dense caches for Γ₅, Γ₆, Λ, and Σ")
    @btime begin
        update!(m_dense, zinit, yinit, Ψinit)
        solve!(m_dense; algorithm = :deterministic, verbose = :none)
    end

    println("Deterministic steady state with sparse caches for Γ₅, Γ₆, Λ, and Σ")
    @btime begin
        update!(m, zinit, yinit, Ψinit)
        solve!(m; algorithm = :deterministic, verbose = :none)
    end

    println("Relaxation with dense caches for Γ₅, Γ₆, Λ, and Σ")
    @btime begin
        update!(m_dense, zdet, ydet, Ψdet)
        solve!(m_dense; algorithm = :relaxation, verbose = :none)
    end

    println("Relaxation with sparse caches for Γ₅, Γ₆, Λ, and Σ")
    @btime begin
        update!(m, zdet, ydet, Ψdet)
        solve!(m; algorithm = :relaxation, verbose = :none)
    end

    println("Homotopy with dense caches for Γ₅, Γ₆, Λ, and Σ")
    @btime begin
        update!(m_dense, zdet, ydet, Ψdet)
        solve!(m_dense; algorithm = :homotopy, verbose = :none)
    end

    println("Homotopy with sparse caches for Γ₅, Γ₆, Λ, and Σ")
    @btime begin
        update!(m, zdet, ydet, Ψdet)
        solve!(m; algorithm = :homotopy, verbose = :none)
    end

    println("Relaxation with sparse caches and differentiation")
    @btime begin
        update!(m_sparsejac, zdet, ydet, Ψdet)
        solve!(m; algorithm = :relaxation, verbose = :none)
    end
end
