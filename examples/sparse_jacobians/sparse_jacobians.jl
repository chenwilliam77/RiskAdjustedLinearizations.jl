# This script shows how to compute risk-adjusted linearizations using sparse Jacobian methods
using RiskAdjustedLinearizations, LinearAlgebra, SparseArrays, BenchmarkTools

# Settings
define_functions = false
time_methods     = true
algorithm        = :relaxation
N_approx         = 10          # Number of periods ahead used for forward-difference equations

if define_functions
    include(joinpath(dirname(@__FILE__), "..", "nk_with_capital", "nk_with_capital.jl"))
end

# Set up
m_nk = NKCapital(; N_approx = N_approx) # create parameters
m = nk_capital(m_nk) # instantiate risk-adjusted linearization
zinit = deepcopy(m.z)
yinit = deepcopy(m.y)
Ψinit = deepcopy(m.Ψ)

# Compute steady state
solve!(m; algorithm = :deterministic, verbose = :none)
zdet = deepcopy(m.z)
ydet = deepcopy(m.y)
Ψdet = deepcopy(m.Ψ)

# Using sparsity pattern and matrix coloring to find deterministic steady state
sparsity_det, colorvec_det = compute_sparsity_pattern(m, :deterministic) # Create sparsity pattern/matrix coloring vector
update!(m, zinit, yinit, Ψinit) # need to re-initialize or else the system of equations will be solved at the initial guess
solve!(m; algorithm = :deterministic, ftol = 1e-10, # this call to solve! tries to infer the sparsity pattern
       sparse_jacobian = true, verbose = :none)     # by computing the Jacobian one time w/finite differences
update!(m, zinit, yinit, Ψinit)
solve!(m; algorithm = :deterministic, ftol = 1e-10,                              # need to add tighter tolerance to avoid
       sparse_jacobian = true, sparsity = sparsity_det, colorvec = colorvec_det, # a LAPACK exception when computing the
       verbose = :none)                                                          # Schur decomposition for Ψ

# Using sparsity pattern and matrix coloring to find stochastic steady state via relaxation
# Note that the syntax is essentially the same as the deterministic case, except that
# compute_sparsity_pattern's second argument is now :relaxation
sparsity_rel, colorvec_rel = compute_sparsity_pattern(m, :relaxation)
update!(m, zdet, ydet, Ψdet)
solve!(m; algorithm = :relaxation, autodiff = autodiff_method,
       sparse_jacobian = true, verbose = :none)
update!(m, zdet, ydet, Ψdet)
solve!(m; algorithm = :relaxation, autodiff = autodiff_method,
       sparse_jacobian = true, sparsity = sparsity_rel, colorvec = colorvec_rel,
       verbose = :none)

# Using sparsity pattern and matrix coloring to find stochastic steady state via homotopy
sparsity_hom, colorvec_hom = compute_sparsity_pattern(m, :homotopy)
update!(m, zdet, ydet, Ψdet)
solve!(m; algorithm = :homotopy, autodiff = autodiff_method,
       sparse_jacobian = true, verbose = :none)
update!(m, zdet, ydet, Ψdet)
solve!(m; algorithm = :homotopy, autodiff = autodiff_method,
       sparse_jacobian = true, sparsity = sparsity_hom, colorvec = colorvec_hom,
       verbose = :none)

# Using Jacobian cache to find deterministic steady state
# Like compute_sparsity_pattern, the user only needs to select the
# algorithm for which the Jacobian cache will be used.
jac_cache_det = preallocate_jac_cache(m, :deterministic)
update!(m, zinit, yinit, Ψinit)
solve!(m; algorithm = :deterministic, autodiff = autodiff_method,
       ftol = 1e-10, sparse_jacobian = true, jac_cache = jac_cache_det,
       verbose = :none)

# Using Jacobian cache to find stochastic steady state via relaxation
jac_cache_rel = preallocate_jac_cache(m, :relaxation)
update!(m, zdet, ydet, Ψdet)
solve!(m; algorithm = :relaxation, ftol = 1e-10,
       sparse_jacobian = true, jac_cache = jac_cache_rel,
       verbose = :none)

# Using Jacobian cache to find stochastic steady state via homotopy
jac_cache_hom = preallocate_jac_cache(m, :homotopy)
update!(m, zdet, ydet, Ψdet)
update!(m, zdet, ydet, Ψdet)
solve!(m; algorithm = :homotopy, autodiff = autodiff_method,
       sparse_jacobian = true, jac_cache = jac_cache_hom,
       verbose = :none)

if time_methods
    println("Deterministic steady state with dense Jacobian")
    @btime begin
        update!(m, zinit, yinit, Ψinit)
        solve!(m; algorithm = :deterministic, verbose = :none)
    end

    println("Deterministic steady state with sparsity pattern of Jacobian and matrix coloring vector")
    @btime begin
        update!(m, zinit, yinit, Ψinit)
        solve!(m; algorithm = :deterministic, ftol = 1e-10,
               sparse_jacobian = true, sparsity = sparsity_det, colorvec = colorvec_det,
               verbose = :none)
    end

    println("Deterministic steady state with sparsity pattern of Jacobian, matrix coloring vector, and caching")
    @btime begin
        update!(m, zinit, yinit, Ψinit)
        solve!(m; algorithm = :deterministic, autodiff = autodiff_method,
               ftol = 1e-10, sparse_jacobian = true, jac_cache = jac_cache_det,
               verbose = :none)
    end

    println("Relaxation with dense Jacobian")
    @btime begin
        update!(m, zdet, ydet, Ψdet)
        solve!(m; algorithm = :relaxation, verbose = :none)
    end

    println("Relaxation with sparsity pattern of Jacobian and matrix coloring vector")
    @btime begin
        update!(m, zdet, ydet, Ψdet)
        solve!(m; algorithm = :relaxation, autodiff = autodiff_method,
               sparse_jacobian = true, sparsity = sparsity_rel, colorvec = colorvec_rel,
               verbose = :none)
    end

    println("Relaxation with sparsity pattern of Jacobian, matrix coloring vector, and caching")
    @btime begin
        update!(m, zdet, ydet, Ψdet)
        solve!(m; algorithm = :relaxation, autodiff = autodiff_method,
               sparse_jacobian = true, jac_cache = jac_cache_rel,
               verbose = :none)
    end

    println("Homotopy with dense Jacobian")
    @btime begin
        update!(m, zdet, ydet, Ψdet)
        solve!(m; algorithm = :homotopy, verbose = :none)
    end

    println("Homotopy with sparsity pattern of Jacobian and matrix coloring vector")
    @btime begin
        update!(m, zdet, ydet, Ψdet)
        solve!(m; algorithm = :homotopy,
               sparse_jacobian = true, sparsity = sparsity_hom, colorvec = colorvec_hom,
               verbose = :none)
    end

    println("Homotopy with sparsity pattern of Jacobian, matrix coloring vector, and caching")
    @btime begin
        update!(m, zdet, ydet, Ψdet)
        solve!(m; algorithm = :homotopy, autodiff = autodiff_method,
               sparse_jacobian = true, jac_cache = jac_cache_hom,
               verbose = :none)
    end
end
