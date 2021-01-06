using BenchmarkTools, RiskAdjustedLinearizations, MATLAB
include(joinpath(dirname(@__FILE__), "..", "rbc_cc", "rbc_cc.jl"))

# Settings: what do you want to do?
autodiff = false
time_matlab = true

# Set up
autodiff_method = autodiff ? :forward : :central
m_rbc_cc = RBCCampbellCochraneHabits()
m = rbc_cc(m_rbc_cc, 0)
z0 = copy(m.z)
y0 = copy(m.y)
Ψ0 = copy(m.Ψ)

# Use deterministic steady state as guesses
solve!(m, z0, y0; algorithm = :deterministic, autodiff = autodiff_method, verbose = :none)
zdet = copy(m.z)
ydet = copy(m.y)
Ψdet = copy(m.Ψ)

if time_matlab
    println("Relaxation algorithm in MATLAB")
    mat"""
    genaffine_rbc_cc_relaxation;
    """
end

println("Relaxation algorithm in Julia")
@btime begin # called the "iterative" method in the original paper
    solve!(m, zdet, ydet, Ψdet; algorithm = :relaxation, autodiff = autodiff_method, verbose = :none)
end

println("Relaxation algorithm with Anderson acceleration")
@btime begin
    solve!(m, zdet, ydet, Ψdet; algorithm = :relaxation, use_anderson = true, m = 3, autodiff = autodiff_method, verbose = :none)
end

sparsity, colorvec = compute_sparsity_pattern(m, :relaxation; sparsity_detection = false)
jac_cache = preallocate_jac_cache(m, :relaxation; sparsity_detection = false)

println("Relaxation with matrix coloring of sparse Jacobians")
@btime begin
    solve!(m, zdet, ydet, Ψdet; algorithm = :relaxation, autodiff = autodiff_method,
           sparse_jacobian = true, sparsity = sparsity, colorvec = colorvec, verbose = :none)
end

println("Relaxation with matrix coloring of sparse Jacobians and caching")
@btime begin
    solve!(m, zdet, ydet, Ψdet; algorithm = :relaxation, autodiff = autodiff_method,
           sparse_jacobian = true, jac_cache = jac_cache, verbose = :none)
end

if time_matlab
    println("Homotopy algorithm in MATLAB")
    mat"""
    genaffine_rbc_cc_homotopy;
    """
end

println("Homotopy algorithm in Julia")
@btime begin # called the "continuation" method in the original paper, but is called homotopy in the original code
    solve!(m, zdet, ydet, Ψdet; algorithm = :homotopy, autodiff = autodiff_method, verbose = :none)
end

sparsity, colorvec = compute_sparsity_pattern(m, :homotopy; sparsity_detection = false)
jac_cache = preallocate_jac_cache(m, :homotopy; sparsity_detection = false)

println("Homotopy with matrix coloring of sparse Jacobians")
@btime begin
    solve!(m, zdet, ydet, Ψdet; algorithm = :homotopy, autodiff = autodiff_method,
           sparse_jacobian = true, sparsity = sparsity, colorvec = colorvec, verbose = :none)
end

println("Homotopy with matrix coloring of sparse Jacobians and caching")
@btime begin
    solve!(m, zdet, ydet, Ψdet; algorithm = :homotopy, autodiff = autodiff_method,
           sparse_jacobian = true, jac_cache = jac_cache, verbose = :none)
end
