isdefined(Base, :__precompile__) && __precompile__(false)

module RiskAdjustedLinearizations

import Base: show, getindex
using ArrayInterface, FastGaussQuadrature, FiniteDiff, ForwardDiff, LabelledArrays, LinearAlgebra, Printf
using SparseArrays, SparseDiffTools, SparsityDetection, UnPack
using BandedMatrices: Zeros
using NLsolve: nlsolve, OnceDifferentiable

# Utilities
include("util.jl") # in principle, several of these utility functions could work for other packages

# RiskAdjustedLinearization
include("cache_types.jl")
include("risk_adjusted_linearization.jl")
include("sparse_jacobian_helpers.jl")

# Numerical Algorithms
include("numerical_algorithms/compute_psi.jl")
include("numerical_algorithms/blanchard_kahn.jl")
include("numerical_algorithms/relaxation.jl")
include("numerical_algorithms/homotopy.jl")
include("numerical_algorithms/solve.jl")

# Simulations and Simulation-based Diagnostics
include("simulation/simulate.jl")
include("simulation/impulse_responses.jl")
include("simulation/gausshermite_expectation.jl")
include("simulation/euler_equation_error.jl")

export
    # main functions
    RiskAdjustedLinearization, update!, nonlinear_system, linearized_system, solve!,
    steady_state_errors,

    # simulation related functions
    simulate, impulse_responses, gausshermite_expectation, euler_equation_error,
    dynamic_euler_equation_error,

    # sparsity helpers
    preallocate_jac_cache, compute_sparsity_pattern, update_sparsity_pattern!
end # module
