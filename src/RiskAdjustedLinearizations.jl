isdefined(Base, :__precompile__) && __precompile__(false)

module RiskAdjustedLinearizations

import Base.show
using ForwardDiff, LinearAlgebra, Printf, SparseArrays# , SparseDiffTools, SparsityDetection
using UnPack
using BandedMatrices: Ones, Zeros
using DiffEqBase: DiffCache, get_tmp, dualcache
using NLsolve: nlsolve

# Utilities
include("util.jl")

# RiskAdjustedLinearization
include("risk_adjusted_linearization.jl")

# Numerical Algorithms
include("numerical_algorithms/compute_psi.jl")
include("numerical_algorithms/blanchard_kahn.jl")
include("numerical_algorithms/relaxation.jl")
include("numerical_algorithms/solve.jl")

export RiskAdjustedLinearization, update!, nonlinear_system, linearized_system, solve!
end # module
