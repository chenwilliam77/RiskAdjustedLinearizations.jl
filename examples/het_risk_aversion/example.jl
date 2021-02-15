# This script actually solves the CoeurdacierReyWinant model with a risk-adjusted linearization
# and times the methods, if desired
using BenchmarkTools, RiskAdjustedLinearizations, Test, JLD2
include("bansal_yaron_2004.jl")
# include("het_risk_aversion.jl")

# What to do?
solve_rep_agent = true
check_mat       = true

# Set up
if solve_rep_agent
    m_rep = BansalYaron2004()
    m = bansal_yaron_2004(m_rep; sparse_arrays = true, sparse_jacobian = [:μ, :ξ])
    z0 = copy(m.z)
    y0 = copy(m.y)
    Ψ0 = copy(m.Ψ)

    # Solve!
    solve!(m, m.z, m.y; algorithm = :deterministic)
    solve!(m; algorithm = :relaxation)

    # Now solve with a higher risk aversion, using previous steady state as a guess
    # Note the parameters used in the equations for the RiskAdjustedLinearization
    # are from the field `m_rep.p`, which is a LabelledArray and thus allows us to
    # mutate the parameters without re-compiling the functions contained in `m`.
    m_rep.p[:γ] = 9.5
    solve!(m; algorithm = :relaxation)
    m_rep.p[:γ] = 9. # restore the original parameter so, for example, the check_mat test passes
end

if check_mat
    using MAT
    matout = matread("dynare_ss.mat")
    @unpack J, S = m_rep

    solve!(m, z0, y0; algorithm = :deterministic)
    @test log(matout["Q"]) ≈ m.y[J[:q]] atol=1e-6
    @test log(matout["Omega"]) ≈ m.y[J[:ω]] atol=1e-6
    @test log(matout["V"]) ≈ m.y[J[:v]] atol=1e-6
    @test log(matout["CE"]) ≈ m.y[J[:ce]] atol=1e-6
    @test log(matout["PQ"]) ≈ m.y[J[:pq1]] atol=1e-6
    @test log(matout["DQ"]) ≈ m.y[J[:dq1]] atol=1e-6
    @test log(matout["POmega"]) ≈ m.y[J[:pω1]] atol=1e-6
    @test log(matout["DOmega"]) ≈ m.y[J[:dω0]] atol=1e-6
    @test log(matout["X"]) ≈ m.z[S[:x]] atol=1e-6
    @test log(matout["Y"]) ≈ m.z[S[:yy]] atol=1e-6
    @test matout["SigSq"] ≈ m.z[S[:σ²_y]] atol=1e-6
end
