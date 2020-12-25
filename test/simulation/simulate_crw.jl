using RiskAdjustedLinearizations, Test
include(joinpath(dirname(@__FILE__), "..", "..", "examples", "crw", "crw.jl"))

# Solve model
m_crw = CoeurdacierReyWinant()
m = crw(m_crw)
solve!(m, m.z, m.y, m.Ψ; algorithm = :homotopy, verbose = :none, testing = true)

# Simulate with no shocks
horizon = 100
zero_shocks = zeros(2, horizon) # 100 fake draws
@testset "Simulate Coeurdacier et al. (2011) with no shocks" begin
    state1, jump1 = simulate(m, horizon)
    state2, jump2 = simulate(m, horizon, m.z)
    state3, jump3 = simulate(m, horizon, zero_shocks)
    state4, jump4 = simulate(m, horizon, zero_shocks, m.z)
    state5, jump5 = simulate(m, horizon, 1.01 * m.z) # perturb away from steady state
    state6, jump6 = simulate(m, horizon, zero_shocks, 1.01 * m.z)
    state7, jump7 = simulate(m, vec(zero_shocks[:, 1]), 1.01 * m.z)

    @test state1 ≈ state2
    @test state1 ≈ state3
    @test state1 ≈ state4
    @test jump1 ≈ jump2
    @test jump1 ≈ jump3
    @test jump1 ≈ jump4
    @test state5 ≈ state6
    @test jump5 ≈ jump6
    @test state1 ≈ repeat(m.z, 1, horizon) # check state1 remains at steady state
    @test jump1 ≈ repeat(m.y, 1, horizon)
    @test state7 ≈ state6[:, 1]
    @test jump7 ≈ jump6[:, 1]
end

shocks = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "reference", "crw_shocks.jld2"), "r")["shocks"] # 100 draws from a bivariate standard normal

@testset "Simulate Coeurdacier et al. (2011) with shocks" begin
    state1, jump1 = simulate(m, horizon)
    state3, jump3 = simulate(m, horizon, shocks)
    state4, jump4 = simulate(m, horizon, shocks, m.z)
    state5, jump5 = simulate(m, shocks[:, 1], m.z)

    @test state3 ≈ state4
    @test jump3 ≈ jump4
    @test !(state1 ≈ state4)
    @test !(jump1 ≈ jump4)
    @test state5 ≈ state4[:, 1]
    @test jump5 ≈ jump4[:, 1]
end
