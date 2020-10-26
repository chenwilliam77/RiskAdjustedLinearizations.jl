using RiskAdjustedLinearizations, Test, Random
include(joinpath(dirname(@__FILE__), "../../examples/rbc_cc/rbc_cc.jl"))

# Solve model
m_rbc_cc = RBCCampbellCochraneHabits()
m = rbc_cc(m_rbc_cc)
solve!(m, m.z, m.y)

# Simulate with no shocks
horizon = 100
zero_shocks = zeros(1, horizon) # 100 fake draws
@testset "Simulate an RBC Model w/Campbell-Cochrane Habits with no shocks" begin
    state1, jump1 = simulate(m, horizon)
    state2, jump2 = simulate(m, horizon, m.z)
    state3, jump3 = simulate(m, horizon, zero_shocks)
    state4, jump4 = simulate(m, horizon, zero_shocks, m.z)
    state5, jump5 = simulate(m, horizon, 1.01 * m.z) # perturb away from steady state
    state6, jump6 = simulate(m, horizon, zero_shocks, 1.01 * m.z)

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
end

Random.seed!(1793)
shocks = rand(Normal(0, 1.), 1, horizon) # 100 draws from a standard normal

@testset "Simulate an RBC Model w/Campbell-Cochrane Habits with shocks" begin
    state1, jump1 = simulate(m, horizon)
    state3, jump3 = simulate(m, horizon, shocks)
    state4, jump4 = simulate(m, horizon, shocks, m.z)

    @test state3 ≈ state4
    @test jump3 ≈ jump4
    @test !(state1 ≈ state4)
    @test !(jump1 ≈ jump4)
end
