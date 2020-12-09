using RiskAdjustedLinearizations, Test
include(joinpath(dirname(@__FILE__), "..", "..", "examples", "crw", "crw.jl"))

# Solve model
m_crw = CoeurdacierReyWinant()
m = crw(m_crw)
for i in 1:100
    try
        solve!(m, m.z, m.y, m.Ψ; algorithm = :homotopy, verbose = :none)
        break
    catch e
        update!(ral, 1.0001 * m.z, 1.0001 .* m.y, 1.0001 .* m.Ψ)
    end
    if i == 100
        solve!(m, m.z, m.y, m.Ψ; algorithm = :homotopy, verbose = :none)
    end
end

# Verify impulse responses with a zero shock is the same as simulate with no shocks
horizon = 100
@testset "Calculate impulse responses for Coeurdacier et al. (2011)" begin

    # No shocks and start from steady state
    state1, jump1 = simulate(m, horizon)
    state2, jump2 = impulse_responses(m, horizon, 1, 0.)

    @test state1 ≈ state2
    @test jump1 ≈ jump2
    @test state1 ≈ repeat(m.z, 1, horizon)
    @test jump1 ≈ repeat(m.y, 1, horizon)

    # No shocks but perturb away from steady state
    state1, jump1 = simulate(m, horizon, 1.01 * m.z)
    state2, jump2 = impulse_responses(m, horizon, 1, 0., 1.01 * m.z)

    @test state1 ≈ state2
    @test jump1 ≈ jump2
    @test !(state1[:, 2] ≈ m.z)
    @test !(jump1[:, 2] ≈ m.y)

    # Now with shocks, from steady state
    shocks = zeros(2, horizon)
    shocks[1] = -3.

    state1, jump1 = impulse_responses(m, horizon, 1, -3.)
    state2, jump2 = impulse_responses(m, horizon, 1, -3., m.z)
    state3, jump3 = simulate(m, horizon, shocks)

    @test state1 ≈ state2
    @test state1 ≈ state3
    @test jump1 ≈ jump2
    @test jump1 ≈ jump3
end
