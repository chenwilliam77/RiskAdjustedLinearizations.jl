using RiskAdjustedLinearizations, FastGaussQuadrature, Test

@testset "Gauss-Hermite Quadrature for Expectations of Functions of Normally Distributed Random Variables/Vectors" begin
    f(x) = x  # calculate the expected value
    g(x) = 1. # calculate the probability

    ϵᵢ, wᵢ = RiskAdjustedLinearizations.standard_normal_gausshermite(3)
    true_eps, true_wts = gausshermite(3)
    @test ϵᵢ == true_eps .* sqrt(2.)
    @test wᵢ == true_wts ./ sqrt(π)

    mean51 = gausshermite_expectation(f, 5., 1., 5)
    mean01 = gausshermite_expectation(f, 0., 1., 5)
    mean53 = gausshermite_expectation(f, 5., 3., 5)
    mean03 = gausshermite_expectation(f, 0., 3., 5)
    prob   = gausshermite_expectation(g, -5., .1)

    @test mean51 ≈ 5.
    @test mean53 ≈ 5.
    @test isapprox(mean01, 0., atol = 1e-14)
    @test isapprox(mean03, 0., atol = 1e-14)
    @test prob ≈ 1.

    h1(x) = x[1]
    h2(x) = x[2]

    prob1 = gausshermite_expectation(g, [.5, 5.], [1., 1.], 5)
    mean11 = gausshermite_expectation(h1, [.5, 5.], [1., 1.], 5)
    mean21 = gausshermite_expectation(h2, [.5, 5.], [1., 1.], 5)
    prob2 = gausshermite_expectation(g, [5., -1.], [1., 1.], (5, 5))
    mean12 = gausshermite_expectation(h1, [.5, 5.], [1., 1.], (5, 5))
    mean22 = gausshermite_expectation(h2, [.5, 5.], [1., 1.], (5, 5))
    prob3 = gausshermite_expectation(g, [5., -1.], [1., 1.], [5, 5])
    mean13 = gausshermite_expectation(h1, [.5, 5.], [1., 1.], [5, 5])
    mean23 = gausshermite_expectation(h2, [.5, 5.], [1., 1.], [5, 5])

    @test prob1 ≈ prob2 ≈ prob3 ≈ 1.
    @test mean11 ≈ mean12 ≈ mean13 ≈ .5
    @test mean21 ≈ mean22 ≈ mean23 ≈ 5.
end

nothing
