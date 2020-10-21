using RiskAdjustedLinearizations, ForwardDiff, Random, Test

Random.seed!(1793)

aR = rand(2)
bR = rand(3)
aD = rand(2)
bD = rand(3)

AR = rand(2, 2, 3)
BR = rand(3, 2, 3, 2)
AD = ForwardDiff.Dual.(rand(2, 2, 3))
BD = ForwardDiff.Dual.(rand(3, 2, 3, 2))

@testset "dualarray and dualvector" begin
    @test length(RiskAdjustedLinearizations.dualarray(AR, BR)) == length(AR)
    @test length(RiskAdjustedLinearizations.dualarray(BR, AR)) == length(BR)
    @test length(RiskAdjustedLinearizations.dualarray(AR, BD)) == length(similar(AR, eltype(BD)))
    @test length(RiskAdjustedLinearizations.dualarray(BD, AR)) == length(similar(BD, eltype(AR)))
    @test length(RiskAdjustedLinearizations.dualarray(AD, BD)) == length(AD)
    @test length(RiskAdjustedLinearizations.dualarray(BD, AD)) == length(BD)
    @test length(RiskAdjustedLinearizations.dualarray(AD, BR)) == length(AD)
    @test length(RiskAdjustedLinearizations.dualarray(BR, AD)) == length(similar(BR, eltype(AD)))

    @test length(RiskAdjustedLinearizations.dualvector(aR, bR)) == length(aR)
    @test length(RiskAdjustedLinearizations.dualvector(bR, aR)) == length(bR)
    @test length(RiskAdjustedLinearizations.dualvector(aR, bD)) == length(similar(aR, eltype(bD)))
    @test length(RiskAdjustedLinearizations.dualvector(bD, aR)) == length(similar(bD, eltype(aR)))
    @test length(RiskAdjustedLinearizations.dualvector(aD, bD)) == length(aD)
    @test length(RiskAdjustedLinearizations.dualvector(bD, aD)) == length(bD)
    @test length(RiskAdjustedLinearizations.dualvector(aD, bR)) == length(aD)
    @test length(RiskAdjustedLinearizations.dualvector(bR, aD)) == length(similar(bR, eltype(aD)))
end

nothing
