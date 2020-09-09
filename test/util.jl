using RiskAdjustedLinearizations, ForwardDiff, Random

Random.seed!(1793)

aR = rand(2)
bR = rand(3)
aD = rand(2)
bD = rand(3)

AR = rand(ForwardDiff.Dual, 2, 2, 3)
BR = rand(ForwardDiff.Dual, 3, 2, 3, 2)
AD = rand(ForwardDiff.Dual, 2, 2, 3)
BD = rand(ForwardDiff.Dual, 3, 2, 3, 2)
@testset "dualarray and dualvector" begin
    @test RiskAdjustedLinearizations.dualarray(AR, BR) == similar(AR)
    @test RiskAdjustedLinearizations.dualarray(BR, AR) == similar(BR)
    @test RiskAdjustedLinearizations.dualarray(AR, BD) == similar(AR, eltype(BD))
    @test RiskAdjustedLinearizations.dualarray(BD, AR) == similar(BD, eltype(AR))
    @test RiskAdjustedLinearizations.dualarray(AD, BD) == similar(AD)
    @test RiskAdjustedLinearizations.dualarray(BD, AD) == similar(BD)
    @test RiskAdjustedLinearizations.dualarray(AD, BR) == similar(AD)
    @test RiskAdjustedLinearizations.dualarray(BR, AD) == similar(BR, eltype(AD))

    @test RiskAdjustedLinearizations.dualvector(aR, bR) == similar(aR)
    @test RiskAdjustedLinearizations.dualvector(bR, aR) == similar(bR)
    @test RiskAdjustedLinearizations.dualvector(aR, bD) == similar(aR, eltype(bD))
    @test RiskAdjustedLinearizations.dualvector(bD, aR) == similar(bD, eltype(aR))
    @test RiskAdjustedLinearizations.dualvector(aD, bD) == similar(aD)
    @test RiskAdjustedLinearizations.dualvector(bD, aD) == similar(bD)
    @test RiskAdjustedLinearizations.dualvector(aD, bR) == similar(aD)
    @test RiskAdjustedLinearizations.dualvector(bR, aD) == similar(bR, eltype(aD))
end

nothing
