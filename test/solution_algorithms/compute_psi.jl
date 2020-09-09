using RiskAdjustedLinearizations, JLD2, Test

output  = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/compute_psi_output.jld2"), "r")
GAM1    = output["GAM1"]
GAM2    = output["GAM2"]
GAM3    = output["GAM3"]
GAM4    = output["GAM4"]
GAM5    = output["GAM5"]
GAM6    = output["GAM6"]
JV      = output["JV"]
Psi     = output["Psi"]
Psi_det = output["Psi_det"]

@testset "QZ decomposition for Ψ" begin
    @test RiskAdjustedLinearizations.compute_Ψ(GAM1, GAM2, GAM3, GAM4, GAM5, GAM6, JV)              ≈ Psi
    @test RiskAdjustedLinearizations.compute_Ψ(GAM1, GAM2, GAM3, GAM4, GAM5, GAM6)                  ≈ Psi_det
    @test RiskAdjustedLinearizations.compute_Ψ(GAM1, GAM2, GAM3, GAM4, GAM5, GAM6, zeros(size(JV))) ≈ Psi_det
end
