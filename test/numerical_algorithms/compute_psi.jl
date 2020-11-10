using RiskAdjustedLinearizations, JLD2, Test

output  = JLD2.jldopen(joinpath(dirname(@__FILE__), "..", "reference", "compute_psi_output.jld2"), "r")
GAM1    = output["GAM1"]
GAM2    = output["GAM2"]
GAM3    = output["GAM3"]
GAM4    = output["GAM4"]
GAM5    = output["GAM5"]
GAM6    = output["GAM6"]
JV      = output["JV"]
Psi     = output["Psi"]
Psi_det = output["Psi_det"]

Nzy = sum(size(GAM5))
AA = Matrix{Complex{Float64}}(undef, Nzy, Nzy)
BB = similar(AA)
@testset "QZ decomposition for Ψ" begin
    # Out-of-place
    @test RiskAdjustedLinearizations.compute_Ψ(GAM1, GAM2, GAM3, GAM4, GAM5, GAM6, JV)              ≈ Psi
    @test RiskAdjustedLinearizations.compute_Ψ(GAM1, GAM2, GAM3, GAM4, GAM5, GAM6)                  ≈ Psi_det
    @test RiskAdjustedLinearizations.compute_Ψ(GAM1, GAM2, GAM3, GAM4, GAM5, GAM6, zeros(size(JV))) ≈ Psi_det

    # In-place
    @test RiskAdjustedLinearizations.compute_Ψ!(AA, BB, GAM1, GAM2, GAM3, GAM4, GAM5, GAM6, JV)              ≈ Psi
    @test RiskAdjustedLinearizations.compute_Ψ!(AA, BB, GAM1, GAM2, GAM3, GAM4, GAM5, GAM6)                  ≈ Psi_det
    @test RiskAdjustedLinearizations.compute_Ψ!(AA, BB, GAM1, GAM2, GAM3, GAM4, GAM5, GAM6, zeros(size(JV))) ≈ Psi_det
end
