include("risk_adjusted_linearization.jl")
include("wachter.jl")
using JLD2, Test

m = WachterDisasterRisk()

### In-place RiskAdjustedLinearization

## Deterministic steady state
detout = JLD2.jldopen("det_ss_output.jld2", "r")
z = vec(detout["z"])
y = vec(detout["y"])
Ψ = zeros(eltype(y), length(y), length(z))
ral = inplace_wachter_disaster_risk(m)

# Check outputs
update!(ral, z, y, Ψ)
nl = nonlinear_system(ral)
li = linearized_system(ral)
@testset "Evaluate WachterDisasterRisk in-place RiskAdjustedLinearization at deterministic steady state" begin
    @test nl.μ_sss ≈ detout["MU"]
    @test nl.Λ.cache ==  detout["LAM"]
    @test nl.Σ.cache.du ≈ detout["SIG"]
    @test nl.ξ_sss ≈ detout["XI"]
    @test nl.𝒱_sss ≈ detout["V"]
    @test li.Γ₁ ≈ detout["GAM1"]
    @test li.Γ₂ ≈ detout["GAM2"]
    @test li.Γ₃ ≈ detout["GAM3"]
    @test li.Γ₄ ≈ detout["GAM4"]
    @test li.Γ₅ ≈ detout["GAM5"]
    @test li.Γ₆ ≈ detout["GAM6"]
    @test li.JV ≈ detout["JV"]
end

## Stochastic steady state
sssout = JLD2.jldopen("iterative_sss_output.jld2", "r")
z = vec(sssout["z"])
y = vec(sssout["y"])
Ψ = sssout["Psi"]

# Check outputs
update!(ral, z, y, Ψ)
nl = nonlinear_system(ral)
li = linearized_system(ral)
@testset "Evaluate WachterDisasterRisk in-place RiskAdjustedLinearization at stochastic steady state" begin
    @test nl.μ_sss ≈ sssout["MU"]
    @test nl.Λ.cache ==  sssout["LAM"]
    @test nl.Σ.cache.du ≈ sssout["SIG"]
    @test nl.ξ_sss ≈ sssout["XI"]
    @test nl.𝒱_sss ≈ sssout["V"]
    @test li.Γ₁ ≈ sssout["GAM1"]
    @test li.Γ₂ ≈ sssout["GAM2"]
    @test li.Γ₃ ≈ sssout["GAM3"]
    @test li.Γ₄ ≈ sssout["GAM4"]
    @test li.Γ₅ ≈ sssout["GAM5"]
    @test li.Γ₆ ≈ sssout["GAM6"]
    @test li.JV ≈ sssout["JV"]
end

### Out-of-place RiskAdjustedLinearization

## Deterministic steady state
detout = JLD2.jldopen("det_ss_output.jld2", "r")
z = vec(detout["z"])
y = vec(detout["y"])
Ψ = zeros(eltype(y), length(y), length(z))
ral = outofplace_wachter_disaster_risk(m)

# Check outputs
update!(ral, z, y, Ψ)
nl = nonlinear_system(ral)
li = linearized_system(ral)
@testset "Evaluate WachterDisasterRisk out-of-place RiskAdjustedLinearization at deterministic steady state" begin
    @test nl.μ_sss ≈ detout["MU"]
    @test nl.Λ.cache ==  detout["LAM"]
    @test nl.Σ.cache.du ≈ detout["SIG"]
    @test nl.ξ_sss ≈ detout["XI"]
    @test nl.𝒱_sss ≈ detout["V"]
    @test li.Γ₁ ≈ detout["GAM1"]
    @test li.Γ₂ ≈ detout["GAM2"]
    @test li.Γ₃ ≈ detout["GAM3"]
    @test li.Γ₄ ≈ detout["GAM4"]
    @test li.Γ₅ ≈ detout["GAM5"]
    @test li.Γ₆ ≈ detout["GAM6"]
    @test li.JV ≈ detout["JV"]
end

## Stochastic steady state
sssout = JLD2.jldopen("iterative_sss_output.jld2", "r")
z = vec(sssout["z"])
y = vec(sssout["y"])
Ψ = sssout["Psi"]

# Check outputs
update!(ral, z, y, Ψ)
nl = nonlinear_system(ral)
li = linearized_system(ral)
@testset "Evaluate WachterDisasterRisk out-of-place RiskAdjustedLinearization at stochastic steady state" begin
    @test nl.μ_sss ≈ sssout["MU"]
    @test nl.Λ.cache ==  sssout["LAM"]
    @test nl.Σ.cache.du ≈ sssout["SIG"]
    @test nl.ξ_sss ≈ sssout["XI"]
    @test nl.𝒱_sss ≈ sssout["V"]
    @test li.Γ₁ ≈ sssout["GAM1"]
    @test li.Γ₂ ≈ sssout["GAM2"]
    @test li.Γ₃ ≈ sssout["GAM3"]
    @test li.Γ₄ ≈ sssout["GAM4"]
    @test li.Γ₅ ≈ sssout["GAM5"]
    @test li.Γ₆ ≈ sssout["GAM6"]
    @test li.JV ≈ sssout["JV"]
end
