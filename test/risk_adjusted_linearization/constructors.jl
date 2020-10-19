using JLD2, Test, RiskAdjustedLinearizations

# Use Wachter model with Disaster Risk to assess the constructors of a RiskAdjustedLinearization type
# for in-place and out-of-place functions
include(joinpath(dirname(@__FILE__), "../../examples/wachter_disaster_risk/wachter.jl"))
m = WachterDisasterRisk()

### In-place RiskAdjustedLinearization

## Deterministic steady state
detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/det_ss_output.jld2"), "r")
z = vec(detout["z"])
y = vec(detout["y"])
Ψ = zeros(eltype(y), length(y), length(z))
ral = inplace_wachter_disaster_risk(m)

# Check outputs
update!(ral, z, y, Ψ)
nl = nonlinear_system(ral)
li = linearized_system(ral)
@testset "Evaluate WachterDisasterRisk in-place RiskAdjustedLinearization at deterministic steady state" begin
    @test nl[:μ_sss] ≈ detout["MU"]
    @test nl[:Λ.cache] ==  detout["LAM"]
    @test nl[:Σ_sss] ≈ detout["SIG"]
    @test nl[:ξ_sss] ≈ detout["XI"]
    @test nl[:𝒱_sss] ≈ detout["V"]
    @test li[:Γ₁]] ≈ detout["GAM1"]
    @test li[:Γ₂] ≈ detout["GAM2"]
    @test li[:Γ₃] ≈ detout["GAM3"]
    @test li[:Γ₄] ≈ detout["GAM4"]
    @test li[:Γ₅] ≈ detout["GAM5"]
    @test li[:Γ₆] ≈ detout["GAM6"]
    @test li[:JV] ≈ detout["JV"]
    @test ral[:μ_sss] ≈ detout["MU"]
    @test ral[:Λ.cache] ==  detout["LAM"]
    @test ral[:Σ_sss] ≈ detout["SIG"]
    @test ral[:ξ_sss] ≈ detout["XI"]
    @test ral[:𝒱_sss] ≈ detout["V"]
    @test ral[:Γ₁]] ≈ detout["GAM1"]
    @test ral[:Γ₂] ≈ detout["GAM2"]
    @test ral[:Γ₃] ≈ detout["GAM3"]
    @test ral[:Γ₄] ≈ detout["GAM4"]
    @test ral[:Γ₅] ≈ detout["GAM5"]
    @test ral[:Γ₆] ≈ detout["GAM6"]
    @test ral[:JV] ≈ detout["JV"]
end

## Stochastic steady state
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/iterative_sss_output.jld2"), "r")
z = vec(sssout["z"])
y = vec(sssout["y"])
Ψ = sssout["Psi"]

# Check outputs
update!(ral, z, y, Ψ)
nl = nonlinear_system(ral)
li = linearized_system(ral)
@testset "Evaluate WachterDisasterRisk in-place RiskAdjustedLinearization at stochastic steady state" begin
    @test nl[:μ_sss] ≈ sssout["MU"]
    @test nl[:Λ.cache] ==  sssout["LAM"]
    @test nl[:Σ_sss] ≈ sssout["SIG"]
    @test nl[:ξ_sss] ≈ sssout["XI"]
    @test nl[:𝒱_sss] ≈ sssout["V"]
    @test li[:Γ₁] ≈ sssout["GAM1"]
    @test li[:Γ₂] ≈ sssout["GAM2"]
    @test li[:Γ₃] ≈ sssout["GAM3"]
    @test li[:Γ₄] ≈ sssout["GAM4"]
    @test li[:Γ₅] ≈ sssout["GAM5"]
    @test li[:Γ₆] ≈ sssout["GAM6"]
    @test li[:JV] ≈ sssout["JV"]
    @test ral[:μ_sss] ≈ sssout["MU"]
    @test ral[:Λ.cache] ==  sssout["LAM"]
    @test ral[:Σ_sss] ≈ sssout["SIG"]
    @test ral[:ξ_sss] ≈ sssout["XI"]
    @test ral[:𝒱_sss] ≈ sssout["V"]
    @test ral[:Γ₁] ≈ sssout["GAM1"]
    @test ral[:Γ₂] ≈ sssout["GAM2"]
    @test ral[:Γ₃] ≈ sssout["GAM3"]
    @test ral[:Γ₄] ≈ sssout["GAM4"]
    @test ral[:Γ₅] ≈ sssout["GAM5"]
    @test ral[:Γ₆] ≈ sssout["GAM6"]
    @test ral[:JV] ≈ sssout["JV"]
end

### Out-of-place RiskAdjustedLinearization

## Deterministic steady state
detout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/det_ss_output.jld2"), "r")
z = vec(detout["z"])
y = vec(detout["y"])
Ψ = zeros(eltype(y), length(y), length(z))
ral = outofplace_wachter_disaster_risk(m)

# Check outputs
update!(ral, z, y, Ψ)
nl = nonlinear_system(ral)
li = linearized_system(ral)
@testset "Evaluate WachterDisasterRisk out-of-place RiskAdjustedLinearization at deterministic steady state" begin
    @test nl[:μ_sss] ≈ detout["MU"]
    @test nl[:Λ_sss] == detout["LAM"]
    @test isnothing(nl.Σ.cache])
    @test nl.Σ(z) ≈ detout["SIG"]
    @test nl[:ξ_sss] ≈ detout["XI"]
    @test nl[:𝒱_sss] ≈ detout["V"]
    @test li[:Γ₁] ≈ detout["GAM1"]
    @test li[:Γ₂] ≈ detout["GAM2"]
    @test li[:Γ₃] ≈ detout["GAM3"]
    @test li[:Γ₄] ≈ detout["GAM4"]
    @test li[:Γ₅] ≈ detout["GAM5"]
    @test li[:Γ₆] ≈ detout["GAM6"]
    @test li[:JV] ≈ detout["JV"]
end

## Stochastic steady state
sssout = JLD2.jldopen(joinpath(dirname(@__FILE__), "../reference/iterative_sss_output.jld2"), "r")
z = vec(sssout["z"])
y = vec(sssout["y"])
Ψ = sssout["Psi"]

# Check outputs
update!(ral, z, y, Ψ)
nl = nonlinear_system(ral)
li = linearized_system(ral)
@testset "Evaluate WachterDisasterRisk out-of-place RiskAdjustedLinearization at stochastic steady state" begin
    @test nl[:μ_sss] ≈ sssout["MU"]
    @test nl[:Λ_sss] ==  sssout["LAM"]
    @test isnothing(nl.Σ.cache)
    @test nl.Σ(z) ≈ detout["SIG"]
    @test nl[:ξ_sss] ≈ sssout["XI"]
    @test nl[:𝒱_sss] ≈ sssout["V"]
    @test li[:Γ₁] ≈ sssout["GAM1"]
    @test li[:Γ₂] ≈ sssout["GAM2"]
    @test li[:Γ₃] ≈ sssout["GAM3"]
    @test li[:Γ₄] ≈ sssout["GAM4"]
    @test li[:Γ₅] ≈ sssout["GAM5"]
    @test li[:Γ₆] ≈ sssout["GAM6"]
    @test li[:JV] ≈ sssout["JV"]
end

nothing
