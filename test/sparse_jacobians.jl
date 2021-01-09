using RiskAdjustedLinearizations, SparseArrays, SparseDiffTools, Test
include(joinpath(dirname(@__FILE__), "..", "examples", "rbc_cc", "rbc_cc.jl"))
include(joinpath(dirname(@__FILE__), "..", "examples", "crw", "crw.jl"))

# Set up
n_strips = 3
m_rbc_cc = RBCCampbellCochraneHabits()
m_crw = CoeurdacierReyWinant()

# Test sparse Jacobians on RBC-CC
m_dense = rbc_cc(m_rbc_cc, n_strips)
z0 = copy(m_dense.z)
y0 = copy(m_dense.y)
Ψ0 = copy(m_dense.Ψ)

# m = rbc_cc(m_rbc_cc, n_strips; sparse_jacobian = [:μ, :ξ, :𝒱])
#=@testset "Construct a RiskAdjustedLinearization that exploits sparsity in Jacobians" begin
    @test isempty(m_dense.linearization.sparse_jac_caches)
    @test m.z ≈ z0
    @test m.y ≈ y0
    @test m.Ψ ≈ Ψ0
    for k in [:μz, :μy, :ξz, :ξy, :J𝒱]
        @test haskey(m.linearization.sparse_jac_caches, k)
        if k != :J𝒱
            @test issparse(m.linearization.sparse_jac_caches[k][:sparsity])
            @test isa(m.linearization.sparse_jac_caches[k][:colorvec], AbstractVector{Int})
        end
    end
    @test m.linearization.sparse_jac_caches[:J𝒱][:colorvec] == 1:2
    @test m.linearization.sparse_jac_caches[:J𝒱][:sparsity] == ones(size(m.Ψ))
end

@testset "Update a RiskAdjustedLinearization to exploit sparsity in Jacobians" begin
    update_sparsity_pattern!(m_dense, :𝒱)
    for k in [:μz, :μy, :ξz, :ξy]
        @test !haskey(m_dense.linearization.sparse_jac_caches, k)
    end
    @test m_dense.linearization.sparse_jac_caches[:J𝒱][:colorvec] == 1:2
    @test m_dense.linearization.sparse_jac_caches[:J𝒱][:sparsity] == ones(size(m.Ψ))

    update_sparsity_pattern!(m_dense, [:μ, :ξ, :𝒱])
    for k in [:μz, :μy, :ξz, :ξy]
        @test haskey(m_dense.linearization.sparse_jac_caches, k)
        @test issparse(m_dense.linearization.sparse_jac_caches[k][:sparsity])
        @test isa(m_dense.linearization.sparse_jac_caches[k][:colorvec], AbstractVector{Int})
    end
end=#

# Now provide the sparsity pattern and matrix coloring vector
# to update the Jacobians of objects
m_dense = rbc_cc(m_rbc_cc, n_strips) # recompute to get dense Jacobians again
solve!(m_dense, m_dense.z, m_dense.y; algorithm = :relaxation)
ztrue = copy(m_dense.z)
ytrue = copy(m_dense.y)
Ψtrue = copy(m_dense.Ψ)

sparsity = Dict{Symbol, SparseMatrixCSC{Float64, Int64}}()
colorvec = Dict{Symbol, Vector{Int64}}()
sparsity[:μz] = sparse(m_dense[:Γ₁])
sparsity[:μy] = sparse(m_dense[:Γ₂])
sparsity[:ξz] = sparse(m_dense[:Γ₃])
sparsity[:ξy] = sparse(m_dense[:Γ₄])
sparsity[:J𝒱] = sparse(m_dense[:JV])
for (k, v) in sparsity
    colorvec[k] = matrix_colors(v)
end

# Check updating dense Jacobians works
update_sparsity_pattern!(m_dense, [:μ, :ξ, :𝒱], ccgf = rbc_cc_ccgf)
solve!(m_dense, z0, y0; algorithm = :relaxation)
#=@test m_dense.z ≈ ztrue
@test m_dense.y ≈ ytrue
@test m_dense.Ψ ≈ Ψtrue

# Check updating sparse Jacobians w/new patterns works
update_sparsity_pattern!(m, :𝒱; sparsity = sparsity,
                         colorvec = colorvec)
solve!(m, z0, y0; algorithm = :relaxation)
@test m.z ≈ m_dense.z
@test m.y ≈ m_dense.y
@test m.Ψ ≈ m_dense.Ψ

update_sparsity_pattern!(m, [:μ, :ξ, :𝒱]; sparsity = sparsity,
                         colorvec = colorvec)
solve!(m, z0, y0; algorithm = :relaxation)
@test m.z ≈ m_dense.z
@test m.y ≈ m_dense.y
@test m.Ψ ≈ m_dense.Ψ

# Test sparse Jacobians on CRW
m_dense = crw(m_crw, n_strips)
z0 = copy(m_dense.z)
y0 = copy(m_dense.y)
Ψ0 = copy(m_dense.Ψ)

m = crw(m_crw, n_strips; sparse_jacobian = [:μ, :ξ, :𝒱])
@testset "Construct a RiskAdjustedLinearization that exploits sparsity in Jacobians" begin
    @test isempty(m_dense.linearization.sparse_jac_caches)
    @test m.z ≈ z0
    @test m.y ≈ y0
    @test m.Ψ ≈ Ψ0
    for k in [:μz, :μy, :ξz, :ξy, :J𝒱]
        @test haskey(m.linearization.sparse_jac_caches, k)
        if k != :J𝒱
            @test issparse(m.linearization.sparse_jac_caches[k][:sparsity])
            @test isa(m.linearization.sparse_jac_caches[k][:colorvec], AbstractVector{Int})
        end
    end
    @test m.linearization.sparse_jac_caches[:J𝒱][:colorvec] == 1:2
    @test m.linearization.sparse_jac_caches[:J𝒱][:sparsity] == ones(size(m.Ψ))
end

@testset "Update a RiskAdjustedLinearization to exploit sparsity in Jacobians" begin
    update_sparsity_pattern!(m_dense, :𝒱)
    for k in [:μz, :μy, :ξz, :ξy]
        @test !haskey(m_dense.linearization.sparse_jac_caches, k)
    end
    @test m_dense.linearization.sparse_jac_caches[:J𝒱][:colorvec] == 1:2
    @test m_dense.linearization.sparse_jac_caches[:J𝒱][:sparsity] == ones(size(m.Ψ))

    update_sparsity_pattern!(m_dense, [:μ, :ξ, :𝒱])
    for k in [:μz, :μy, :ξz, :ξy]
        @test haskey(m_dense.linearization.sparse_jac_caches, k)
        @test issparse(m_dense.linearization.sparse_jac_caches[k][:sparsity])
        @test isa(m_dense.linearization.sparse_jac_caches[k][:colorvec], AbstractVector{Int})
    end
end

# Now provide the sparsity pattern and matrix coloring vector
# to update the Jacobians of objects
m_dense = crw(m_crw, n_strips) # recompute to get dense Jacobians again
solve!(m_dense, m_dense.z, m_dense.y; algorithm = :relaxation)
ztrue = copy(m_dense.z)
ytrue = copy(m_dense.y)
Ψtrue = copy(m_dense.Ψ)

sparsity = Dict{Symbol, SparseMatrixCSC{Float64, Int64}}()
colorvec = Dict{Symbol, Vector{Int64}}()
sparsity[:μz] = sparse(m_dense[:Γ₁])
sparsity[:μy] = sparse(m_dense[:Γ₂])
sparsity[:ξz] = sparse(m_dense[:Γ₃])
sparsity[:ξy] = sparse(m_dense[:Γ₄])
sparsity[:J𝒱] = sparse(m_dense[:JV])
for (k, v) in sparsity
    colorvec[k] = matrix_colors(v)
end

# Check updating dense Jacobians works
update_sparsity_pattern!(m_dense, [:μ, :ξ, :𝒱])
solve!(m_dense, z0, y0; algorithm = :relaxation)
@test m_dense.z ≈ ztrue
@test m_dense.y ≈ ytrue
@test m_dense.Ψ ≈ Ψtrue

# Check updating sparse Jacobians w/new patterns works
update_sparsity_pattern!(m, :𝒱; sparsity = sparsity,
                         colorvec = colorvec)
solve!(m, z0, y0; algorithm = :relaxation)
@test m.z ≈ m_dense.z
@test m.y ≈ m_dense.y
@test m.Ψ ≈ m_dense.Ψ

update_sparsity_pattern!(m, [:μ, :ξ, :𝒱]; sparsity = sparsity,
                         colorvec = colorvec)
solve!(m, z0, y0; algorithm = :relaxation)
@test m.z ≈ m_dense.z
@test m.y ≈ m_dense.y
@test m.Ψ ≈ m_dense.Ψ
=#
