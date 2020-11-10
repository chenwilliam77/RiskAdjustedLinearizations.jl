using RiskAdjustedLinearizations, Test
include(joinpath(dirname(@__FILE__), "..", "..", "examples", "crw", "crw.jl"))

# Solve model
m_crw = CoeurdacierReyWinant()
m = crw(m_crw)
solve!(m, m.z, m.y, m.Ψ; algorithm = :homotopy)

function crw_logSDFxR(m, zₜ, εₜ₊₁, cₜ)

end
