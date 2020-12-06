# This script actually solves the WachterDisasterRisk model with a risk-adjusted linearization
# and times the methods, if desired
using RiskAdjustedLinearizations
include("textbook_nk.jl")

# Set up
m_nk = TextbookNK()
m = textbook_nk(m_nk)

# Solve!
solve!(m; algorithm = :deterministic)
zdet = copy(m.z)
ydet = copy(m.y)
solve!(m; algorithm = :homotopy)
