# Functions for checking blanchard_kahn conditions
function blanchard_kahn(m::RiskAdjustedLinearization; verbose::Symbol = :high)
    Γ₅ = m.linearization.Γ₅
    Γ₆ = m.linearization.Γ₆
    A = [Γ₅ Γ₆; Matrix{eltype(Γ₅)}(I, m.Nz, m.Nz) zeros(m.Ny, m.Nz)]
    B = [(-Γ₃(m) - JV(m)) (-Γ₄(m)); Γ₁(m) Γ₂(m)]

    if count(abs.(eigen(A, B).values) .> 1) != m.Nz
        BlanchardKahnError("First-order perturbation around stochastic steady state is not saddle-path stable")
    else
        if verbose in [:low, :high]
            println("Blanchard-Kahn conditions for a unique locally bounded stochastic steady-state perturbation are satisfied")
        end

        return true
    end
end

mutable struct BlanchardKahnError <: Exception
    msg::String
end
BlanchardKahnError() = BlanchardKahnError("First-order perturbation is not saddle-path stable")
Base.showerror(io::IO, ex::BlanchardKahnError) = print(io, ex.msg)
