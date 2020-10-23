"""
```
blanchard_kahn(m::RiskAdjustedLinearization; deterministic::Bool = false, verbose::Symbol = :high)
```

checks the Blanchard-Kahn conditions for whether a first-order perturbation is saddle-path stable or not.

If `verbose` is `:low` or `:high`, a print statement will be shown if the Blanchard-Kahn conditions are satisfied.
"""
function blanchard_kahn(m::RiskAdjustedLinearization; deterministic::Bool = false, verbose::Symbol = :high)

    A = [m[:Γ₅] m[:Γ₆]; Matrix{eltype(m[:Γ₅])}(I, m.Nz, m.Nz) Zeros{eltype(m[:Γ₅])}(m.Nz, m.Ny)]
    B = [(-m[:Γ₃] - m[:JV]) (-m[:Γ₄]); m[:Γ₁] m[:Γ₂]]

    if count(abs.(eigen(A, B).values) .> 1) != m.Nz
        if deterministic
            throw(BlanchardKahnError("First-order perturbation around deterministic steady state is not saddle-path stable"))
        else
            throw(BlanchardKahnError("First-order perturbation around stochastic steady state is not saddle-path stable"))
        end
    else
        if verbose in [:low, :high]
            if deterministic
                println("Blanchard-Kahn conditions for a unique locally bounded deterministic " *
                        "steady-state perturbation are satisfied")
            else
                println("Blanchard-Kahn conditions for a unique locally bounded stochastic " *
                        "steady-state perturbation are satisfied")
            end
        end

        return true
    end
end

mutable struct BlanchardKahnError <: Exception
    msg::String
end
BlanchardKahnError() = BlanchardKahnError("First-order perturbation is not saddle-path stable")
Base.showerror(io::IO, ex::BlanchardKahnError) = print(io, ex.msg)
