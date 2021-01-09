"""
```
blanchard_kahn(m::RiskAdjustedLinearization; deterministic::Bool = false, verbose::Symbol = :high)
```

checks the Blanchard-Kahn conditions for whether a first-order perturbation is saddle-path stable or not.

If `verbose` is `:low` or `:high`, a print statement will be shown if the Blanchard-Kahn conditions are satisfied.
"""
function blanchard_kahn(m::RiskAdjustedLinearization; deterministic::Bool = false, verbose::Symbol = :high)

    li = linearized_system(m)

    Γ₅ = issparse(li[:Γ₅]) ? Array(li[:Γ₅]) : li[:Γ₅]
    Γ₆ = issparse(li[:Γ₆]) ? Array(li[:Γ₆]) : li[:Γ₆]

    if isempty(li.sparse_jac_caches)
        A = [Γ₅ Γ₆; Matrix{eltype(Γ₅)}(I, m.Nz, m.Nz) Zeros{eltype(Γ₅)}(m.Nz, m.Ny)]
        B = [(-li[:Γ₃] - li[:JV]) (-li[:Γ₄]); li[:Γ₁] li[:Γ₂]]
    else
        Γ₁ = haskey(li.sparse_jac_caches, :μz) ? Array(li[:Γ₁]) : li[:Γ₁]
        Γ₂ = haskey(li.sparse_jac_caches, :μy) ? Array(li[:Γ₂]) : li[:Γ₂]
        Γ₃ = haskey(li.sparse_jac_caches, :ξz) ? Array(li[:Γ₃]) : li[:Γ₃]
        Γ₄ = haskey(li.sparse_jac_caches, :ξy) ? Array(li[:Γ₄]) : li[:Γ₄]
        JV = haskey(li.sparse_jac_caches, :J𝒱) ? Array(li[:JV]) : li[:JV]

        A = [Γ₅ Γ₆; Matrix{eltype(Γ₅)}(I, m.Nz, m.Nz) Zeros{eltype(Γ₅)}(m.Nz, m.Ny)]
        B = [(-Γ₃ - JV) (-Γ₄); Γ₁ Γ₂]
    end

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
