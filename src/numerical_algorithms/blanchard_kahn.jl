"""
```
blanchard_kahn(m::RiskAdjustedLinearization; deterministic::Bool = false, verbose::Symbol = :high)
```

checks the Blanchard-Kahn conditions for whether a first-order perturbation is saddle-path stable or not.

If `verbose` is `:low` or `:high`, a print statement will be shown if the Blanchard-Kahn conditions are satisfied.
"""
function blanchard_kahn(m::RiskAdjustedLinearization; deterministic::Bool = false, verbose::Symbol = :high)

    li = linearized_system(m)

    Nz = m.Nz
    Ny = m.Ny
    N_zy = m.Nz + m.Ny
    ztype = eltype(m.z)
    AA = Matrix{ztype}(undef, N_zy, N_zy)
    BB = similar(AA)

    # Populate AA
    AA[1:Ny, 1:Nz]                 = li[:Γ₅]
    AA[1:Ny, (Nz + 1):end]         = li[:Γ₆]
    AA[(Ny + 1):end, 1:Nz]         = Matrix{ztype}(I, m.Nz, m.Nz) # faster but makes allocations, unlike Diagonal(Ones{ztype}(Nz))
    AA[(Ny + 1):end, (Nz + 1):end] = Zeros{ztype}(m.Nz, m.Ny)

    # Populate BB
    BB[1:Ny, 1:Nz]                 = deterministic ? -li[:Γ₃] : -(li[:Γ₃] + li[:JV])
    BB[1:Ny, (Nz + 1):end]         = -li[:Γ₄]
    BB[(Ny + 1):end, 1:Nz]         =  li[:Γ₁]
    BB[(Ny + 1):end, (Nz + 1):end] =  li[:Γ₂]

    if count(abs.(eigen(AA, BB).values) .> 1) != m.Nz
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
