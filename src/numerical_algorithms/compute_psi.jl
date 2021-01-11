# TODO: Check if computing the Blanchard Kahn conditions every time within compute_Ψ
#       ends up being faster than doing the computation once but saving on the
#       cost of calling eigen.

"""
```
compute_Ψ(Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, JV = []; schur_fnct = schur!)
compute_Ψ(m::RALLinearizedSystem; zero_entropy_jacobian = false, schur_fnct = schur!)
compute_Ψ(m::RiskAdjustedLinearization; zero_entropy_jacobian = false, schur_fnct = schur!)
```

solves via QZ decomposition for ``\\Psi_n`` in the quadratic matrix equation

``math
\\begin{aligned}
0 =  JV + \\Gamma_3 + \\Gamma_4\\Psi_n + (\\Gamma_5 + \\Gamma_6\\Psi_n) (\\Gamma_2 + \\Gamma_1 \\Psi_n).
\\end{aligned}
``

See the documentation of `RiskAdjustedLinearizations.jl` for details about what these matrices are.

### Inputs
For the first method, all the required inputs must have type `AbstractMatrix{<: Number}`. The `JV` term is empty by default,
in which case `qzdecomp` assumes that `JV` is the zero matrix, which corresponds to the
case of the deterministic steady state.

The second and third methods are wrappers for the first method.

Internally, there are in-place versions (`compute_Ψ!`) to avoid allocations when
`compute_Ψ!` is called repeatedly by one of the numerical algorithms (e.g. `relaxation!`).

### Keywords
- `schur_fnct::Function`: specifies which Generalized Schur algorithm is desired. By default,
    the implementation from BLAS is called, but the user may want to use the Generalized Schur algorithm
    from packages like `GenericLinearAlgebra.jl` and `GenericSchur.jl`.
- `zero_entropy_jacobian::Bool`: if true, then we assume the Jacobian of the entropy is all zeros (i.e. in the deterministic steady state).
    This keyword allows the user to avoid adding zeros unnecessarily and/or let the cached entropy Jacobian remain undefined.
"""
function compute_Ψ(Γ₁::AbstractMatrix{S}, Γ₂::AbstractMatrix{S}, Γ₃::AbstractMatrix{S}, Γ₄::AbstractMatrix{S},
                   Γ₅::AbstractMatrix{S}, Γ₆::AbstractMatrix{S}, JV::AbstractMatrix{S} = Matrix{S}(undef, 0, 0);
                   schur_fnct::Function = schur!) where {S <: Number}

    Nzy = sum(size(Γ₅))
    AA  = Matrix{Complex{S}}(undef, Nzy, Nzy)
    BB  = similar(AA)

    return compute_Ψ!(AA, BB, Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, JV; schur_fnct = schur_fnct)
end

function compute_Ψ(m::RALLinearizedSystem; zero_entropy_jacobian::Bool = false, schur_fnct::Function = schur!) where {S <: Number}
    if zero_entropy_jacobian
        return compute_Ψ(m[:Γ₁], m[:Γ₂], m[:Γ₃], m[:Γ₄], m[:Γ₅], m[:Γ₆]; schur_fnct = schur_fnct)
    else
        return compute_Ψ(m[:Γ₁], m[:Γ₂], m[:Γ₃], m[:Γ₄], m[:Γ₅], m[:Γ₆], m[:JV]; schur_fnct = schur_fnct)
    end
end

@inline function compute_Ψ(m::RiskAdjustedLinearization; zero_entropy_jacobian::Bool = false, schur_fnct::Function = schur!) where {S <: Number}
    return compute_Ψ(m.linearization; zero_entropy_jacobian = zero_entropy_jacobian, schur_fnct = schur_fnct)
end

function compute_Ψ!(AA::AbstractMatrix{Complex{S}}, BB::AbstractMatrix{Complex{S}},
                    Γ₁::AbstractMatrix{S}, Γ₂::AbstractMatrix{S}, Γ₃::AbstractMatrix{S}, Γ₄::AbstractMatrix{S},
                    Γ₅::AbstractMatrix{S}, Γ₆::AbstractMatrix{S}, JV::AbstractMatrix{S} = Matrix{S}(undef, 0, 0);
                    schur_fnct::Function = schur!) where {S <: Number}

    if !isempty(JV)
        Γ₃ += JV
    end
    Ny, Nz = size(Γ₅)

    # Populate AA
    AA[1:Ny, 1:Nz]                 = Γ₅
    AA[1:Ny, (Nz + 1):end]         = Γ₆
    AA[(Ny + 1):end, 1:Nz]         = Matrix{Complex{S}}(I, Nz, Nz) # faster but makes allocations, unlike Diagonal(Ones{Complex{S}}(Nz))
    AA[(Ny + 1):end, (Nz + 1):end] = Zeros{Complex{S}}(Nz, Ny)

    # Populate BB
    BB[1:Ny, 1:Nz]                 = -Γ₃
    BB[1:Ny, (Nz + 1):end]         = -Γ₄
    BB[(Ny + 1):end, 1:Nz]         =  Γ₁
    BB[(Ny + 1):end, (Nz + 1):end] =  Γ₂

    # Compute QZ and back out Ψ
    schurfact = schur_fnct(AA, BB)
	ordschur!(schurfact, [abs(αᵢ) >= abs(βᵢ) for (αᵢ, βᵢ) in zip(schurfact.α, schurfact.β)]) # eigenvalues = schurfact.β / schurfact.α
    return real(schurfact.Z[Nz + 1:end, 1:Nz] / schurfact.Z[1:Nz, 1:Nz])
end

function compute_Ψ!(AA::AbstractMatrix{Complex{S}}, BB::AbstractMatrix{Complex{S}},
                    m::RALLinearizedSystem; zero_entropy_jacobian::Bool = false, schur_fnct::Function = schur!) where {S <: Number}
    if zero_entropy_jacobian
        return compute_Ψ!(AA, BB, m[:Γ₁], m[:Γ₂], m[:Γ₃], m[:Γ₄], m[:Γ₅], m[:Γ₆]; schur_fnct = schur_fnct)
    else
        return compute_Ψ!(AA, BB, m[:Γ₁], m[:Γ₂], m[:Γ₃], m[:Γ₄], m[:Γ₅], m[:Γ₆], m[:JV]; schur_fnct = schur_fnct)
    end
end

@inline function compute_Ψ!(AA::AbstractMatrix{Complex{S}}, BB::AbstractMatrix{Complex{S}},
                            m::RiskAdjustedLinearization; zero_entropy_jacobian::Bool = false, schur_fnct::Function = schur!) where {S <: Number}
    return compute_Ψ!(AA, BB, m.linearization; zero_entropy_jacobian = zero_entropy_jacobian, schur_fnct = schur_fnct)
end
