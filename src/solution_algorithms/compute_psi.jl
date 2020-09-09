# TODO: save Matlab output from qzdecomp using WachterDisasterRisk
"""
```
function qzdecomp(Γ₁, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, JV = []; schur_fnct::Function = schur!) where {S <: Real}
```

solves for ``\\Psi_n`` in the quadratic matrix equation

``math
\\begin{aligned}
0 =  JV + \\Gamma_3 + \\Gamma_4\\Psi_n + (\\Gamma_5 + \\Gamma_6\\Psi_n) (\\Gamma_2 + \\Gamma_1 \\Psi_n).
\\end{aligned}
``

See the documentation of `RiskAdjustedLinearizations.jl` for details about what these matrices are.

### Inputs
All the required inputs must have type `AbstractMatrix{<: Number}`. The `JV` term is empty by default,
in which case `qzdecomp` assumes that `JV` is the zero matrix, which corresponds to the
case of the deterministic steady state.

### Keywords
- `schur_fnct::Function`: specifies which Generalized Schur algorithm is desired. By default,
    the implementation from BLAS is called, but the user may want to use the Generalized Schur algorithm
    from packages like `GenericLinearAlgebra.jl` and `GenericSchur.jl`.
"""
function qzdecomp(Γ₁::AbstractMatrix{S}, Γ₂::AbstractMatrix{S}, Γ₃::AbstractMatrix{S}, Γ₄::AbstractMatrix{S},
                  Γ₅::AbstractMatrix{S}, Γ₆::AbstractMatrix{S}, JV::AbstractMatrix{S} = Matrix{S}(undef, 0, 0);
                  schur_fnct::Function = schur!) where {S <: Number}

    if !isempty(JV)
        Γ₃ += JV
    end

    # Initialize left and right matrices for QZ decomposition (faster to initialize and then populate)
    Ny, Nz = size(Γ₆)
    Nzy    = Nz + Ny
    AA     = Matrix{Complex{S}}(undef, Nzy, Nzy)
    BB     = similar(AA)

    # Populate AA
    AA[1:Ny, 1:Nz]                 = Γ₅
    AA[1:Ny, (Nz + 1):end]         = Γ₆
    AA[(Ny + 1):end, 1:Nz]         = Diagonal(Ones{Complex{S}}(Nz))
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
