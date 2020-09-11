# [Example](@id example)

This example shows how to calculate the risk-adjusted linearization of the
discrete-time version of the [Wachter (2013)](http://finance.wharton.upenn.edu/~jwachter/research/Wachter2013jf.pdf)
model with disaster-risk. You can run this example using the script [examples/wachter_disaster_risk/example_wachter.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations/tree/master/examples/wachter_disaster_risk/example_wachter.jl).
For the equivalent code in MATLAB provided by Lopez et al., see [here](https://github.com/fvazquezgrande/gen_affine/blob/master/examples/wac_disaster/genaffine_ezdis.m).

## Create a `RiskAdjustedLinearization`


### Define Nonlinear System

The user generally needs to define
- ``\\mu``: expected state transition function
- ``\\xi`` nonlinear terms of the expectational equations
- ccgf: conditional cumulant generating function of the exogenous shocks
- ``\\Lambda``: function or matrix mapping endogenous risk into state transition equations
- ``\\Sigma``: function or matrix mapping exogenous risk into state transition equations
- ``\\Gamma_5```: coefficient matrix on one-period ahead expectation of state variables
- ``\\Gamma_6``: coefficient matrix on one-period ahead expectation of jump variables
-

The quantities ``\\mu``, ``\\xi``, and ccgf are always functions. The quantities ``\\Lambda`` and ``\\Sigma`` can
either be functions or matrices. For example, in endowment economies like Wachter (2013), ``\\Lambda`` is
the zero matrix since there is no endogenous risk. In other applications, ``\\Sigma`` may not be state-dependent
and thus a constant matrix. The last two quantities ``\\Gamma_5`` and ``\\Gamma_6`` are always matrices.

In addition, you need to define initial guesses for the coefficients `z, y, Ψ` and specify the number of exogenous shocks `Nε`.
The initial guesses can be undefined if you don't want to use actual numbers yet, but
you will eventually need to provide guesses in order for the nonlinear solvers to work in
the numerical algorithms.


### Instantiate the object
Once you have the required quantities, simply call

```
ral = RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nε)
```

### Example
The following code presents a function that defines the desired functions and matrices, given
the parameters for the model in Wachter (2013), and returns a `RiskAdjustedLinearization` object.
The code is from this script [examples/wachter_disaster_risk/wachter.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations/tree/master/examples/wachter_disaster_risk/wachter.jl), which has examples for both in-place and out-of-place functions.


```
function inplace_wachter_disaster_risk(m::WachterDisasterRisk{T}) where {T <: Real}
    @unpack μₐ, σₐ, ν, δ, ρₚ, pp, ϕₚ, ρ, γ, β = m

    @assert ρ != 1. # Forcing ρ to be non-unit for this example

    S  = OrderedDict{Symbol, Int}(:p => 1,  :εc => 2, :εξ => 3) # State variables
    J  = OrderedDict{Symbol, Int}(:vc => 1, :xc => 2, :rf => 3) # Jump variables
    SH = OrderedDict{Symbol, Int}(:εₚ => 1, :εc => 2, :εξ => 3) # Exogenous shocks
    Nz = length(S)
    Ny = length(J)
    Nε = length(SH)

    function μ(F, z, y)
        F_type    = eltype(F)
        F[S[:p]]  = (1 - ρₚ) * pp + ρₚ * z[S[:p]]
        F[S[:εc]] = zero(F_type)
        F[S[:εξ]] = zero(F_type)
    end

    function ξ(F, z, y)
        F[J[:vc]] = log(β) - γ * μₐ + γ * ν * z[S[:p]] - (ρ - γ) * y[J[:xc]] + y[J[:rf]]
        F[J[:xc]] = log(1. - β + β * exp((1. - ρ) * y[J[:xc]])) - (1. - ρ) * y[J[:vc]]
        F[J[:rf]] = (1. - γ) * (μₐ - ν * z[S[:p]] - y[J[:xc]])
    end

    Λ = zeros(T, Nz, Ny)

    function Σ(F, z)
        F_type = eltype(F)
        F[SH[:εₚ], SH[:εₚ]] = sqrt(z[S[:p]]) * ϕₚ * σₐ
        F[SH[:εc], SH[:εc]] = one(F_type)
        F[SH[:εξ], SH[:εξ]] = one(F_type)
    end

    function ccgf(F, α, z)
        F .= .5 .* α[:, 1].^2 + .5 * α[:, 2].^2 + (exp.(α[:, 3] + α[:, 3].^2 .* δ^2 ./ 2.) .- 1. - α[:, 3]) * z[S[:p]]
    end

    Γ₅ = zeros(T, Ny, Nz)
    Γ₅[J[:vc], S[:εc]] = (-γ * σₐ)
    Γ₅[J[:vc], S[:εξ]] = (γ * ν)
    Γ₅[J[:rf], S[:εc]] = (1. - γ) * σₐ
    Γ₅[J[:rf], S[:εξ]] = -(1. - γ) * ν

    Γ₆ = zeros(T, Ny, Ny)
    Γ₆[J[:vc], J[:vc]] = (ρ - γ)
    Γ₆[J[:rf], J[:vc]] = (1. - γ)

    z = [pp, 0., 0.]
    xc_sss = log((1. - β) / (exp((1. - ρ) * (ν * pp - μₐ)) - β)) / (1. - ρ)
    vc_sss = xc_sss + ν * pp - μₐ
    y = [vc_sss, xc_sss, -log(β) + γ * (μₐ - ν * pp) - (ρ - γ) * (vc_sss - xc_sss)]
    Ψ = zeros(T, Ny, Nz)
    return RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nε)
end
```

## Solve using a Newton-type Numerical Algorithm
To solve the model using the relaxation algorithm, just call

```
solve!(ral; algorithm = :relaxation)
```

This form of `solve!` uses the coefficients in `ral` as initial guesses. To specify
other initial guesses, call

```
solve!(ral, z0, y0, Ψ0; algorithm = :relaxation)
```

If you don't have a guess for ``\\Psi``, then you can just provide guesses for ``z`` and ``y``:

```
solve!(ral, z0, y0; algorithm = :relaxation)
```

In this case, we calculate the deterministic steady state first using ``z`` and ``y``;
back out the implied ``\\Psi``; and then proceed with the relaxation algorithm using
the deterministic steady state as the initial guess.
