# [Risk-Adjusted Linearizations](@id risk-adjusted linearization)

## Theory

### Nonlinear Model
Most dynamic economic models can be formulated as the system of nonlinear equations

```math
\begin{aligned}
    z_{t + 1} & = \mu(z_t, y_t) + \Lambda(z_t)(y_{t + 1} - \mathbb{E}_t y_{t + 1}) + \Sigma(z_t) \varepsilon_{t + 1},\\
    0 & = \log\mathbb{E}_t[\exp(\xi(z_t, y_t) + \Gamma_5 z_{t + 1} + \Gamma_6 y_{t + 1})].
\end{aligned}
```

The vectors ``z_t\in \mathbb{R}^{n_z}`` and ``y_t \in \mathbb{R}^{n_y}`` are the state and jump variables, respectively.
The first vector equation comprise the model's expectational equations, which are typically
the first-order conditions for the jump variables from agents' optimization problem.
The second vector equation comprise the transition equations of the state variables. The exogenous shocks
``\varepsilon\in\mathbb{R}^{n_\varepsilon}`` form a martingale difference sequence whose distribution
is described by the differentiable, conditional cumulant generating function (ccgf)

```math
\begin{aligned}
\kappa[\alpha(z_t) \mid z_t] = \log\mathbb{E}_t[\exp(\alpha(z_t)' \varepsilon_{t + 1})],\quad \text{ for any differentiable map }\alpha:\mathbb{R}^{n_z}\rightarrow\mathbb{R}^{n_\varepsilon}.
\end{aligned}
```

The functions
```math
\begin{aligned}
\xi:\mathbb{R}^{2n_y + 2n_z}\rightarrow \mathbb{R}^{n_y},& \quad \mu:\mathbb{R}^{n_y + n_z}\rightarrow \mathbb{R}^{n_z},\\
\Lambda:\mathbb{R}^{n_z} \rightarrow \mathbb{R}^{n_z \times n_y}, & \quad \Sigma:\mathbb{R}^{n_z}\ \rightarrow \mathbb{R}^{n_z\times n_\varepsilon},
\end{aligned}
```
are differentiable. The first two functions characterize the effects of time ``t`` variables on the expectational and
state transition equations. The function ``\Lambda`` characterizes heteroskedastic endogenous risk that depends on
innovations in jump variables while the function ``\Sigma`` characterizes exogenous risk.

### Risk-Adjusted Linearizations by Affine Approximation

Many economic models are typically solved by perturbation around the deterministic steady state. To break certainty equivalence so that
asset pricing is meaningful, these perturbations need to be at least third order. However, even third-order perturbations
can poorly approximate the true global solution. A key problem is that the economy may not spend much time near the
deterministic steady state, so a perturbation around this point will be inaccurate.


Instead of perturbing the model's nonlinear equations around the deterministic steady state, we could perturb around the
stochastic or "risky" steady state. This point is better for a perturbation because the economy will spend a
large amount of time near the stochastic steady state. [Lopez et al. (2018)](https://ideas.repec.org/p/bfr/banfra/702.html)
show that an affine approximation of the model's nonlinear equation is equivalent to a linearization around the
stochastic steady state. Further, they confirm that in practice this "risk-adjusted" linearization well approximate
global solutions of canonical economic models and outperforms perturbations around the deterministic steady state.

The affine approximation of an dynamic economic model is
```math
\begin{aligned}
    \mathbb{E}[z_{t + 1}] & = \mu(z, y) + \Gamma_1(z_t - z) + \Gamma_2(y_t - y),\\
    0                      & = \xi(z, y) + \Gamma_3(z_t - z) + \Gamma_4(y_t - y) + \Gamma_5 \mathbb{E}_t z_{t + 1} + \Gamma_6 \mathbb{E}_t y_{t + 1} + \mathcal{V}(z) + J\mathcal{V}(z)(z_t  - z),
\end{aligned}
```

where ``\Gamma_1, \Gamma_2`` are the Jacobians of ``\mu`` with respect to ``z_t`` and ``y_t``, respectively;
``\Gamma_3, \Gamma_4`` are the Jacobians of ``\xi`` with respect to ``z_t`` and ``y_t``, respectively;
``\Gamma_5, \Gamma_6`` are constant matrices; ``\mathcal{V}(z)`` is the model's entropy;
``J\mathcal{V}(z)`` is the Jacobian of the entropy;
``J\mathcal{V}(z)`` is the Jacobian of the entropy;

and the state variables ``z_t`` and jump variables ``y_t`` follow
```math
\begin{aligned}
    z_{t + 1} & = z + \Gamma_1(z_t - z) + \Gamma_2(y_t - y) + (I_{n_z} - \Lambda(z_t) \Psi)^{-1}\Sigma(z_t)\varepsilon_{t + 1},\\
    y_t       & = y + \Psi(z_t - z).
\end{aligned}
```

The unknowns ``(z, y, \Psi)`` solve the system of equations
```math
\begin{aligned}
0 & = \mu(z, y) - z,\\
0 & = \xi(z, y) + \Gamma_5 z + \Gamma_6 y + \mathcal{V}(z),\\
0 & = \Gamma_3 + \Gamma_4 \Psi + (\Gamma_5 + \Gamma_6 \Psi)(\Gamma_1 + \Gamma_2 \Psi) + J\mathcal{V}(z).
\end{aligned}
```

Refer to [Lopez et al. (2018) "Risk-Adjusted Linearizations of Dynamic Equilibrium Models"](https://ideas.repec.org/p/bfr/banfra/702.html) for more details about the theory justifying this approximation approach.

## Implementation: `RiskAdjustedLinearization`

We implement risk-adjusted linearizations of nonlinear dynamic economic models
through the wrapper type `RiskAdjustedLinearization`.
The user only needs to define the functions and matrices characterizing the equilibrium of the nonlinear model. Once these
functions are defined, the user can create a `RiskAdjustedLinearization` object, which will automatically
create the Jacobian functions needed to compute the affine approximation.

To ensure efficiency in speed and memory, this package takes advantage of a number of features that are easily
accessible through Julia.

1. The Jacobians are calculated using forward-mode automatic differentiation rather than symbolic differentiation.

2. The Jacobian functions are constructed to be in-place with pre-allocated caches.

3. Functions provided by the user will be converted into in-place functions with pre-allocated caches.

4. (Coming in the future) Calculation of Jacobians with automatic differentiation is accelereated by exploiting sparsity with SparseDiffTools.jl

See the [Example](@ref example) for how to use the type.

```
@docs
RiskAdjustedLinearizations.RiskAdjustedLinearization
```


## Helper Types
TBD
