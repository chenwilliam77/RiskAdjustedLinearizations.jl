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
The first vector equation comprise the transition equations of the state variables.
The second vector equation comprise the model's expectational equations, which are typically
the first-order conditions for the jump variables from agents' optimization problem.
The exogenous shocks
``\varepsilon_t \in\mathbb{R}^{n_\varepsilon}`` form a martingale difference sequence. Given some
differentiable mapping ``\alpha:\mathbb{R}^{n_z}\rightarrow\mathbb{R}^{n_\varepsilon}``,
the random variable ``X_t = \alpha(z_t)^T \varepsilon_{t + 1}`` has the
differentiable, conditional (on ``z_t``) cumulant generating function (ccgf)

```math
\begin{aligned}
\kappa[\alpha(z_t) \mid z_t] = \log\mathbb{E}_t[\exp(\alpha(z_t)^T \varepsilon_{t + 1})].
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
innovations in jump variables while the function ``\Sigma`` characterizes exogenous risk. These latter two functions
can also depend on jump variables. Denote the jump-dependent versions as
``\tilde{\Lambda}:\mathbb{R}^{n_z\times n_y} \rightarrow \mathbb{R}^{n_z \times n_y}``
and ``\tilde{\Sigma}:\mathbb{R}^{n_z \times n_y}\ \rightarrow \mathbb{R}^{n_z\times n_\varepsilon}``.
If there exists a mapping ``y_t = y(z_t)``, then we define ``\Lambda(z_t) = \tilde{\Lambda}(z_t, y(z_t))``
and ``\Sigma(z_t) = \tilde{\Sigma}(z_t, y(z_t))``.

The expectational equations can be simplified as
```math
\begin{aligned}
0 & = \log\mathbb{E}_t[\exp(\xi(z_t, y_t) + \Gamma_5 z_{t + 1} + \Gamma_6 y_{t + 1})]\\
  & = \log[\exp(\xi(z_t, y_t))\mathbb{E}_t[\exp(\Gamma_5 z_{t + 1} + \Gamma_6 y_{t + 1})]]\\
  & = \xi(z_t, y_t) + \Gamma_5\mathbb{E}_t z_{t + 1} + \Gamma_6 \mathbb{E}_t y_{t + 1} + \log\mathbb{E}_t[\exp(\Gamma_5 z_{t + 1} + \Gamma_6 y_{t + 1})] - (\Gamma_5\mathbb{E}_t z_{t + 1} + \Gamma_6 \mathbb{E}_t y_{t + 1})
  & = \xi(z_t, y_t) + \Gamma_5\mathbb{E}_t z_{t + 1} + \Gamma_6 \mathbb{E}_t y_{t + 1} + \mathcal{V}(\Gamma_5 z_{t + 1} + \Gamma_6 y_{t + 1}),
\end{aligned}
```
where the last term is defined to be
```math
\begin{aligned}
\mathcal{V}(x_{t + 1}) = \log\mathbb{E}_t[\exp(x_{t + 1})] - \mathbb{E}_t x_{t + 1}.
\end{aligned}
```
As Lopez et al. (2018) describe it, this quantity "is a relative entropy measure, i.e. a nonnegative measure of dispersion that generalizes variance."

### [Risk-Adjusted Linearizations by Affine Approximation](@id affine-theory)

Many economic models are typically solved by perturbation around the deterministic steady state. To break certainty equivalence so that
asset pricing is meaningful, these perturbations need to be at least third order. However, even third-order perturbations
can poorly approximate the true global solution. A key problem is that the economy may not spend much time near the
deterministic steady state, so a perturbation around this point will be inaccurate.


Instead of perturbing the model's nonlinear equations around the deterministic steady state, we could perturb around the
stochastic or "risky" steady state. This point is better for a perturbation because the economy will spend a
large amount of time near the stochastic steady state. [Lopez et al. (2018)](https://ideas.repec.org/p/bfr/banfra/702.html)
show that an affine approximation of the model's nonlinear equation is equivalent to a linearization around the
stochastic steady state. Further, they confirm that in practice this "risk-adjusted" linearization approximates
global solutions of canonical economic models very well and outperforms perturbations around the deterministic steady state.

The affine approximation of an dynamic economic model is
```math
\begin{aligned}
    \mathbb{E}[z_{t + 1}] & = \mu(z, y) + \Gamma_1(z_t - z) + \Gamma_2(y_t - y),\\
    0                      & = \xi(z, y) + \Gamma_3(z_t - z) + \Gamma_4(y_t - y) + \Gamma_5 \mathbb{E}_t z_{t + 1} + \Gamma_6 \mathbb{E}_t y_{t + 1} + \mathcal{V}(z) + J\mathcal{V}(z)(z_t  - z),
\end{aligned}
```

where ``\Gamma_1, \Gamma_2`` are the Jacobians of ``\mu`` with respect to ``z_t`` and ``y_t``, respectively;
``\Gamma_3, \Gamma_4`` are the Jacobians of ``\xi`` with respect to ``z_t`` and ``y_t``, respectively;
``\Gamma_5, \Gamma_6`` are constant matrices; ``\mathcal{V}(z)`` is the model's entropy; and
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

Under an affine approximation, the entropy term is a nonnegative function
``\mathcal{V}:\mathbb{R}^{n_z} \rightarrow \mathbb{R}_+^{n_y}`` defined such that
```math
\begin{aligned}
\mathcal{V}(z_t) \equiv \mathcal{V}_t(\exp((\Gamma_5 + \Gamma_6 \Psi)z_{t + 1})) = \vec{\kappa}[(\Gamma_5 + \Gamma_6 \Psi)(I_{n_z} - \Lambda(z_t) \Psi)^{-1} \Sigma(z_t) \mid z_t]
\end{aligned}
```
where the notation ``\vec{\kappa}`` means that each component ``\kappa_i[\cdot \mid \cdot]`` is a conditional cumulant-generating
function. Explicitly, define
```math
\begin{aligned}
A(z_t) = (\Gamma_5 + \Gamma_6 \Psi)(I_{n_z} - \Lambda(z_t) \Psi)^{-1} \Sigma(z_t) = [A_1(z_t), \dots, A_{n_y}(z_t)]^T.
\end{aligned}
```
Each ``A_i(z_t)`` is a mapping from ``z_t`` to the ``i``th row vector in ``A(z_t)``. Then
``\kappa_i[\cdot \mid \cdot]`` is the ccgf
```math
\begin{aligned}
\kappa_i[A_i(z_t)\mid z_t] = \log\mathbb{E}_t[\exp(A_i(z_t) \varepsilon_{t + 1})].
\end{aligned}
```
Every ``\kappa_i[\cdot \mid \cdot]`` corresponds to an expectational equation and thus
acts as a risk correction to each one. In the common case where the individual components of
``\varepsilon_{t + 1}`` are independent, the
ccgf simplifies to
```math
\begin{aligned}
\kappa_i[A_i(z_t)\mid z_t] = \sum_{j = 1}^{n_\varepsilon}\log\mathbb{E}_t[\exp(A_{ij}(z_t) \varepsilon_{j, t + 1})],
\end{aligned}
```
i.e. it is the sum of the cumulant-generating functions for each shock ``\varepsilon_{j, t + 1}``.

Refer to [Lopez et al. (2018) "Risk-Adjusted Linearizations of Dynamic Equilibrium Models"](https://ideas.repec.org/p/bfr/banfra/702.html) for more details about the theory justifying this approximation approach.
See [Deriving the conditional cumulant generating function](@ref ccgf-tips) for some guidance on calculating the ccgf, which
many users may not have seen before.

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

4. (Coming in the future) Calculation of Jacobians with automatic differentiation is accelerated by exploiting sparsity with SparseDiffTools.jl

See the [Example](@ref example) for how to use the type.

```
@docs
RiskAdjustedLinearizations.RiskAdjustedLinearization
```


## Helper Types
To organize the functions comprisng a risk-adjusted linearization, we create two helper types, `RALNonlinearSystem` and `RALLinearizedSystem`.
The first type holds the ``\mu``, ``\Lambda``, ``\Sigma``, ``\xi``, and ``\mathcal{V}`` functions while the second type holds
the ``\mu_z``, ``\mu_y``, ``\xi_z``, ``\xi_y``, ``J\mathcal{V}``, ``\Gamma_5``, and ``\Gamma_6`` quantities.
The `RALNonlinearSystem` type holds potentially nonlinear functions, and in particular ``\mu``, ``\xi``, and ``\mathcal{V}``,
which need to be linearized (e.g. by automatic differentiation). The `RALLinearizedSystem` holds both matrices that
are only relevant once the model is linearized, such as ``\Gamma_1`` (calculated by ``\mu_z``), as well as ``\Gamma_5`` and ``\Gamma_6``
since these latter two quantities are always constant matrices.

Aside from providing a way to organize the various functions comprising a risk-adjusted linearization, these helper types do not
have much additional functionality. The `update!` functions for a `RiskAdjustedLinearization`, for example, are implemented
underneath the hood by calling `update!` functions written for the `RALNonlinearSystem` and `RALLinearizedSystem`.
