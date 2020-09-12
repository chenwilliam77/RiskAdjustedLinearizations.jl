# [Numerical Algorithms](@id numerical algorithms)

To calculate the risk-adjusted linearization, we need to solve a system of nonlinear equations.
These equations are generally solvable using Newton-type methods. The package currently has two
available algorithms, [relaxation](@ref relaxation) and [homotopy continuation](@ref homotopy)

## `solve!`
The primary interface for calculating a risk-adjusted linearization once
a `RiskAdjustedLinearization` object is created is the function `solve!`.
The user selects the desired numerical algorithm through `algorithm`
keyword of `solve!`.

All of the available algorithms need to solve a system of nonlinear
equations. We use `nlsolve` for this purpose, and all keyword arguments
for `nlsolve` can be passed as keyword arguments to `solve!`, e.g.
`autodiff` and `ftol`.

```@doc
RiskAdjustedLinearizations.solve!
```

## [Relaxation](@id relaxation)
The first and default numerical algorithm is a relaxation algorithm. The key problem in
solving the equations characterizing ``(z, y, \Psi)`` is that it is difficult to jointly solve the nonlinear matrix
equation for ``\Psi`` along with the steady-state equations for ``z`` and ``y`` due to the presence of the
entropy term. The relaxation algorithm splits the solution of these equations into two steps, which
allows us to calculate guesses of ``\Psi`` using linear algebra. It is in this sense that
this iterative algorithm is a relaxation algorithm.

The system of equations
characterizing the coefficients ``(z, y, \Psi)`` are solved iteratively in two separate steps.
Given previous guesses ``(z_{n - 1}, y_{n - 1}, \Psi_{n - 1})``, we calculate ``(z_n, y_n)``
such that

```math
\begin{aligned}
0 & = \mu(z_n, y_n) - z_n,\\
0 & = \xi(z_n, y_n) + \Gamma_5 z_n + \Gamma_6 y_n + \mathcal{V}(z_{n - 1}),\\
\end{aligned}
```

is satisfied. In other words, we hold the entropy term constant and update ``(z_n, y_n)`` in the remaining terms.
The coefficients are solved efficiently through `nlsolve` with ``(z_{n - 1}, y_{n - 1})`` as initial guesses.

Then we compute ``\Psi_n`` by solving

```math
\begin{aligned}
0 & = \Gamma_3 + \Gamma_4 \Psi_n + (\Gamma_5 + \Gamma_6 \Psi_n)(\Gamma_1 + \Gamma_2 \Psi_n) + J\mathcal{V}(z_{n - 1}).
\end{aligned}
```

with a [Generalized Schur decomposition](https://en.wikipedia.org/wiki/Schur_decomposition#Generalized_Schur_decomposition)
(also known as QZ decomposition). Notice that we also hold the Jacobian of the entropy constant. Only after we have
a new round of ``(z_n, y_n, \Psi_n)`` do we update the entropy-related terms.

Convergence is achieved once ``(z_n, y_n, \Psi_n)`` are sufficiently close under some norm. By default,
we use the ``L^\infty`` norm (maximum absolute error).

## [Homotopy Continuation](@id homotopy)
When the deterministic steady state exists, it is typically an easy problem to solve numerically. We can therefore
use the equations characterizing the deterministic steady state for a
[homotopy continuation method](https://en.wikipedia.org/wiki/Numerical_algebraic_geometry).
Let ``q`` be the embedding parameter. Then the homotopy continuation method iteratively solves

```math
\begin{aligned}
0 & = \mu(z, y) - z,\\
0 & = \xi(z, y) + \Gamma_5 z + \Gamma_6 y + q \mathcal{V}(z),\\
0 & = \Gamma_3 + \Gamma_4 \Psi + (\Gamma_5 + \Gamma_6 \Psi)(\Gamma_1 + \Gamma_2 \Psi) + q J\mathcal{V}(z)
\end{aligned}
```

for the coefficients ``(z_q, y_q, \Psi_q)`` by increasing ``q`` from 0 to 1.


## [Blanchard-Kahn Conditions](@id blanchard-kahn)

At the end of `solve!`, we check the stochastic steady state found is
locally unique and saddle-path stable by checking what are known as the Blanchard-Kahn conditions.
Standard references for computational macroeconomics explain what these conditions are, so
we defer to them (e.g. [Blanchard-Kahn (1980)](http://dept.ku.edu/~empirics/Emp-Coffee/blanchard-kahn_eca80.pdf),
[Klein (2000)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.8685&rep=rep1&type=pdf), and
[Sims (2002)](https://link.springer.com/article/10.1023/A:1020517101123)).
For the stochastic steady state, these conditions are essentially identical to the conditions for
the deterministic steady state, but the Jacobian of the expectational equations to ``z_t``
also includes the Jacobian of the entropy. In the deterministic steady state, the entropy is zero,
hence the Jacobian of the entropy is zero. In the stochastic steady state, the entropy is no longer zero
and varies with ``z_t``, hence the Jacobian of the expectational equations to ``z_t`` depends on entropy.


## Docstrings
```@docs
RiskAdjustedLinearizations.relaxation!
RiskAdjustedLinearizations.homotopy!
RiskAdjustedLinearizations.blanchard_kahn
```