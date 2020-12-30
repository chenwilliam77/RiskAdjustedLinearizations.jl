# [Tips](@id tips)

This page of the documentation holds miscellaneous tips for using the package.

## [Deriving the conditional cumulant generating function](@id ccgf-tips)
The cumulant generating function is based upon the moment-generating function. If
```math
\begin{aligned}
M_X(t) \equiv \mathbb{E}[e^{tX}],\quad \quad \quad t\in \mathbb{R},
\end{aligned}
```
is the moment-generating function of a random variable ``X``, then the cumulant-generating function is just
```math
\begin{aligned}
cgf_X(t) \equiv \log\mathbb{E}[e^{tX}],\quad \quad \quad t\in \mathbb{R}.
\end{aligned}
```
As an example, if ``X \sim N(\mu, \sigma^2)``, then ``M_X(t) = \exp(t\mu + \sigma^2 t^2 / 2)`` and
``cgf_X(t) = t\mu + \sigma^2 t^2 / 2``.

Risk-adjusted linearizations imply that the relative entropy measure ``\mathcal{V}(\Gamma_5 z_{t + 1} + \Gamma_6 y_{t + 1})``
becomes a vector of conditional cumulant-generating functions for the random variables ``A_i(z_t) \varepsilon_{t + 1}``,
where ``A_i(z_t)`` is the ``i``th row vector of
```math
\begin{aligned}
A(z_t) = (\Gamma_5 + \Gamma_6 \Psi)(I_{n_z} - \Lambda(z_t) \Psi)^{-1} \Sigma(z_t).
\end{aligned}
```

To create a `RiskAdjustedLinearization`, the user needs to define a function `ccgf` in the form
`ccgf(F, A, z)` or `ccgf(A, z)`, where `A` refers to the matrix ``A(z_t)`` once it has already been
evaluated at ``z_t``. In other words, the input `A` should seen as a ``n_y \times n_\varepsilon`` matrix
of real scalars. However,
depending on the distributions of the martingale difference sequence ``\varepsilon_{t + 1}``,
writing the conditional cumulant-generating function may also require knowing the current state ``z_t``.

Let us consider two didactic examples. First, assume ``\varepsilon_{t + 1}\sim \mathcal{N}(0, I)``.
Then we claim
```
ccgf(A, z) = sum(A.^2, dims = 2) / 2
```
Based on the definition of ``\mathcal{V}(z_t)``, one may be tempted to derive the conditional cumulant-generating function
for the random vector ``A(z_t) \varepsilon_{t + 1}``. However, this is not actually what we want.
Rather, `ccgf` should just return a vector of conditional cumulant-generating functions
for the ``n_y`` random variables ``X_i = A_i(z_t)\varepsilon_{t + 1}``.

Because the individual components of ``\varepsilon_{t + 1}`` are independent and each $\varepsilon_{i, t}$ has
a standard Normal distribution,
the moment-generating function for ``X_i`` is ``\exp\left(\frac{1}{2}\left(\sum_{j = 1}^{n_\varepsilon} (t A_{ij})^2 / 2\right)\right)``, hence the ``i``th cumulant-generating function is ``\frac{1}{2}\left(\sum_{j = 1}^{n_\varepsilon} (t A_{ij})^2 / 2\right)``.
For risk-adjusted linearizations, we evaluate at $t = 1$ since we want the
conditional cumulant-generating function ``\log\mathbb{E}_t[\exp(A_i(z_t)\varepsilon_{t + 1})]``.
This is precisely what the code above achieves.

Second, let us consider a more complicated example. In the [Wachter (2013) Example](@ref example),
the ccgf is
```
function ccgf(F, α, z) # α is used here instead of A
    # the S term in z[S[:p]] is just an ordered dictionary mapping the symbol :p to the desired index of z
    F .= .5 .* α[:, 1].^2 + .5 * α[:, 2].^2 + (exp.(α[:, 3] + α[:, 3].^2 .* δ^2 ./ 2.) .- 1. - α[:, 3]) * z[S[:p]]
end
```
Observe that the first two quantities `.5 .* α[:, 1].^2 + .5 * α[:, 2].^2` resemble what would be obtained
from a standard multivariate normal distribution. The remaining terms are more complicated because
the Wachter (2013) model involves a Poisson mixture of normal distributions. It will be instructive to spell the details out.

Consumption growth follows the exogenous process
```math
\begin{aligned}
c_{t + 1} = \mu + c_t + \sigma \varepsilon^c_{t + 1} - \theta \xi_{t + 1},
\end{aligned}
```
where ``\varepsilon_t^c \sim N(0, 1)`` is iid over time and ``\xi_t \mid j_t \sim N(j_t, j_t\delta^2)``, where the number of jumps
``j_t \sim Poisson(p_{t - 1})``, hence ``\mathbb{E}_t \xi_{t + 1} = \mathbb{E}_t j_{t + 1} = p_t``. Assume that ``\varepsilon_t^c``
and ``\varepsilon_t^\xi = \xi_t - \mathbb{E}_{t - 1}\xi_t`` are independent.
Finally, the intensity ``p_t`` follows the process
```math
\begin{aligned}
p_{t + 1} = (1 - \rho_p) p + \rho_p p_t + \sqrt{p_t} \phi_p \sigma \varepsilon_{t + 1}^p,
\end{aligned}
```
where ``\varepsilon_t^p \sim N(0, 1)`` is iid over time and independent of ``\varepsilon_t^c`` and ``\varepsilon_t^\xi``.

Note that ``\xi_t`` and
``\mathbb{E}_{t - 1}\xi_t`` are not independent because ``\mathbb{E}_{t - 1}\xi_t = p_{t - 1}`` and ``j_t \sim Poisson(p_{t - 1})``,
hence a higher ``p_{t - 1}`` implies ``\xi_t`` is more likely to be higher. Re-centering ``\xi_t`` by ``\mathbb{E}_{t - 1}\xi_t``
creates a martingale difference sequence since ``\xi_t \mid j_t`` is normal.

By independence of the components of ``\varepsilon_t = [\varepsilon_t^c, \varepsilon_t^p, \varepsilon_t^\xi]^T``,
the conditional cumulant-generating function for the ``i``th row of the ``A(z_t)`` matrix described in this
[section](@ref affine-theory) is
```math
\begin{aligned}
ccgf_i[A_i(z_t) \mid z_t] &  =  \log\mathbb{E}_t[\exp(A_{i1}(z_t) \varepsilon_{t + 1}^c)]  + \log\mathbb{E}_t[\exp(A_{i2}(z_t) \varepsilon_{t + 1}^p)] + \log\mathbb{E}_t[\exp(A_{i3}(z_t) \varepsilon_{t + 1}^\xi)].
\end{aligned}
```
The first two terms on the RHS are for normal random variables and simplify to ``(A_{i1}(z_t)^2 + A_{i2}(z_t)^2) / 2``.
To calculate the remaining term, note that ``\mathbb{E}_{t}\xi_{t + 1} = p_t`` is already part of the information set
at ``z_t``, hence
```math
\begin{aligned}
\log\mathbb{E}_t[\exp(A_{i3}(z_t) \varepsilon_{t + 1}^\xi)] = \log\left[\frac{1}{\exp(A_{i3}(z_t) p_t)}\mathbb{E}_t\left[\exp(A_{i3}(z_t) \xi_{t + 1})\right]\right] = \log\mathbb{E}_t\left[\exp(A_{i3}(z_t) \xi_{t + 1})\right] - A_{i3}(z_t) p_t.
\end{aligned}
```

To calculate the cumulant-generating function of ``\xi_t``, aside from direct calculation,
we can also use the results for mixture distributions in
[Villa and Escobr (2006)](https://www.jstor.org/stable/27643733?seq=2#metadata_info_tab_contents) or
[Bagui et al. (2020)](https://www.atlantis-press.com/journals/jsta/125944282/view).
Given random variables ``X`` and ``Y``, assume the conditional distribution ``X\mid Y`` and the
marginal distribution for ``Y`` are available. If we can write the moment-generating function
for the random variable ``X\mid Y`` as
```math
\begin{aligned}
M_{X \mid Y}(s) = C_1(s) \exp(C_2(s) Y),
\end{aligned}
```
then the moment-generating function of ``X`` is
```math
\begin{aligned}
M_{X}(s) = C_1(s) M_Y[C_2(s)].
\end{aligned}
```

In our case, we have
```math
\begin{aligned}
M_{\xi_t \mid j_t}(s) = \exp\left(s j_t  + \frac{1}{2} s^2 \delta^2j_t  \right),
\end{aligned}
```
hence ``C_1(s) = 0`` and ``C_2(s) = (s + s^2\delta^2 / 2)``. The variable ``j_t`` has a Poisson distribution
with intensity ``p_t``, which implies the moment-generating function
```math
\begin{aligned}
M_{j_t}(s) = \exp((\exp(s) - 1) p_t).
\end{aligned}
```
Thus, as desired,
```math
\begin{aligned}
\log\mathbb{E}_t\left[\exp(A_{i3}(z_t) \xi_{t + 1})\right] - A_{i3}(z_t) p_t & = (\exp(A_{i3}(z_t)  + A_{i3}(z_t)^2\delta^2) - 1)p_t - A_{i3}(z_t) p_t.
\end{aligned}
```
Computing this quantity for each expectational equation yields the `ccgf` used in the [Wachter (2013) Example](@ref example).


## Writing functions compatible with automatic differentiation

- **Use an in-place function to avoid type errors.**

  For example, define the `ccgf` as `ccgf(F, x)`.
  You can use the element type of `F` via `eltype(F)` to ensure that you don't get a type error
  from using `Float64` instead of `Dual` inside the function. If `ccgf` was out-of-place, then
  depending on how the vector being returned is coded, you may get a type error if elements
  of the return vector are zero or constant numbers. By having `F` available, you can
  guarantee these numbers can be converted to `Dual` types if needed without always
  declaring them as `Dual` types.



- **Use `dualvector` or `dualarray`.**

  The package provides these two helper functions
  in the case where you have a function `f(x, y)`, and you need to be able to automatcally
  differentiate with respect to `x` and `y` separately. For example, the nonlinear
  terms of the expectational equation `ξ(z, y)` takes this form. Within , you can
  pre-allocate the return vector by calling `F = RiskAdjustedLinearizations.dualvector(z, y)`.
  The `dualvector` function will infer from `z` and `y` whether `F` should be have `Dual` element types
  or not so you can repeatedly avoid writing if-else conditional blocks. The `dualarray` function
  generalizes this to arbitrary `AbstractMatrix` inputs.
  See the out-of-place function for `ξ` in [examples/wachter\_disaster\_risk/wachter.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations/tree/main/examples/wachter_disaster_risk/wachter.jl).



- **Don't pre-allocate the return vector.**

  Instead of pre-allocating the return vector at the
   top of the function for an out-of-place function, just concatenate the individual elements
   at the very end. Julia will figure out the appropriate element type for you. The downside of this
   approach is that you won't be able to assign names to the specific indices of the return vector (e.g.
   does this equation define the risk-free interest rate?). For small models, this disadvantage is generally not a problem.
   See the definition of the out-of-place expected state transition function `μ` in
  [examples/wachter\_disaster\_risk/wachter.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations/tree/main/examples/wachter_disaster_risk/wachter.jl).
