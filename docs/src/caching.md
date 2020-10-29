# [Caching](@id caching)

If users create a `RiskAdjustedLinearization` with the constructor
```
RiskAdjustedLinearization(μ, Λ, Σ, ξ, Γ₅, Γ₆, ccgf, z, y, Ψ, Nε)
```
where `μ, Λ, Σ, ξ, ccgf` are either in-place or out-of-place functions, then
we use the wrapper types `RALF1` and `RALF2` to convert these
functions to non-allocating ones. The implementation of these wrappers is similar to the
implementation of `LinSolveFactorize` in
[DifferentialEquations.jl](https://diffeq.sciml.ai/stable/features/linear_nonlinear/#Implementing-Your-Own-LinSolve:-How-LinSolveFactorize-Was-Created).
See [Automated Caching via `RALF1` and `RALF2`](@ref ralfwrappers) for further details.
Unlike `LinSolveFactorize`, however, we need to be able to automatically differentiate with `RALF1` and `RALF2`
so the cache needs to be handled more carefully. To do this, we
utilize and extend the `DiffCache` type implemented by [DiffEqBase.jl](https://github.com/SciML/DiffEqBase.jl).

## [`TwoDiffCache` and `ThreeDiffCache`](@id new cache types)
The idea for `DiffCache` is that you need two caches, one for Dual numbers when applying automatic differentiation
and one for the subtype of `Number` used for the actual array (e.g. Float64). For the ``\Lambda`` and ``\Sigma`` functions,
this type works because they are functions of one input variables. The functions
``\mu_z``, ``\mu_y``, ``\xi_z``, and ``\xi_y`` also can use `DiffCache` once it is extended to work for functions with
two input variables (e.g. the chunk size should depend on the length of both input variables).

However, for the ``\mu``, ``\xi``, and ``\mathcal{V}`` functions, we actually need multiple caches for Dual numbers
that differ in their chunk size. The reason is that not only do we need to be able to evaluate them with Dual numbers but we also
need to apply automatic differentiation to calculate their Jacobians. Because all three of these functions take two input variables,
the chunk size for the cache used to evaluate the functions themselves will be different from the cache
used to calculate the Jacobians, which occur with respect to only one of the input variables.

*Note that the cache is always initialized as zeros by defaults (rather than undefined arrays).*

## [Automated Caching via `RALF1` and `RALF2` Wrappers](@id ralfwrappers)
The `RALF1` type applies to functions with 1 input variables (``\Lambda`` and ``\Sigma``) and
`RALF2` to functions with 2 input variables (e.g. ``\mu``, ``\mu_z``). The way these wrappers work is that
they take a user-defined function `f` and convert it to a new in-place function whose first input argument
is a cache, which is a `DiffCache`, `TwoDiffCache`, or a `ThreeDiffCache`.
The `RALF1` and `RALF2` types are callable in the same way `LinSolveFactorize` is.

For `RALF2`, the syntax `(x::RALF2)(x1, x2)` on its own would not work, however, because (1) it is not clear
which input should be used to infer whether or not to use a Dual cache and (2) there
are potentially multiple Dual caches. To get around this problem, we add an optional third argument
named `select`, which is a `Tuple{Int, Int}`. The first element specifies which input argument
to use for infering whether a Dual cache is needed, and the second element specifies which
cache to use. By default, `select = (1, 1)`.
