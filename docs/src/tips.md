# [Tips](@id tips)

This page of the documentation holds miscellaneous tips for using the package.

## Writing functions compatible with automatic differentiation

- **Use an in-place function to avoid type errors**. For example, define the `ccgf` as `ccgf(F, x)`.
  You can use the element type of `F` via `eltype(F)` to ensure that you don't get a type error
  from using `Float64` instead of `Dual` inside the function. If `ccgf` was out-of-place, then
  depending on how the vector being returned is coded, you may get a type error if elements
  of the return vector are zero or constant numbers. By having `F` available, you can
  guarantee these numbers can be converted to `Dual` types if needed without always
  declaring them as `Dual` types.

- **Use `dualvector` or `dualarray`**. The package provides these two helper functions
  in the case where you have a function `f(x, y)`, and you need to be able to automatcally
  differentiate with respect to `x` and `y` separately. For example, the nonlinear
  terms of the expectational equation `ξ(z, y)` takes this form. Within , you can
  pre-allocate the return vector by calling `F = RiskAdjustedLinearizations.dualvector(z, y)`.
  The `dualvector` function will infer from `z` and `y` whether `F` should be have `Dual` element types
  or not so you can repeatedly avoid writing if-else conditional blocks. The `dualarray` function
  generalizes this to arbitrary `AbstractMatrix` inputs.
  See the out-of-place function for `ξ` in [examples/wachter_disaster_risk/wachter.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations/tree/master/examples/wachter_disaster_risk/wachter.jl).

- **Don't pre-allocate the return vector**. Instead of pre-allocating the return vector at the
   top of the function for an out-of-place function, just concatenate the individual elements
   at the very end. Julia will figure out the appropriate element type for you. The downside of this
   approach is that you won't be able to assign names to the specific indices of the return vector (e.g.
   does this equation define the risk-free interest rate?). For small models, this disadvantage is generally not a problem.
   See the definition of the out-of-place expected state transition function `μ` in
  [examples/wachter_disaster_risk/wachter.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations/tree/master/examples/wachter_disaster_risk/wachter.jl).
