# [Sparse Arrays and Jacobians](@id sparse-arrays-jacs)
The risk-adjusted linearization of many economic models contains
substantial amounts of sparsity. The matrices ``\Gamma_5``
and ``\Gamma_6`` as well as the output of the functions
``\Lambda(\cdot)`` and ``\Sigma(\cdot)`` are typically sparse.
All of the Jacobians, ``\Gamma_1``, ``\Gamma_2``, ``\Gamma_3``,
``\Gamma_4``, and ``J\mathcal{V}``, are also very sparse.
To optimize performance, RiskAdjustedLinearizations.jl
allows users to leverage the sparsity of these objects.
The caches for the first set of objects can be
sparse matrices, assuming that ``\Lambda(\cdot)`` and ``\Sigma(\cdot)``
are written properly. The second set of objects are usually computed
with forward-mode automatic differentiation. By using matrix coloring techniques
implemented by [SparseDiffTools](https://github.com/JuliaDiff/SparseDiffTools.jl),
we can accelerate the calculation of these Jacobians and cache their output
as sparse matrices.

These methods can be easily used through keyword arguments of the main constructor of the
`RiskAdjustedLinearization` type.
We have also written examples which show how to use these methods and time their speed.
See the folder [examples/sparse_methods](https://github.com/chenwilliam77/RiskAdjustedLinearizations.jl/tree/main/examples/sparse_methods).
The script [sparse_arrays_and_jacobians.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations.jl/tree/main/examples/sparse_methods/sparse_arrays_and_jacobians.jl)
illustrates how to apply the methods described in this documentation page while
[sparse_nlsolve_jacobians.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations.jl/tree/main/examples/sparse_methods/sparse_nlsolve_jacobians.jl) describe how to use sparse automatic differentiation
to accelerate the calculation of Jacobians during calls to `nlsolve`. See [Numerical Algorithms](@ref numerical-algorithms)
for more details on the latter. Finally, the script
[combined_sparse_methods.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations.jl/tree/main/examples/sparse_methods/combined_sparse_methods.jl) combines these methods to achieve the fastest possible speeds with this package.

## Sparsity with ``\Gamma_5``, ``\Gamma_6``, ``\Lambda``, and ``\Sigma``
The matrices ``\Gamma_5`` and ``\Gamma_6`` are constants and can be passed in directly as
sparse matrices. The caches for ``\Lambda`` and ``\Sigma`` can be initialized as sparse matrices by using
the `Λ_Σ_cache_init` keyword for `RiskAdjustedLinearization`. This keyword is
a function which takes as input a `Tuple` of `Int` dimensions and allocates an array with
those dimensions. By default, the keyword has the value

```
Λ_Σ_cache_init = dims -> Matrix{Float64}(undef, dims...)
```

To use `SparseMatrixCSC` arrays, the user would instead pass

```
Λ_Σ_cache_init = dims -> spzeros(dims...)
```

However, using sparse arrays for caches may not always be faster
for calculating the steady state. To obtain ``\Psi``,
we need to apply the Schur decomposition, which requires dense matrices.
Thus, we still have to allocate dense versions of the sparse caches.

## Sparse Jacobians and Automatic Differentiation
To calculate a risk-adjusted linearization, we need to compute the Jacobians of ``\mu`` and ``\xi``
with respect to ``z`` and ``y`` as well as the Jacobian of ``\mathcal{V}`` with respect to ``z``.
These Jacobians are typically sparse because each equation in economic models
only has a small subset of variables. To exploit this sparsity, we utilize methods from
[SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl).

There are two ways to instruct a `RiskAdjustedLinearization` that the Jacobians of ``\mu``, ``\xi``,
and/or ``\mathcal{V}`` are sparse. The first applies during the construction of an instance
while the second occurs after an instance exists.

Note that sparse differentiation for this package is still a work in progress.
While working examples exist, the code still has bugs. The major problems
are listed below:

- Homotopy does not work yet with sparse automatic differentiation.
- `NaN`s or undefined values sometimes occur during calls to `nlsolve` within `solve!`.
  However,  the numerical algorithm can succeed if `solve!` is repeatedly run,
  even when using the same initial guess for the coefficients ``(z, y, \Psi)``. This happens
  at a sufficiently high frequency that using sparse automatic differentiation is not reliable.

### Specify Sparsity during Construction
When constructing a `RiskAdjustedLinearization`, the keyword `sparse_jacobian::Vector{Symbol}`
is a vector containing the symbols `:μ`, `:ξ`, and/or `:𝒱`. For example, if

```
sparse_jacobian = [:μ, :𝒱]
```

then the constructor will interpret that ``\mu`` has sparse Jacobians with respect to ``z`` and ``y`,
and that ``\mathcal{V}`` has a sparse Jacobian with respect to ``z``.

To implement sparse differentiation, the user needs to provide a sparsity pattern and a matrix coloring vector.
The user can use the keywords `sparsity` and `colorvec` to provide this information. These keywords
are dictionaries whose keys are the names of the Jacobians and values are the sparsity pattern and matrix coloring vector.
The relevant keys are `:μz`, `:μy`, `:ξz`, `:ξy`, and `:J𝒱`, where

- `:μz` and `:μy` are the Jacobians of `μ` with respect to ``z`` and ``y``,
- `:ξz` and `:ξy` are the Jacobians of `ξ` with respect to ``z`` and ``y``, and
- `:J𝒱` is the Jacobian of `𝒱` with respect to ``z``.

If `sparse_jacobian` is nonempty, but
one of these dictionaries is empty or does not contain the correct subset of the keys
`:μz`, `:μy`, `:ξz`, `:ξy`, and `:J𝒱`, then we attempt to determine the sparsity pattern
and/or matrix coloring vector. Once the sparsity pattern is known, the matrix coloring
vector is determined by calling `matrix_colors`.
We implement two approaches to discern the sparsity pattern. By default, we compute the dense Jacobian
once using ForwardDiff and assume that any zeros in the computed Jacobian are supposed to be zero. If this
assumption is true, then this Jacobian can be used as the sparsity pattern. Alternatively,
the user can set the keyword `sparsity_detection = true`, in which case we call `jacobian_sparsity`
from [SparsityDetection.jl](https://github.com/SciML/SparsityDetection.jl).
to determine the sparsity pattern. Currently, only the first approach works.

For ``\mu`` and ``\xi``, the first approach typically works fine. For ``\mathcal{V}``, however,
if the user guesses that ``\Psi`` is a matrix of zeros, then the Jacobian will be zero as well.
A good guess of ``\Psi`` is crucial to inferring the correct sparsity pattern of
``\mathcal{V}`` because different ``\Psi`` can imply different sparsity patterns.
For this reason, to fully exploit the sparsity in a model,
we recommend calculating the risk-adjusted linearization once using dense Jacobian methods.
The calculated Jacobians can be used subsequently as the sparsity patterns.


For reference, we again show the docstring for `RiskAdjustedLinearization`.

```@docs
RiskAdjustedLinearizations.RiskAdjustedLinearization
```

### Update a `RiskAdjustedLinearization` with Sparse Jacobians after Construction
Sparse Jacobians can be specified after a `RiskAdjustedLinearization` object `m` already exists
by calling `update_sparsity_pattern!(m, function_names)`.
The syntax of `update_sparsity_pattern!` is very similar to the specification of
sparse Jacobians in the constructor. The second input `function_names` is either
a `Symbol` or `Vector{Symbol}`, and it specifies the Jacobian(s) whose sparsity pattern(s) should be updated.
The relevent symbols are `:μz`, `:μy`, `:ξz`, `:ξy`, and `:J𝒱`.
If the Jacobians calculated by `m` are dense Jacobians, then `update_sparsity_pattern!`
will replace the functions computing dense Jacobians with functions that exploit sparsity.
If the Jacobians are already being calculated as sparse Jacobians,
then `update_sparsity_pattern!` can update the sparsity pattern and matrix coloring vector
being used.

If no keywords are passed, then `update_sparsity_pattern!` will
use the same methods as the constructor to infer the sparsity pattern. Either
we compute the dense Jacobian once using ForwardDiff, or we utilize SparsityDetection.
The new sparsity pattern and matrix coloring vectors can be specified using the
`sparsity` and `colorvec` keywords, just like the constructor.
Different values for ``z``, ``y``, and ``\Psi`` can also be used
when trying to infer the sparsity pattern by passing the new values as keywords.

```@docs
RiskAdjustedLinearizations.update_sparsity_pattern!
```
