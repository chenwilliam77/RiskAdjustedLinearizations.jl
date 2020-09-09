"""
```
dualarray(a::AbstractArray, b::AbstractArray)

```

returns an `Array` that has the size of input `a` and an element type consistent with the element types
of the inputs `a` and `b`. For example, suppose you want to write a function of the form

```
julia> function f(a, b)
           F = similar(a)
           F[1] = a[1] * b[1]
           F[2] = a[2] + b[2]
       end
```

If you were to automatically differentiate `f` with respect to `b`, then `F` would not have the correct element type since its type
will be the same as `a`. Rather than require the user to write

```
julia> F = if eltype(b) <: ForwardDiff.Dual
           similar(a, eltype(b))
       else
           similar(a)
       end
```

the user can use `dualvector` to write

```jldoctest
julia> a = rand(3)
julia> b = ones(ForwardDiff.Dual, 5)
julia> F = dualvector(a, b)
3-element Array{ForwardDiff.Dual,1}:
 #undef
 #undef
 #undef
```

Note that the element types of `a` and `b` must be subtypes of `Real` (or else `ForwardDiff` will not work).
"""
@inline dualarray(a::AbstractArray{<: ForwardDiff.Dual}, b::AbstractArray{<: ForwardDiff.Dual}) = similar(a)
@inline dualarray(a::AbstractArray{<: ForwardDiff.Dual}, b::AbstractArray{<: Real})             = similar(a)
@inline dualarray(a::AbstractArray{<: Real}, b::AbstractArray{<: Real})                         = similar(a)
@inline dualarray(a::AbstractArray{<: Real}, b::AbstractArray{<: ForwardDiff.Dual})             = similar(a, eltype(b))

"""
```
dualvector(a::AbstractVector, b::AbstractVector)
```

has the same behavior of `dualarray` but acts specifically on vectors. This function
is primarily for user convenience.
"""
@inline dualvector(a::AbstractVector{<: ForwardDiff.Dual}, b::AbstractVector{<: ForwardDiff.Dual}) = similar(a)
@inline dualvector(a::AbstractVector{<: ForwardDiff.Dual}, b::AbstractVector{<: Real})             = similar(a)
@inline dualvector(a::AbstractVector{<: Real}, b::AbstractVector{<: Real})                         = similar(a)
@inline dualvector(a::AbstractVector{<: Real}, b::AbstractVector{<: ForwardDiff.Dual})             = similar(a, eltype(b))
