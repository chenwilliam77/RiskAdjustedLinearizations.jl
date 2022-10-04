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

```@meta
DocTestSetup = quote
    import ForwardDiff
    using RiskAdjustedLinearizations, ForwardDiff
end
```

```jldoctest
julia> a = rand(3)
julia> b = ones(ForwardDiff.Dual, 5)
julia> F = RiskAdjustedLinearizations.dualvector(a, b)
3-element Array{ForwardDiff.Dual,1}:
 #undef
 #undef
 #undef
```

```@meta
DocTestSetup = nothing
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

# Port of DiffCache type from old DiffEqBase, which is not in DiffEqBase nor in SciMLBase
struct DiffCache{T<:AbstractArray, S<:AbstractArray}
    du::T
    dual_du::S
end

function DiffCache(u::AbstractArray{T}, siz, ::Type{Val{chunk_size}}) where {T, chunk_size}
    x = ArrayInterface.restructure(u,zeros(ForwardDiff.Dual{nothing,T,chunk_size}, siz...))
    DiffCache(u, x)
end

dualcache(u::AbstractArray, N=Val{ForwardDiff.pickchunksize(length(u))}) = DiffCache(u, size(u), N)

function get_tmp(dc::DiffCache, u::AbstractArray{T}) where T<:ForwardDiff.Dual
    x = reinterpret(T, dc.dual_du)
end

function get_tmp(dc::DiffCache{A, B}, u::AbstractArray{T}) where {T <: ForwardDiff.Dual,
                                                                  A <: AbstractArray, B <: SparseMatrixCSC}
    if VERSION <= v"1.5"
        x = reinterpret(T, dc.dual_du)

        return x
    else
        error("It is not possible to reinterpret sparse matrices in your current version of Julia. " *
        "In order to cache sparse Jacobians in the current implementation of this package, " *
        "you need to be able to reinterpret sparse matrices. Please use an earlier version " *
        "of Julia (e.g. 1.3 or 1.5), or do not use the sparse Jacobian functionality.")
    end
end

function get_tmp(dc::DiffCache, u::LabelledArrays.LArray{T,N,D,Syms}) where {T,N,D,Syms}
    x = reinterpret(T, dc.dual_du.__x)
    LabelledArrays.LArray{T,N,D,Syms}(x)
end

get_tmp(dc::DiffCache, u::AbstractArray) = dc.du

# Extend get_tmp(dc::DiffCache, ...) to allow for two arguments
function get_tmp(dc::DiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual}
    if select[1] == 1
        get_tmp(dc, u1)
    elseif select[1] == 2
        get_tmp(dc, u2)
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache."))
    end
end
function get_tmp(dc::DiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 select::Tuple{Int, Int}) where {T1 <: Number, T2 <: ForwardDiff.Dual}
    if select[1] == 1
        get_tmp(dc, u1)
    elseif select[1] == 2
        get_tmp(dc, u2)
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache."))
    end
end
function get_tmp(dc::DiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: Number}
    if select[1] == 1
        get_tmp(dc, u1)
    elseif select[1] == 2
        get_tmp(dc, u2)
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache."))
    end
end

get_tmp(dc::DiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2}, select::Tuple{Int, Int}) where {T1 <: Number, T2 <: Number} = dc.du

# Extend get_tmp to allow 3 input arguments, only done for the case required by RiskAdjustedLinearizations.jl
function get_tmp(dc::DiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 u3::AbstractArray{T3}, select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual,
                                                                        T3 <: ForwardDiff.Dual}
    if select[1] == 1
        get_tmp(dc, u1)
    elseif select[1] == 2
        get_tmp(dc, u2)
    elseif select[1] == 3
        get_tmp(dc, u3)
    else
        throw(MethodError("Sixth input argument to get_tmp points to a non-existent cache."))
    end
end

get_tmp(dc::DiffCache, u1::AbstractArray{<: Number}, u2::AbstractArray{<: Number}, u3::AbstractArray{<: Number}, select::Tuple{Int, Int}) = dc.du

# Extend get_tmp to allow 4 input arguments, only done for the case required by RiskAdjustedLinearizations.jl
function get_tmp(dc::DiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 u3::AbstractArray{T3}, u4::AbstractArray{T4},
                 select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual,
                                                 T3 <: ForwardDiff.Dual, T4 <: ForwardDiff.Dual}
    if select[1] == 1
        get_tmp(dc, u1)
    elseif select[1] == 2
        get_tmp(dc, u2)
    elseif select[1] == 3
        get_tmp(dc, u3)
    elseif select[1] == 4
        get_tmp(dc, u4)
    else
        throw(MethodError("Sixth input argument to get_tmp points to a non-existent cache."))
    end
end

function get_tmp(dc::DiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 u3::AbstractArray{T3}, u4::AbstractArray{T4},
                 select::Tuple{Int, Int}) where {T1 <: Number, T2 <: Number,
                                                 T3 <: Number, T4 <: ForwardDiff.Dual}
    if select[1] == 1
        get_tmp(dc, u1)
    elseif select[1] == 2
        get_tmp(dc, u2)
    elseif select[1] == 3
        get_tmp(dc, u3)
    elseif select[1] == 4
        get_tmp(dc, u4)
    else
        throw(MethodError("Sixth input argument to get_tmp points to a non-existent cache."))
    end
end

get_tmp(dc::DiffCache, u1::AbstractArray{<: Number}, u2::AbstractArray{<: Number}, u3::AbstractArray{<: Number}, u4::AbstractArray{<: Number}, select::Tuple{Int, Int}) = dc.du

"""
```
TwoDiffCache
```

The TwoDiffCache type extends DiffCache from DiffEqBase to permit two Dual Array caches
to permit the case where you need to autodiff w.r.t 2 different lengths of arguments.
For example, suppose we have

```
function f(out, x, y) # in-place function
    out .= x .* y
end

function F(cache::DiffCache, x, y) # wrapper function for f
    f(get_tmp(cache, x), x, y)
    return get_tmp(cache, x)
end

# Instantiate inputs and caches
x = rand(3)
y = rand(3)
cache3 = dualcache(zeros(3), Val{3})
cache6 = dualcache(zeros(3), Val{6})
```

Then the following block of code will work

```
JF3 = (G, x1, y1) -> ForwardDiff.jacobian!(G, z -> F(cache3, z, y1), x1)
JF3(rand(3, 3), x, y)
```

but it may be the case that we also sometimes need to calculate

```
JF6 = (G, x1, y1) -> ForwardDiff.jacobian!(G, z -> F(cache3, z[1:3], z[4:6]), vcat(x1, y1))
JF6(rand(3, 3), x, y)
```

This block of code will error because the chunk size needs to be 6, not 3 here.
Therefore, the correct definition of `JF6` is

```
JF6 = (G, x1, y1) -> ForwardDiff.jacobian!(G, z -> F(cache6, z[1:3], z[4:6]), vcat(x1, y1))
```

Rather than carry around two `DiffCache` objects, it is better to simply add another dual cache
to a single `DiffCache` object. In principle, this code could be generalized to `n` dual caches,
but this would require some additional thought to implement generically.
"""
struct TwoDiffCache{T <: AbstractArray, C1 <: AbstractArray, C2 <: AbstractArray}
    du::T
    dual_du1::C1
    dual_du2::C2
end

function TwoDiffCache(u::AbstractArray{T}, siz, ::Type{Val{chunk_size1}}, ::Type{Val{chunk_size2}}) where {T, chunk_size1, chunk_size2}
    x1 = ArrayInterface.restructure(u, zeros(ForwardDiff.Dual{nothing, T, chunk_size1}, siz...))
    x2 = ArrayInterface.restructure(u, zeros(ForwardDiff.Dual{nothing, T, chunk_size2}, siz...))
    TwoDiffCache(u, x1, x2)
end

twodualcache(u::AbstractArray, N1, N2) = TwoDiffCache(u, size(u), N1, N2)

# get_tmp for AbstractArray cases
function get_tmp(tdc::TwoDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual}
    if select[1] == 1
        x = reinterpret(T1, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 2
        x = reinterpret(T2, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache."))
    end
    return x
end
function get_tmp(tdc::TwoDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 select::Tuple{Int, Int}) where {T1 <: Number, T2 <: ForwardDiff.Dual}
    if select[1] == 1
        x = reinterpret(T1, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 2
        x = reinterpret(T2, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache."))
    end
    return x
end
function get_tmp(tdc::TwoDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: Number}
    if select[1] == 1
        x = reinterpret(T1, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 2
        x = reinterpret(T2, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache."))
    end
    return x
end

# get_tmp for no Dual cases
get_tmp(tdc::TwoDiffCache, u1::AbstractArray, u2::AbstractArray, select::Tuple{Int, Int}) = tdc.du

# Extend get_tmp to allow 4 input arguments, only done for the case required by RiskAdjustedLinearizations.jl
function get_tmp(tdc::TwoDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 u3::AbstractArray{T3}, select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual,
                                                 T3 <: ForwardDiff.Dual}
    if select[1] == 1
        x = reinterpret(T1, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 2
        x = reinterpret(T2, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 3
        x = reinterpret(T3, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    else
        throw(MethodError("Sixth input argument to get_tmp points to a non-existent cache."))
    end
end

get_tmp(tdc::TwoDiffCache, u1::AbstractArray{<: Number}, u2::AbstractArray{<: Number}, u3::AbstractArray{<: Number}, select::Tuple{Int, Int}) = tdc.du

# Extend get_tmp to allow 4 input arguments, only done for the case required by RiskAdjustedLinearizations.jl
function get_tmp(tdc::TwoDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 u3::AbstractArray{T3}, u4::AbstractArray{T4},
                 select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual,
                                                 T3 <: ForwardDiff.Dual, T4 <: ForwardDiff.Dual}
    if select[1] == 1
        x = reinterpret(T1, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 2
        x = reinterpret(T2, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 3
        x = reinterpret(T3, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 4
        x = reinterpret(T4, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    else
        throw(MethodError("Sixth input argument to get_tmp points to a non-existent cache."))
    end
end

function get_tmp(tdc::TwoDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 u3::AbstractArray{T3}, u4::AbstractArray{T4},
                 select::Tuple{Int, Int}) where {T1 <: Number, T2 <: Number,
                                                 T3 <: Number, T4 <: ForwardDiff.Dual}
    if select[1] == 1
        x = reinterpret(T1, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 2
        x = reinterpret(T2, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 3
        x = reinterpret(T3, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    elseif select[1] == 4
        x = reinterpret(T4, select[2] == 1 ? tdc.dual_du1 : tdc.dual_du2)
    else
        throw(MethodError("Sixth input argument to get_tmp points to a non-existent cache."))
    end
end

get_tmp(tdc::TwoDiffCache, u1::AbstractArray{<: Number}, u2::AbstractArray{<: Number}, u3::AbstractArray{<: Number}, u4::AbstractArray{<: Number}, select::Tuple{Int, Int}) = tdc.du

"""
```
ThreeDiffCache
```

The ThreeDiffCache type extends DiffCache from DiffEqBase to permit three Dual Array caches.
"""
struct ThreeDiffCache{T <: AbstractArray, C1 <: AbstractArray, C2 <: AbstractArray, C3 <: AbstractArray}
    du::T
    dual_du1::C1
    dual_du2::C2
    dual_du3::C3
end

function ThreeDiffCache(u::AbstractArray{T}, siz, ::Type{Val{chunk_size1}},
                        ::Type{Val{chunk_size2}}, ::Type{Val{chunk_size3}}) where {T, chunk_size1, chunk_size2, chunk_size3}
    x1 = ArrayInterface.restructure(u, zeros(ForwardDiff.Dual{nothing, T, chunk_size1}, siz...))
    x2 = ArrayInterface.restructure(u, zeros(ForwardDiff.Dual{nothing, T, chunk_size2}, siz...))
    x3 = ArrayInterface.restructure(u, zeros(ForwardDiff.Dual{nothing, T, chunk_size3}, siz...))
    ThreeDiffCache(u, x1, x2, x3)
end

threedualcache(u::AbstractArray, N1, N2, N3) = ThreeDiffCache(u, size(u), N1, N2, N3)

# get_tmp for both AbstractArray
function get_tmp(tdc::ThreeDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual}
    dual_du = if select[2] == 1
        tdc.dual_du1
    elseif select[2] == 2
        tdc.dual_du2
    elseif select[2] == 3
        tdc.dual_du3
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache"))
    end

    if select[1] == 1
        x = reinterpret(T1, dual_du)
    elseif select[1] == 2
        x = reinterpret(T2, dual_du)
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache."))
    end
    return x
end
function get_tmp(tdc::ThreeDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 select::Tuple{Int, Int}) where {T1 <: Number, T2 <: ForwardDiff.Dual}
    dual_du = if select[2] == 1
        tdc.dual_du1
    elseif select[2] == 2
        tdc.dual_du2
    elseif select[2] == 3
        tdc.dual_du3
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache"))
    end

    if select[1] == 1
        x = reinterpret(T1, dual_du)
    elseif select[1] == 2
        x = reinterpret(T2, dual_du)
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache."))
    end
    return x
end
function get_tmp(tdc::ThreeDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: Number}
    dual_du = if select[2] == 1
        tdc.dual_du1
    elseif select[2] == 2
        tdc.dual_du2
    elseif select[2] == 3
        tdc.dual_du3
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache"))
    end

    if select[1] == 1
        x = reinterpret(T1, dual_du)
    elseif select[1] == 2
        x = reinterpret(T2, dual_du)
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache."))
    end
    return x
end

# get_tmp for no Dual cases
get_tmp(tdc::ThreeDiffCache, u1::AbstractArray, u2::AbstractArray, select::Tuple{Int, Int}) = tdc.du

# Extend get_tmp to allow 3 input arguments, only done for the case required by RiskAdjustedLinearizations.jl
function get_tmp(tdc::ThreeDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 u3::AbstractArray{T3}, select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual,
                                                                        T3 <: ForwardDiff.Dual}
    dual_du = if select[2] == 1
        tdc.dual_du1
    elseif select[2] == 2
        tdc.dual_du2
    elseif select[2] == 3
        tdc.dual_du3
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache"))
    end

    if select[1] == 1
        x = reinterpret(T1, dual_du)
    elseif select[1] == 2
        x = reinterpret(T2, dual_du)
    elseif select[1] == 3
        x = reinterpret(T3, dual_du)
    else
        throw(MethodError("Sixth input argument to get_tmp points to a non-existent cache."))
    end
end

get_tmp(tdc::ThreeDiffCache, u1::AbstractArray{<: Number}, u2::AbstractArray{<: Number}, u3::AbstractArray{<: Number}, select::Tuple{Int, Int}) = tdc.du

# Extend get_tmp to allow 4 input arguments, only done for the case required by RiskAdjustedLinearizations.jl
function get_tmp(tdc::ThreeDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 u3::AbstractArray{T3}, u4::AbstractArray{T4},
                 select::Tuple{Int, Int}) where {T1 <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual,
                                                 T3 <: ForwardDiff.Dual, T4 <: ForwardDiff.Dual}
    dual_du = if select[2] == 1
        tdc.dual_du1
    elseif select[2] == 2
        tdc.dual_du2
    elseif select[2] == 3
        tdc.dual_du3
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache"))
    end

    if select[1] == 1
        x = reinterpret(T1, dual_du)
    elseif select[1] == 2
        x = reinterpret(T2, dual_du)
    elseif select[1] == 3
        x = reinterpret(T3, dual_du)
    elseif select[1] == 4
        x = reinterpret(T4, dual_du)
    else
        throw(MethodError("Sixth input argument to get_tmp points to a non-existent cache."))
    end
end

function get_tmp(tdc::ThreeDiffCache, u1::AbstractArray{T1}, u2::AbstractArray{T2},
                 u3::AbstractArray{T3}, u4::AbstractArray{T4},
                 select::Tuple{Int, Int}) where {T1 <: Number, T2 <: Number,
                                                 T3 <: Number, T4 <: ForwardDiff.Dual}
    dual_du = if select[2] == 1
        tdc.dual_du1
    elseif select[2] == 2
        tdc.dual_du2
    elseif select[2] == 3
        tdc.dual_du3
    else
        throw(MethodError("Fourth input argument to get_tmp points to a non-existent cache"))
    end

    if select[1] == 1
        x = reinterpret(T1, dual_du)
    elseif select[1] == 2
        x = reinterpret(T2, dual_du)
    elseif select[1] == 3
        x = reinterpret(T3, dual_du)
    elseif select[1] == 4
        x = reinterpret(T4, dual_du)
    else
        throw(MethodError("Sixth input argument to get_tmp points to a non-existent cache."))
    end
end

get_tmp(tdc::ThreeDiffCache, u1::AbstractArray{<: Number}, u2::AbstractArray{<: Number}, u3::AbstractArray{<: Number}, u4::AbstractArray{<: Number}, select::Tuple{Int, Int}) = tdc.du
