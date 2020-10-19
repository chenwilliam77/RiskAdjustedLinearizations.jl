# Concrete types used to create non-allocating functions
# by caching the output using the dualcache function from DiffEqBase.jl.
# This code could be automatically generated through macros or some other
# type of metaprogramming, but for our purposes, it is sufficient to define
# these types for up to two inputs (aside from the array used for in-place operations),
# i.e. RALF1 works for functions of the form f(F, x).
#
# For caching, we always infer whether to cache or not from the first element type since
# that element w/in context of this package always tells us whether or not we need to use the
# autodiffed version
# TODO: copy and paste this description for each function type
mutable struct RALF1{F <: Function, LC}
    f::F
    cache::LC
end

function RALF1(f::Function, x1::C1, array_type::DataType, dims::NTuple{N, Int};
               chunksize::Int = length(x1)) where {C1 <: AbstractArray{<: Number}, N}
    cache = array_type(undef, ntuple(x -> 0, length(dims))) # Create empty array first, just to check if f is in place or not
    if applicable(f, cache, x1)
        cache = array_type(undef, dims)
        fnew = function _f_ip(cache::LCN, x1::C1N) where {LCN <: DiffCache, C1N <: AbstractArray{<: Number}}
            f(get_tmp(cache, x1), x1)
            return get_tmp(cache, x1)
        end
        return RALF1(fnew, dualcache(cache, Val{chunksize}))
    else
        fnew = function _f_oop(cache::LCN, x1::C1N) where {LCN <: Nothing, C1N <: AbstractArray{<: Number}}
            return f(x1)
        end
        return RALF1(fnew, nothing)
    end
end

function RALF1(fin::LC) where {LC <: AbstractArray{<: Number}}
    f(cache::LCN, x1::C1N) where {LCN <: AbstractMatrix{ <: Number}, C1N <: AbstractArray{<: Number}} = cache
    return RALF1{Function, LC}(f, fin)
end

function (ralf::RALF1)(x1::C1) where {C1 <: AbstractArray{<: Number}}
    return ralf.f(ralf.cache, x1)
end

mutable struct RALF2{F <: Function, LC}
    f::F
    cache::LC
end

function RALF2(f::Function, x1::C1, x2::C2, array_type::DataType,
               dims::NTuple{N, Int}, chunksizes::Ntuple{Nc, Int} =
               (length(x1) + length(x2), )) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number}, N, Nc}
    cache = array_type(undef, ntuple(x -> 0, length(dims))) # Create empty array first, just to check if f is in place or not
    if applicable(f, cache, x1, x2)
        cache = array_type(undef, dims)
        fnew = function _f_ip(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: TwoDiffCache,
                                                                                           C1N <: AbstractArray{<: Number},
                                                                                            C2N <: AbstractArray{<: Number}}
                f(get_tmp(cache, x1, x2, select), x1, x2)
                return get_tmp(cache, x1, x2, select)
            end
        if length(chunksizes) == 1
            return RALF2(fnew, dualcache(cache, Val{chunksizes[1]}))
        elseif length(chunksizes) == 2
            return RALF2(fnew, twodualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}))
        elseif length(chunksizes) == 3
            return RALF2(fnew, threedualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}, Val{chunksizes[3]}))
        else
            throw(MethodError("The length of the sixth input argument, chunksizes, must be 1, 2, or 3."))
        end
    else
        fnew = function _f_oop(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int,Int}) where {LCN <: Nothing,
                                                                                            C1N <: AbstractArray{<: Number},
                                                                                            C2N <: AbstractArray{<: Number}}
            return f(x1, x2)
        end
        return RALF2(fnew, nothing)
    end
end

function RALF2(fin::LC) where {LC <: AbstractArray{<: Number}}
    f(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: AbstractArray{<: Number}, C1N <: AbstractArray{<: Number}, C2N <: AbstractArray{<: Number}} = cache
    return RALF2{Function, LC}(f, fin)
end

# Using the default of (1, 1) for `select` is important. This way, we can use autodiff
# during the homotopy algorithm without requiring additional arguments when calling `update!`
# to ensure the correct cache is used.
function (ralf::RALF2)(x1::C1, x2::C2, select::Tuple{Int, Int} = (1, 1)) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number}}
    return ralf.f(ralf.cache, x1, x2, select)
end

# The following two functions are required for RAL-specific purposes. Rather than add an additional field
# to enforce the safety of this call, we rely on the underlying `f` to throw a MethodError.
#=function (ralf::RALF2)(x1::C1, x2::C2, x3::C3, x4::C4) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number},
                                                              C3 <: AbstractArray{<: Number}, C4 <: AbstractArray{<: Number}}
    return ralf.f(ralf.cache, x1, x2, x3, x4)
end

function (ralf::RALF2)(x1::C1, x2::C2, x3::C3, x4::C4, x5::C5) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number},
                                                                      C3 <: AbstractArray{<: Number}, C4 <: AbstractArray{<: Number},
                                                                      C5 <: AbstractArray{<: Number}}
    return ralf.f(ralf.cache, x1, x2, x3, x4, x5)
end=#
