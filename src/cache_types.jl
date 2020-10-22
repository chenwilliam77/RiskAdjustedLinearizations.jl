# RALF1
mutable struct RALF1{F <: Function, LC}
    f::F
    cache::LC
end

function RALF1(f::Function, x1::C1, array_type::DataType, dims::NTuple{N, Int};
               chunksize::Int = ForwardDiff.pickchunksize(length(x1))) where {C1 <: AbstractArray{<: Number}, N}
    cache = array_type(undef, dims)
    if applicable(f, cache, x1)
        fnew = function _f_ip(cache::LCN, x1::C1N) where {LCN <: DiffCache, C1N <: AbstractArray{<: Number}}
            f(get_tmp(cache, x1), x1)
            return get_tmp(cache, x1)
        end
    else
        fnew = function _f_oop(cache::LCN, x1::C1N) where {LCN <: DiffCache, C1N <: AbstractArray{<: Number}}
            get_tmp(cache, x1) .= f(x1)
            return get_tmp(cache, x1)
        end
    end
    return RALF1(fnew, dualcache(cache, Val{chunksize}))
end

function RALF1(fin::LC) where {LC <: AbstractArray{<: Number}}
    f(cache::LCN, x1::C1N) where {LCN <: AbstractMatrix{ <: Number}, C1N <: AbstractArray{<: Number}} = cache
    return RALF1{Function, LC}(f, fin)
end

function (ralf::RALF1)(x1::C1) where {C1 <: AbstractArray{<: Number}}
    return ralf.f(ralf.cache, x1)
end

# RALF2
mutable struct RALF2{F <: Function, LC}
    f::F
    cache::LC
end

function RALF2(f::Function, x1::C1, x2::C2, array_type::DataType,
               dims::NTuple{N, Int}, chunksizes::NTuple{Nc, Int} =
               (ForwardDiff.pickchunksize(min(length(x1), length(x2))), )) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number}, N, Nc}
    cache = array_type(undef, dims)
    if applicable(f, cache, x1, x2)
        if length(chunksizes) == 1 # Figure out which type of DiffCache is needed
            diffcache = dualcache(cache, Val{chunksizes[1]})
            fnew      = function _f_ip1(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: DiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                f(get_tmp(cache, x1, x2, select), x1, x2)
                return get_tmp(cache, x1, x2, select)
            end
        elseif length(chunksizes) == 2
            diffcache = twodualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]})
            fnew      = function _f_ip2(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: TwoDiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                f(get_tmp(cache, x1, x2, select), x1, x2)
                return get_tmp(cache, x1, x2, select)
            end
        elseif length(chunksizes) == 3
            diffcache = threedualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}, Val{chunksizes[3]})
            fnew      = function _f_ip3(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: ThreeDiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                f(get_tmp(cache, x1, x2, select), x1, x2)
                return get_tmp(cache, x1, x2, select)
            end
        else
            throw(MethodError("The length of the sixth input argument, chunksizes, must be 1, 2, or 3."))
        end
    else
        if length(chunksizes) == 1 # Figure out which type of DiffCache is needed
            diffcache = dualcache(cache, Val{chunksizes[1]})
            fnew      = function _f_oop1(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: DiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                get_tmp(cache, x1, x2, select) .= f(x1, x2)
                return get_tmp(cache, x1, x2, select)
            end
        elseif length(chunksizes) == 2
            diffcache = twodualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]})
            fnew      = function _f_oop2(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: TwoDiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                get_tmp(cache, x1, x2, select) .= f(x1, x2)
                return get_tmp(cache, x1, x2, select)
            end
        elseif length(chunksizes) == 3
            diffcache = threedualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}, Val{chunksizes[3]})
            fnew      = function _f_oop3(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: ThreeDiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                get_tmp(cache, x1, x2, select) .= f(x1, x2)
                return get_tmp(cache, x1, x2, select)
            end
        else
            throw(MethodError("The length of the sixth input argument, chunksizes, must be 1, 2, or 3."))
        end
    end
    return RALF2(fnew, diffcache)
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
