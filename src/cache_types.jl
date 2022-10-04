abstract type AbstractRALF end

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

function DiffEqBase.get_tmp(dc::DiffEqBase.DiffCache, u::LabelledArrays.LArray{T,N,D,Syms}) where {T,N,D,Syms}
    x = reinterpret(T, dc.dual_du.__x)
    LabelledArrays.LArray{T,N,D,Syms}(x)
end

get_tmp(dc::DiffCache, u::AbstractArray) = dc.du

# RALF1
mutable struct RALF1{LC} <: AbstractRALF
    f::Function
    f0::Function
    cache::LC
end

get_cache_type(ral::RALF1{LC}) where {LC} = LC


function RALF1(f::Function, x1::C1, cache::AbstractArray{<: Number};
               chunksize::Int = ForwardDiff.pickchunksize(length(x1))) where {C1 <: AbstractArray{<: Number}, N}
    if applicable(f, cache, x1)
        fnew = function _f_ip(cache::LCN, x1::C1N) where {LCN <: DiffCache, C1N <: AbstractArray{<: Number}}
            target_cache = get_tmp(cache, x1)
            if size(target_cache) != size(cache.du)
                target_cache = reshape(target_cache, size(cache.du))
            end
            f(target_cache, x1)
            return target_cache
        end
    else
        fnew = function _f_oop(cache::LCN, x1::C1N) where {LCN <: DiffCache, C1N <: AbstractArray{<: Number}}
            target_cache = get_tmp(cache, x1)
            if size(target_cache) != size(cache.du)
                target_cache = reshape(target_cache, size(cache.du))
            end
            target_cache .= f(x1)
            return target_cache
        end
    end
    return RALF1(fnew, f, dualcache(cache, Val{chunksize}))
end

function RALF1(fin::LC) where {LC <: AbstractArray{<: Number}}
    f(cache::LCN, x1::C1N) where {LCN <: AbstractMatrix{ <: Number}, C1N <: AbstractArray{<: Number}} = cache
    return RALF1{LC}(f, x -> fin, fin)
end

function (ralf::RALF1)(x1::C1) where {C1 <: AbstractArray{<: Number}}
    return ralf.f(ralf.cache, x1)
end

# RALF2
mutable struct RALF2{LC} <: AbstractRALF
    f::Function
    f0::Function
    cache::LC
end

get_cache_type(ral::RALF2{LC}) where {LC} = LC

function RALF2(f::Function, x1::C1, x2::C2, cache::AbstractArray{<: Number}, chunksizes::NTuple{Nc, Int} =
               (ForwardDiff.pickchunksize(min(length(x1), length(x2))), )) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number}, N, Nc}
    if applicable(f, cache, x1, x2)
        if length(chunksizes) == 1 # Figure out which type of DiffCache is needed
            diffcache = dualcache(cache, Val{chunksizes[1]})
            fnew      = function _f_ip1(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: DiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                f(target_cache, x1, x2)
                return target_cache
            end
        elseif length(chunksizes) == 2
            diffcache = twodualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]})
            fnew      = function _f_ip2(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: TwoDiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                f(target_cache, x1, x2)
                return target_cache
            end
        elseif length(chunksizes) == 3
            diffcache = threedualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}, Val{chunksizes[3]})
            fnew      = function _f_ip3(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: ThreeDiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                f(target_cache, x1, x2)
                return target_cache
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
                target_cache = get_tmp(cache, x1, x2, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                target_cache .= f(x1, x2)
                return target_cache
            end
        elseif length(chunksizes) == 2
            diffcache = twodualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]})
            fnew      = function _f_oop2(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: TwoDiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                target_cache .= f(x1, x2)
                return target_cache
            end
        elseif length(chunksizes) == 3
            diffcache = threedualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}, Val{chunksizes[3]})
            fnew      = function _f_oop3(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: ThreeDiffCache,
                                                                                                     C1N <: AbstractArray{<: Number},
                                                                                                     C2N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                target_cache .= f(x1, x2)
                return target_cache
            end
        else
            throw(MethodError("The length of the sixth input argument, chunksizes, must be 1, 2, or 3."))
        end
    end
    return RALF2(fnew, f, diffcache)
end

function RALF2(fin::LC) where {LC <: AbstractArray{<: Number}}
    f(cache::LCN, x1::C1N, x2::C2N, select::Tuple{Int, Int}) where {LCN <: AbstractArray{<: Number}, C1N <: AbstractArray{<: Number}, C2N <: AbstractArray{<: Number}} = cache
    return RALF2{LC}(f, x -> fin, fin)
end

# Using the default of (1, 1) for `select` is important. This way, we can use autodiff
# during the homotopy algorithm without requiring additional arguments when calling `update!`
# to ensure the correct cache is used.
function (ralf::RALF2)(x1::C1, x2::C2, select::Tuple{Int, Int} = (1, 1)) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number}}
    return ralf.f(ralf.cache, x1, x2, select)
end

# RALF3
mutable struct RALF3{LC} <: AbstractRALF
    f::Function
    f0::Function
    cache::LC
end

get_cache_type(ral::RALF3{LC}) where {LC} = LC

function RALF3(f::Function, x1::C1, x2::C2, x3::C3, cache::AbstractArray{<: Number},
               chunksizes::NTuple{Nc, Int} =
               (ForwardDiff.pickchunksize(min(length(x1),
                                              length(x2), length(x3))), )) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number},
                                                                                              C3 <: AbstractArray{<: Number}, N, Nc}
    if applicable(f, cache, x1, x2, x3)
        if length(chunksizes) == 1 # Figure out which type of DiffCache is needed
            diffcache = dualcache(cache, Val{chunksizes[1]})
            fnew      = function _f_ip1(cache::LCN, x1::C1N, x2::C2N,
                                        x3::C3N, select::Tuple{Int, Int}) where {LCN <: DiffCache,
                                                                                          C1N <: AbstractArray{<: Number},
                                                                                          C2N <: AbstractArray{<: Number},
                                                                                          C3N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                f(target_cache, x1, x2, x3)
                return target_cache
            end
        elseif length(chunksizes) == 2
            diffcache = twodualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]})
            fnew      = function _f_ip2(cache::LCN, x1::C1N, x2::C2N,
                                        x3::C3N, select::Tuple{Int, Int}) where {LCN <: TwoDiffCache,
                                                                                 C1N <: AbstractArray{<: Number},
                                                                                 C2N <: AbstractArray{<: Number},
                                                                                 C3N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                f(target_cache, x1, x2, x3)
                return target_cache
            end
        elseif length(chunksizes) == 3
            diffcache = threedualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}, Val{chunksizes[3]})
            fnew      = function _f_ip3(cache::LCN, x1::C1N, x2::C2N,
                                        x3::C3N, select::Tuple{Int, Int}) where {LCN <: ThreeDiffCache,
                                                                                 C1N <: AbstractArray{<: Number},
                                                                                 C2N <: AbstractArray{<: Number},
                                                                                 C3N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                f(target_cache, x1, x2, x3)
                return target_cache
            end
        else
            throw(MethodError("The length of the seventh input argument, chunksizes, must be 1, 2, or 3."))
        end
    else
        if length(chunksizes) == 1 # Figure out which type of DiffCache is needed
            diffcache = dualcache(cache, Val{chunksizes[1]})
            fnew      = function _f_oop1(cache::LCN, x1::C1N, x2::C2N,
                                         x3::C3N, select::Tuple{Int, Int}) where {LCN <: DiffCache,
                                                                                  C1N <: AbstractArray{<: Number},
                                                                                  C2N <: AbstractArray{<: Number},
                                                                                  C3N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                target_cache .= f(x1, x2, x3)
                return target_cache
            end
        elseif length(chunksizes) == 2
            diffcache = twodualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]})
            fnew      = function _f_oop2(cache::LCN, x1::C1N, x2::C2N,
                                         x3::C3N, select::Tuple{Int, Int}) where {LCN <: TwoDiffCache,
                                                                                  C1N <: AbstractArray{<: Number},
                                                                                  C2N <: AbstractArray{<: Number},
                                                                                  C3N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                target_cache .= f(x1, x2, x3)
                return target_cache
            end
        elseif length(chunksizes) == 3
            diffcache = threedualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}, Val{chunksizes[3]})
            fnew      = function _f_oop3(cache::LCN, x1::C1N, x2::C2N,
                                         x3::C3N, select::Tuple{Int, Int}) where {LCN <: ThreeDiffCache,
                                                                                  C1N <: AbstractArray{<: Number},
                                                                                  C2N <: AbstractArray{<: Number},
                                                                                  C3N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                target_cache .= f(x1, x2, x3)
                return target_cache
            end
        else
            throw(MethodError("The length of the seventh input argument, chunksizes, must be 1, 2, or 3."))
        end
    end
    return RALF3(fnew, f, diffcache)
end

function RALF3(fin::LC) where {LC <: AbstractArray{<: Number}}
    f(cache::LCN, x1::C1N, x2::C2N, x3::C3N, select::Tuple{Int, Int}) where {LCN <: AbstractArray{<: Number}, C1N <: AbstractArray{<: Number}, C2N <: AbstractArray{<: Number}, C3N <: AbstractArray{<: Number}} = cache
    return RALF3{LC}(f, x -> fin, fin)
end

# Using the default of (1, 1) for `select` is important. This way, we can use autodiff
# during the homotopy algorithm without requiring additional arguments when calling `update!`
# to ensure the correct cache is used.
function (ralf::RALF3)(x1::C1, x2::C2, x3::C3,
                       select::Tuple{Int, Int} = (1, 1)) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number},
                                                                C3 <: AbstractArray{<: Number}}
    return ralf.f(ralf.cache, x1, x2, x3, select)
end

# RALF4
mutable struct RALF4{LC} <: AbstractRALF
    f::Function
    f0::Function
    cache::LC
end

get_cache_type(ral::RALF4{LC}) where {LC} = LC

function RALF4(f::Function, x1::C1, x2::C2, x3::C3, x4::C4, cache::AbstractArray{<: Number},
               chunksizes::NTuple{Nc, Int} =
               (ForwardDiff.pickchunksize(min(length(x1),
                                              length(x2), legnth(x3), length(x4))), )) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number},
                                                                                              C3 <: AbstractArray{<: Number}, C4 <: AbstractArray{<: Number}, N, Nc}

    if applicable(f, cache, x1, x2, x3, x4)
        if length(chunksizes) == 1 # Figure out which type of DiffCache is needed
            diffcache = dualcache(cache, Val{chunksizes[1]})
            fnew      = function _f_ip1(cache::LCN, x1::C1N, x2::C2N,
                                        x3::C3N, x4::C4N, select::Tuple{Int, Int}) where {LCN <: DiffCache,
                                                                                          C1N <: AbstractArray{<: Number},
                                                                                          C2N <: AbstractArray{<: Number},
                                                                                          C3N <: AbstractArray{<: Number},
                                                                                          C4N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, x4, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                f(target_cache, x1, x2, x3, x4)
                return target_cache
            end
        elseif length(chunksizes) == 2
            diffcache = twodualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]})
            fnew      = function _f_ip2(cache::LCN, x1::C1N, x2::C2N,
                                        x3::C3N, x4::C4N, select::Tuple{Int, Int}) where {LCN <: TwoDiffCache,
                                                                                          C1N <: AbstractArray{<: Number},
                                                                                          C2N <: AbstractArray{<: Number},
                                                                                          C3N <: AbstractArray{<: Number},
                                                                                          C4N <: AbstractArray{<: Number}}
                f(get_tmp(cache, x1, x2, x3, x4, select), x1, x2, x3, x4)
                return get_tmp(cache, x1, x2, x3, x4, select)
            end
        elseif length(chunksizes) == 3
            diffcache = threedualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}, Val{chunksizes[3]})
            fnew      = function _f_ip3(cache::LCN, x1::C1N, x2::C2N,
                                        x3::C3N, x4::C4N, select::Tuple{Int, Int}) where {LCN <: ThreeDiffCache,
                                                                                          C1N <: AbstractArray{<: Number},
                                                                                          C2N <: AbstractArray{<: Number},
                                                                                          C3N <: AbstractArray{<: Number},
                                                                                          C4N <: AbstractArray{<: Number}}
                f(get_tmp(cache, x1, x2, x3, x4, select), x1, x2, x3, x4)
                return get_tmp(cache, x1, x2, x3, x4, select)
            end
        else
            throw(MethodError("The length of the eighth input argument, chunksizes, must be 1, 2, or 3."))
        end
    else
        if length(chunksizes) == 1 # Figure out which type of DiffCache is needed
            diffcache = dualcache(cache, Val{chunksizes[1]})
            fnew      = function _f_oop1(cache::LCN, x1::C1N, x2::C2N,
                                         x3::C3N, x4::C4N, select::Tuple{Int, Int}) where {LCN <: DiffCache,
                                                                                           C1N <: AbstractArray{<: Number},
                                                                                           C2N <: AbstractArray{<: Number},
                                                                                           C3N <: AbstractArray{<: Number},
                                                                                           C4N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, x4, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                target_cache .= f(x1, x2, x3, x4)
                return target_cache
            end
        elseif length(chunksizes) == 2
            diffcache = twodualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]})
            fnew      = function _f_oop2(cache::LCN, x1::C1N, x2::C2N,
                                         x3::C3N, x4::C4N, select::Tuple{Int, Int}) where {LCN <: TwoDiffCache,
                                                                                           C1N <: AbstractArray{<: Number},
                                                                                           C2N <: AbstractArray{<: Number},
                                                                                           C3N <: AbstractArray{<: Number},
                                                                                           C4N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, x4, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                target_cache .= f(x1, x2, x3, x4)
                return target_cache
            end
        elseif length(chunksizes) == 3
            diffcache = threedualcache(cache, Val{chunksizes[1]}, Val{chunksizes[2]}, Val{chunksizes[3]})
            fnew      = function _f_oop3(cache::LCN, x1::C1N, x2::C2N,
                                         x3::C3N, x4::C4N, select::Tuple{Int, Int}) where {LCN <: ThreeDiffCache,
                                                                                           C1N <: AbstractArray{<: Number},
                                                                                           C2N <: AbstractArray{<: Number},
                                                                                           C3N <: AbstractArray{<: Number},
                                                                                           C4N <: AbstractArray{<: Number}}
                target_cache = get_tmp(cache, x1, x2, x3, x4, select)
                if size(target_cache) != size(cache.du)
                    target_cache = reshape(target_cache, size(cache.du))
                end
                target_cache .= f(x1, x2, x3, x4)
                return target_cache
            end
        else
            throw(MethodError("The length of the eighth input argument, chunksizes, must be 1, 2, or 3."))
        end
    end
    return RALF4(fnew, f, diffcache)
end

function RALF4(fin::LC) where {LC <: AbstractArray{<: Number}}
    f(cache::LCN, x1::C1N, x2::C2N, x3::C3N, x4::C4N, select::Tuple{Int, Int}) where {LCN <: AbstractArray{<: Number}, C1N <: AbstractArray{<: Number}, C2N <: AbstractArray{<: Number}, C3N <: AbstractArray{<: Number}, C4N <: AbstractArray{<: Number}} = cache
    return RALF4{LC}(f, x -> fin, fin)
end

# Using the default of (1, 1) for `select` is important. This way, we can use autodiff
# during the homotopy algorithm without requiring additional arguments when calling `update!`
# to ensure the correct cache is used.
function (ralf::RALF4)(x1::C1, x2::C2, x3::C3,
                       x4::C4, select::Tuple{Int, Int} = (1, 1)) where {C1 <: AbstractArray{<: Number}, C2 <: AbstractArray{<: Number},
                                                                        C3 <: AbstractArray{<: Number}, C4 <: AbstractArray{<: Number}}
    return ralf.f(ralf.cache, x1, x2, x3, x4, select)
end
