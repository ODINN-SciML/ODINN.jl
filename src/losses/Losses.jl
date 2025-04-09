export AbstractLoss, L2Sum, L2SumWithinGlacier
export loss, backward_loss

# include("ML_utils.jl")

# Abstract type as a parent type for all losses
abstract type AbstractLoss end

@kwdef struct L2Sum <: AbstractLoss
end

@kwdef struct L2SumWithinGlacier{I <: Integer} <: AbstractLoss
    distance::I = 3
end

function loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}; normalization::F=1.) where {F <: AbstractFloat}
    return sum((a .- b).^2)/normalization
end
function backward_loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}; normalization::F=1.) where {F <: AbstractFloat}
    return 2*(a .- b)./normalization
end

function loss(lossType::L2SumWithinGlacier, a::Matrix{F}, b::Matrix{F}; normalization::F=1.) where {F <: AbstractFloat}
    return sum(((a .- b)[is_in_glacier(b, lossType.distance)]).^2)/normalization
end
function backward_loss(lossType::L2SumWithinGlacier, a::Matrix{F}, b::Matrix{F}; normalization::F) where {F <: AbstractFloat}
    d = zero(a)
    ind = is_in_glacier(b, lossType.distance)
    d[ind] = a[ind] .- b[ind]
    return 2.0.*d./normalization
end

function loss(lossType::L2Sum, a::Vector{Matrix{F}}, b::Vector{Matrix{F}}; normalization::F=1.) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [sum((ai .- bi).^2)/normalization for (ai,bi) in zip(a,b)]
end
function backward_loss(lossType::L2Sum, a::Vector{Matrix{F}}, b::Vector{Matrix{F}}; normalization::F=1.) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [backward_loss(lossType, ai, bi; normalization=normalization) for (ai,bi) in zip(a,b)]
end

function loss(lossType::L2SumWithinGlacier, a::Vector{Matrix{F}}, b::Vector{Matrix{F}}; normalization::F=1.) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [sum(( (ai .- bi)[is_in_glacier(bi, lossType.distance)] ).^2)/normalization for (ai,bi) in zip(a,b)]
end
function backward_loss(lossType::L2SumWithinGlacier, a::Vector{Matrix{F}}, b::Vector{Matrix{F}}; normalization::F=1.) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [backward_loss(lossType, ai, bi; normalization=normalization) for (ai,bi) in zip(a,b)]
end



# TODO: add unit tests

