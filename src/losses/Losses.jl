export AbstractLoss, L2Sum, LossHV
export loss, backward_loss

# Abstract type as a parent type for all losses
abstract type AbstractLoss end

@kwdef struct L2Sum{I <: Integer} <: AbstractLoss
    distance::I = 3
end

@kwdef struct LossHV{I <: Integer, F <: AbstractFloat, LH <: AbstractLoss, LV <: AbstractLoss} <: AbstractLoss
    hLoss::LH = L2Sum()
    vLoss::LV = L2Sum()
    scaling::F = 1.0
end

function loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}; normalization::F=1.) where {F <: AbstractFloat}
    return loss(lossType, a, b, is_in_glacier(b, lossType.distance); normalization=normalization)
end
function backward_loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}; normalization::F) where {F <: AbstractFloat}
    return backward_loss(lossType, a, b, is_in_glacier(b, lossType.distance); normalization=normalization)
end

function loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}, mask::Matrix{BitMatrix}; normalization::F) where {F <: AbstractFloat}
    return sum(((a .- b)[mask]).^2)/normalization
end
function backward_loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}, mask::Matrix{BitMatrix}; normalization::F) where {F <: AbstractFloat}
    d = zero(a)
    d[mask] = a[mask] .- b[mask]
    return 2.0.*d./normalization
end

function loss(lossType::L2Sum, a::Vector{Matrix{F}}, b::Vector{Matrix{F}}; normalization::F=1.) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [sum(( (ai .- bi)[is_in_glacier(bi, lossType.distance)] ).^2)/normalization for (ai,bi) in zip(a,b)]
end
function backward_loss(lossType::L2Sum, a::Vector{Matrix{F}}, b::Vector{Matrix{F}}; normalization::F=1.) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [backward_loss(lossType, ai, bi; normalization=normalization) for (ai,bi) in zip(a,b)]
end


function lossV(
    lossType::L2SumWithinGlacier,
    scale_loss::Bool,
    Vx_pred::Matrix{F},
    Vy_pred::Matrix{F},
    V_pred::Matrix{F},
    Vx_ref::Matrix{F},
    Vy_ref::Matrix{F},
    V_ref::Matrix{F},
    H_ref::Matrix{F};
    normalization::F=1.
) where {F <: AbstractFloat}
    # TODO: implement time selection
    ind = is_in_glacier(H_ref, lossType.distance) .& (V_ref .> 0.0)
    if scale_loss
        normVref = mean(Vx_ref[ind].^2 .+ Vy_ref[ind].^2)^0.5
        l_V_loc = sum( ((Vx_pred[ind] .- Vx_ref[ind])).^2 + ((Vy_pred[ind] .- Vy_ref[ind])).^2 )/normalization
        return normVref^(-1) * l_V_loc
    else
        ind = is_in_glacier(H_ref, lossType.distance)
        return sum(((V_pred[ind] .- V_ref[ind])).^2)/normalization
    end
end

function loss(
    lossType::LossHV,
    scale_loss::Bool,
    H_pred::Matrix{F},
    Vx_pred::Matrix{F},
    Vy_pred::Matrix{F},
    V_pred::Matrix{F},
    H_ref::Matrix{F},
    Vx_ref::Matrix{F},
    Vy_ref::Matrix{F},
    V_ref::Matrix{F};
    normalization::F=1.,
) where {F <: AbstractFloat}
    lH = loss(LossHV.hLoss, H_pred, H_ref; normalization=normalization)
    mask = is_in_glacier(H_ref, lossType.distance) .& (V_ref .> 0.0)
    if scale_loss
        normVref = mean(Vx_ref[mask].^2 .+ Vy_ref[mask].^2)^0.5
        vxErr = loss(LossHV.vLoss, Vx_pred, Vx_ref, mask; normalization=normalization)
        vyErr = loss(LossHV.vLoss, Vy_pred, Vy_ref, mask; normalization=normalization)
        lV = normVref^(-1) * (vxErr + vyErr)
    else
        lV = loss(LossHV.vLoss, V_pred, V_ref, mask; normalization=normalization)
    end
    return lH, lV
end
function backward_loss(
    lossType::LossHV,
    ∂lossH::F,
    ∂lossV::F,
    H_ref::Matrix{F};
    normalization::F=1.,
) where {F <: AbstractFloat}
    @error "Backward of LossHV not implemented yet."
    backward_loss(LossHV.hLoss, H_pred, H_ref; normalization=normalization)
end



# TODO: add unit tests

