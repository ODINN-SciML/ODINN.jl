export AbstractLoss, L2Sum, lossH, lossV, LossHV
export loss, backward_loss

# Abstract type as a parent type for all losses
abstract type AbstractLoss end

"""
    L2Sum{I <: Integer} <: AbstractLoss

Struct that defines an L2 sum loss.
"""
@kwdef struct L2Sum{I <: Integer} <: AbstractLoss
    distance::I = 3
end

"""
    LossH{L <: AbstractLoss} <: AbstractLoss

Struct that defines the ice thickness loss.

# Fields
- `loss::L`: Type of loss to use for the ice thickness. Default is `L2Sum()`
"""
@kwdef struct LossH{L <: AbstractLoss} <: AbstractLoss
    loss::L = L2Sum()
end

"""
    LossV{L <: AbstractLoss} <: AbstractLoss

Struct that defines the ice velocity loss with an optional weighting coefficient.

# Fields
- `loss::L`: Type of loss to use for the ice velocity. Default is `L2Sum()`
- `scale_loss::Bool`: Whether to scale the loss function with the reference ice
    velocity magnitude.
"""
@kwdef struct LossV{L <: AbstractLoss} <: AbstractLoss
    loss::L = L2Sum()
    scale_loss::Bool = true
end

"""
    LossHV{
        F <: AbstractFloat,
        LH <: AbstractLoss,
        LV <: AbstractLoss,
    } <: AbstractLoss

Struct that defines the ice thickness and ice velocity loss.
It consists in two fields that define the ice thickness and ice velocity loss.
It also has a scaling coefficient that balances the ice velocity term in the loss.

# Fields
- `hLoss::LH`: Type of loss to use for the ice thickness. Default is `LossH()`
- `vLoss::LV`: Type of loss to use for the ice velocity. Default is `LossV()`
- `scaling::F`: Scaling of the ice velocity term. Default is `1`.
"""
@kwdef struct LossHV{
    F <: AbstractFloat,
    LH <: AbstractLoss,
    LV <: AbstractLoss,
} <: AbstractLoss
    hLoss::LH = LossH()
    vLoss::LV = LossV()
    scaling::F = 1.0
end

function loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}; normalization::F=1.) where {F <: AbstractFloat}
    return loss(lossType, a, b, is_in_glacier(b, lossType.distance); normalization=normalization)
end
function backward_loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}; normalization::F) where {F <: AbstractFloat}
    return backward_loss(lossType, a, b, is_in_glacier(b, lossType.distance); normalization=normalization)
end

function loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}, mask::BitMatrix; normalization::F) where {F <: AbstractFloat}
    return sum(((a .- b)[mask]).^2)/normalization
end
function backward_loss(lossType::L2Sum, a::Matrix{F}, b::Matrix{F}, mask::BitMatrix; normalization::F) where {F <: AbstractFloat}
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


function loss(
    lossType::LossH,
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
    lH = loss(lossType.loss, H_pred, H_ref; normalization=normalization)
    return lH, 0.0
end
function backward_loss(
    lossType::LossH,
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
    return backward_loss(lossType.loss, H_pred, H_ref; normalization=normalization), nothing, nothing, nothing
end

function loss(
    lossType::LossV,
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
    mask = is_in_glacier(H_ref, lossType.loss.distance) .& (V_ref .> 0.0)
    if lossType.scale_loss
        normVref = mean(Vx_ref[mask].^2 .+ Vy_ref[mask].^2)^0.5
        vxErr = loss(lossType.loss, Vx_pred, Vx_ref, mask; normalization=normalization)
        vyErr = loss(lossType.loss, Vy_pred, Vy_ref, mask; normalization=normalization)
        lV = normVref^(-1) * (vxErr + vyErr)
    else
        lV = loss(lossType.loss, V_pred, V_ref, mask; normalization=normalization)
    end
    return 0.0, lV
end
function backward_loss(
    lossType::LossV,
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
    mask = is_in_glacier(H_ref, lossType.loss.distance) .& (V_ref .> 0.0)
    ∂L∂Vx = nothing
    ∂L∂Vy = nothing
    ∂L∂V = nothing
    if lossType.scale_loss
        normVref = mean(Vx_ref[mask].^2 .+ Vy_ref[mask].^2)^0.5
        ∂L∂Vx = normVref^(-1) * backward_loss(lossType.loss, Vx_pred, Vx_ref, mask; normalization=normalization)
        ∂L∂Vy = normVref^(-1) * backward_loss(lossType.loss, Vy_pred, Vy_ref, mask; normalization=normalization)
        ∂L∂V = nothing
    else
        ∂L∂Vx = nothing
        ∂L∂Vy = nothing
        ∂L∂V = backward_loss(lossType.vLoss, V_pred, V_ref, mask; normalization=normalization)
    end
    return nothing, ∂L∂Vx, ∂L∂Vy, ∂L∂V
end

function loss(
    lossType::LossHV,
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
    lH = loss(lossType.hLoss, H_pred, Vx_pred, Vy_pred, V_pred, H_ref, Vx_ref, Vy_ref, V_ref; normalization=normalization)[1]
    lV = loss(lossType.vLoss, H_pred, Vx_pred, Vy_pred, V_pred, H_ref, Vx_ref, Vy_ref, V_ref; normalization=normalization)[2]
    return lH, lossType.scaling*lV
end
function backward_loss(
    lossType::LossHV,
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
    ∂L∂H, = backward_loss(lossType.hLoss, H_pred, Vx_pred, Vy_pred, V_pred, H_ref, Vx_ref, Vy_ref, V_ref; normalization=normalization)
    _, ∂L∂Vx, ∂L∂Vy, ∂L∂V = backward_loss(lossType.vLoss, H_pred, Vx_pred, Vy_pred, V_pred, H_ref, Vx_ref, Vy_ref, V_ref; normalization=normalization)
    if !isnothing(∂L∂Vx)
        ∂L∂Vx *= lossType.scaling
    end
    if !isnothing(∂L∂Vy)
        ∂L∂Vy *= lossType.scaling
    end
    if !isnothing(∂L∂V)
        ∂L∂V *= lossType.scaling
    end
    return ∂L∂H, ∂L∂Vx, ∂L∂Vy, ∂L∂V
end
