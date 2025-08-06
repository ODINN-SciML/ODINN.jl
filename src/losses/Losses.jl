export AbstractLoss
export L2Sum, LogSum
export LossH, LossV, LossHV
export loss, backward_loss

# Abstract type as a parent type for all losses
abstract type GeneralAbstractLoss end

# Basic losses whose purpose is to be used within other losses
abstract type AbstractSimpleLoss <: GeneralAbstractLoss end

# More advanced losses that are used in the code
abstract type AbstractLoss <: GeneralAbstractLoss end

"""
    L2Sum{I <: Integer} <: AbstractSimpleLoss

Struct that defines an L2 sum loss.

# Fields
- `distance::I`: Distance to border
"""
@kwdef struct L2Sum{I <: Integer} <: AbstractSimpleLoss
    distance::I = 3
end

"""
    LogSum{I <: Integer, F <: AbstractFloat} <: AbstractSimpleLoss

Struct that defines an Logaritmic sum loss.

# Fields
- `distance::I`: Distance to borde
- `ϵ::F`: Epsilon used inside the loss function representing the minimum velocity
"""
@kwdef struct LogSum{I <: Integer, F <: AbstractFloat} <: AbstractSimpleLoss
    distance::I = 3
    ϵ::F = 0.1
end

"""
    LossH{L <: AbstractSimpleLoss} <: AbstractLoss

Struct that defines the ice thickness loss.

# Fields
- `loss::L`: Type of loss to use for the ice thickness. Default is `L2Sum()`
"""
@kwdef struct LossH{L <: AbstractSimpleLoss} <: AbstractLoss
    loss::L = L2Sum()
end

"""
    LossV{L <: AbstractSimpleLoss} <: AbstractLoss

Struct that defines the ice velocity loss with an optional weighting coefficient.

# Fields
- `loss::L`: Type of loss to use for the ice velocity. Default is `L2Sum()`
- `scale_loss::Bool`: Whether to scale the loss function with the reference ice
    velocity magnitude.
"""
@kwdef struct LossV{L <: AbstractSimpleLoss} <: AbstractLoss
    loss::L = L2Sum()
    component::Symbol = :xy
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

### Definition of SimpleLoss functions

function loss(
    lossType::L2Sum,
    a::Matrix{F},
    b::Matrix{F};
    normalization::F=1.,
) where {F <: AbstractFloat}
    return loss(lossType, a, b, is_in_glacier(b, lossType.distance); normalization=normalization)
end
function backward_loss(
    lossType::L2Sum,
    a::Matrix{F},
    b::Matrix{F};
    normalization::F,
) where {F <: AbstractFloat}
    return backward_loss(lossType, a, b, is_in_glacier(b, lossType.distance); normalization=normalization)
end

function loss(
    lossType::L2Sum,
    a::Matrix{F},
    b::Matrix{F},
    mask::BitMatrix;
    normalization::F,
) where {F <: AbstractFloat}
    return sum(((a .- b)[mask]).^2)/normalization
end
function backward_loss(
    lossType::L2Sum,
    a::Matrix{F},
    b::Matrix{F},
    mask::BitMatrix;
    normalization::F,
) where {F <: AbstractFloat}
    d = zero(a)
    d[mask] = a[mask] .- b[mask]
    return 2.0 .* d ./ normalization
end

function loss(
    lossType::L2Sum,
    a::Vector{Matrix{F}},
    b::Vector{Matrix{F}};
    normalization::F=1.
) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [
        loss(
            lossType,
            ai,
            bi,
            is_in_glacier(bi, lossType.distance);
            normalization = normalization,
            )
        for (ai, bi) in zip(a, b)
        ]
    # return [sum(( (ai .- bi)[is_in_glacier(bi, lossType.distance)] ).^2)/normalization for (ai,bi) in zip(a,b)]
end
function backward_loss(
    lossType::L2Sum,
    a::Vector{Matrix{F}},
    b::Vector{Matrix{F}};
    normalization::F=1.,
) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [
        backward_loss(
            lossType,
            ai,
            bi;
            normalization=normalization,
            )
        for (ai, bi) in zip(a, b)
        ]
end

### Definition of Loss functions

"""
    function loss(
        lossType::LogSum,
        a::Matrix{F},
        b::Matrix{F},
        mask::BitMatrix;
        normalization::F=1.,
    ) where {F <: AbstractFloat}

Compute logaritmic loss function for ice velocity fields following Morlighem, M. et al.,
"Spatial patterns of basal drag inferred using control methods from a full-Stokes and
simpler models for Pine Island Glacier, West Antarctica". Geophys. Res. Lett. 37, (2010).
Given a minimum velocity ϵ the absolute velocity given by a and b, it computes the sum of

    log^2( (a + ϵ) / (b + ϵ) )

It has been shown that this loss function enables robust estimation of drag coefficient.
"""
function loss(
    lossType::LogSum,
    a::Matrix{F},
    b::Matrix{F},
    mask::BitMatrix;
    normalization::F=1.,
) where {F <: AbstractFloat}
    @assert (minimum(a) >= 0.0) & (minimum(b) >= 0.0)
    return sum((log.((a .+ lossType.ϵ) ./ (b .+ lossType.ϵ)).^2)[mask]) ./ normalization
end
function backward_loss(
    lossType::LogSum,
    a::Matrix{F},
    b::Matrix{F},
    mask::BitMatrix;
    normalization::F,
) where {F <: AbstractFloat}
    d = zero(a)
    d[mask] = log.((a[mask] .+ lossType.ϵ) ./ (b[mask] .+ lossType.ϵ)) ./ (b[mask] .+ lossType.ϵ)
    return 2.0 .* d ./ normalization
end

function loss(
    lossType::LogSum,
    a::Matrix{F},
    b::Matrix{F};
    normalization::F=1.,
) where {F <: AbstractFloat}
    @assert (minimum(a) >= 0.0) & (minimum(b) >= 0.0)
    return sum(log.((a .+ lossType.ϵ) ./ (b .+ lossType.ϵ)).^2)
end
function backward_loss(
    lossType::LogSum,
    a::Matrix{F},
    b::Matrix{F};
    normalization::F,
) where {F <: AbstractFloat}
    return 2.0 .* log.((a .+ lossType.ϵ) ./ (b .+ lossType.ϵ)) ./ (b .+ lossType.ϵ)
end

function loss(
    lossType::LossH,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    lH = loss(lossType.loss, H_pred, H_ref; normalization=normalization)
    return lH
end
function backward_loss(
    lossType::LossH,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    ∂L∂H = backward_loss(lossType.loss, H_pred, H_ref; normalization=normalization)
    ∂L∂θ = zero(θ)
    return ∂L∂H, ∂L∂θ
end

function loss(
    lossType::LossV,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    @assert !isnothing(glacier.velocityData)

    # 1- Retrieve the reference velocity Vx_ref, Vy_ref, V_ref
    # Note: This should be do just once in glacier, no in every call of the function
    Vx_ref, Vy_ref, V_ref, useVel = mapVelocity(
        simulation.parameters.simulation.mapping,
        glacier.velocityData,
        t,
    )

    if !useVel
        @info "Discarding reference data for t=$(t)"
        return 0.0
    end

    # 2- Compute the predicted velocity Vx_pred, Vy_pred, V_pred
    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx_pred, Vy_pred, V_pred = Huginn.V_from_H(simulation, H_pred, t, θ)
    # TODO: in the future we should dispatch wrt the iceflow model

    mask = is_in_glacier(H_ref, lossType.loss.distance) .& (V_ref .> 0.0)

    ℓ = 0.0

    if lossType.component == :xy
        vxErr = loss(lossType.loss, Vx_pred, Vx_ref, mask; normalization=normalization)
        vyErr = loss(lossType.loss, Vy_pred, Vy_ref, mask; normalization=normalization)
        ℓ += vxErr + vyErr
    elseif lossType.component == :abs
        vabsErr = loss(lossType.loss, V_pred, V_ref, mask; normalization=normalization)
        ℓ += vabsErr
    else
        @error "Loss type not implemented."
    end

    if lossType.scale_loss
        normVref = mean(Vx_ref[mask].^2 .+ Vy_ref[mask].^2)^0.5
        ℓ /= normVref
    end

    return ℓ
end
function backward_loss(
    lossType::LossV,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    @assert !isnothing(glacier.velocityData)

    # 1- Retrieve the reference velocity Vx_ref, Vy_ref, V_ref
    Vx_ref, Vy_ref, V_ref, useVel = mapVelocity(
        simulation.parameters.simulation.mapping,
        glacier.velocityData,
        t,
    )

    if !useVel
        @info "Discarding reference data for t=$(t)"
        return nothing, nothing
    end

    # 2- Compute the predicted velocity Vx_pred, Vy_pred, V_pred
    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx_pred, Vy_pred, V_pred = Huginn.V_from_H(simulation, H_pred, t, θ)
    # TODO: in the future we should dispatch wrt the iceflow model

    mask = is_in_glacier(H_ref, lossType.loss.distance) .& (V_ref .> 0.0)

    if lossType.component == :xy
        ∂lV∂Vx = backward_loss(lossType.loss, Vx_pred, Vx_ref, mask; normalization=normalization)
        ∂lV∂Vy = backward_loss(lossType.loss, Vy_pred, Vy_ref, mask; normalization=normalization)
    elseif lossType.component == :abs
        ∂lV∂V = backward_loss(lossType.loss, V_pred, V_ref, mask; normalization=normalization)
        ∂lV∂Vx, ∂lV∂Vy = zero(∂lV∂V), zero(∂lV∂V)
        ∂lV∂Vx = ifelse.(mask, ∂lV∂V .* (Vx_pred .- Vx_ref) ./ (V_pred .- V_ref), 0.0)
        ∂lV∂Vy = ifelse.(mask, ∂lV∂V .* (Vy_pred .- Vy_ref) ./ (V_pred .- V_ref), 0.0)
    end

    if lossType.scale_loss
        normVref = mean(Vx_ref[mask].^2 .+ Vy_ref[mask].^2)^0.5
        ∂lV∂Vx /= normVref
        ∂lV∂Vy /= normVref
    end

    ∂L∂H = VJP_λ_∂surface_V∂H(simulation.parameters.UDE.grad.VJP_method, ∂lV∂Vx, ∂lV∂Vy, H_pred, θ, simulation, t)[1]
    ∂L∂θ = VJP_λ_∂surface_V∂θ(simulation.parameters.UDE.grad.VJP_method, ∂lV∂Vx, ∂lV∂Vy, H_pred, θ, simulation, t)[1]

    return ∂L∂H, ∂L∂θ
end

function loss(
    lossType::LossHV,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    lH = loss(lossType.hLoss, H_pred, H_ref, t, glacier, θ, simulation; normalization=normalization)
    lV = loss(lossType.vLoss, H_pred, H_ref, t, glacier, θ, simulation; normalization=normalization)
    return lH + lossType.scaling * lV
end
function backward_loss(
    lossType::LossHV,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    ∂lH∂H, ∂lH∂θ = backward_loss(lossType.hLoss, H_pred, H_ref, t, glacier, θ, simulation; normalization=normalization)
    ∂lV∂H, ∂lV∂θ = backward_loss(lossType.vLoss, H_pred, H_ref, t, glacier, θ, simulation; normalization=normalization)
    ∂L∂H = isnothing(∂lV∂H) ? ∂lH∂H : ∂lH∂H + lossType.scaling * ∂lV∂H
    ∂L∂θ = if isnothing(∂lV∂θ)
        ∂lH∂θ
    elseif isnothing(∂lH∂θ)
        lossType.scaling * ∂lV∂θ
    else
        ∂lH∂θ + lossType.scaling * ∂lV∂θ
    end
    return ∂L∂H, ∂L∂θ
end
