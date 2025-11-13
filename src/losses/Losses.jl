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
The sum is defined only on pixels inside the glacier.
The parameter `distance` controls the pixels that should be used to compute the sum.
In order for a pixel to be used, it should be at least at `distance` from the glacier border.
The mask defining the glacier borders are computed using the ground truth ice thickness.

``loss(a,b) = sum_{i\\in\\text{mask}} (a[i]-b[i])^2 / normalization``

# Fields
- `distance::I`: Distance to border.
"""
@kwdef struct L2Sum{I <: Integer} <: AbstractSimpleLoss
    distance::I = 3
end

"""
    LogSum{I <: Integer, F <: AbstractFloat} <: AbstractSimpleLoss

Struct that defines a Logarithmic sum loss.

``loss(a,b) = log^2( (a + ϵ) / (b + ϵ) ) / normalization``

# Fields
- `distance::I`: Distance to border.
- `ϵ::F`: Epsilon used inside the loss function to handle division by zero and log(0).
    It somehow represents the minimum value the loss function should be sensible to.
"""
@kwdef struct LogSum{I <: Integer, F <: AbstractFloat} <: AbstractSimpleLoss
    distance::I = 3
    ϵ::F = 0.1
end

"""
    LossH{L <: AbstractSimpleLoss} <: AbstractLoss

Struct that defines the ice thickness loss.

# Fields
- `loss::L`: Type of loss to use for the ice thickness. Default is `L2Sum()`.
"""
@kwdef struct LossH{L <: AbstractSimpleLoss} <: AbstractLoss
    loss::L = L2Sum()
end

"""
    LossV{L <: AbstractSimpleLoss} <: AbstractLoss

Struct that defines the ice velocity loss.

# Fields
- `loss::L`: Type of loss to use for the ice velocity. Default is `L2Sum()`.
- `component::Symbol`: Component of the velocity field used in the loss.
    Options include :xy for both x and y component, and :abs for the norm/magnitude of the velocity.
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
It consists of two fields that define the ice thickness and ice velocity loss.
It also has a scaling coefficient that balances the ice velocity term in the loss.

``loss(\\hat H,H) = loss_H(\\hat H,H) + scaling * loss_V(\\hat V,V)``

with ``\\hat V`` computed from ``\\hat H`` for the SIA.

# Fields
- `hLoss::LH`: Type of loss to use for the ice thickness. Default is `LossH()`.
- `vLoss::LV`: Type of loss to use for the ice velocity. Default is `LossV()`.
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

### Definition of simple loss functions

function loss(
    lossType::L2Sum,
    a::Matrix{F},
    b::Matrix{F},
    normalization::F,
) where {F <: AbstractFloat}
    return loss(lossType, a, b, is_in_glacier(b, lossType.distance), normalization)
end
function backward_loss(
    lossType::L2Sum,
    a::Matrix{F},
    b::Matrix{F},
    normalization::F,
) where {F <: AbstractFloat}
    return backward_loss(lossType, a, b, is_in_glacier(b, lossType.distance), normalization)
end

function loss(
    lossType::L2Sum,
    a::Matrix{F},
    b::Matrix{F},
    mask::BitMatrix,
    normalization::F,
) where {F <: AbstractFloat}
    return sum(((a .- b)[mask]).^2)/normalization
end
function backward_loss(
    lossType::L2Sum,
    a::Matrix{F},
    b::Matrix{F},
    mask::BitMatrix,
    normalization::F,
) where {F <: AbstractFloat}
    d = zero(a)
    d[mask] = a[mask] .- b[mask]
    return 2.0 .* d ./ normalization
end

function loss(
    lossType::L2Sum,
    a::Vector{Matrix{F}},
    b::Vector{Matrix{F}},
    normalization::F,
) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [
        loss(
            lossType,
            ai,
            bi,
            is_in_glacier(bi, lossType.distance),
            normalization,
            )
        for (ai, bi) in zip(a, b)
        ]
    # return [sum(( (ai .- bi)[is_in_glacier(bi, lossType.distance)] ).^2)/normalization for (ai,bi) in zip(a,b)]
end
function backward_loss(
    lossType::L2Sum,
    a::Vector{Matrix{F}},
    b::Vector{Matrix{F}},
    normalization::F,
) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [
        backward_loss(
            lossType,
            ai,
            bi,
            normalization,
            )
        for (ai, bi) in zip(a, b)
        ]
end

### Definition of more complete loss functions

"""
    function loss(
        lossType::LogSum,
        a::Matrix{F},
        b::Matrix{F},
        mask::BitMatrix,
        normalization::F,
    ) where {F <: AbstractFloat}

Compute logarithmic loss function for ice velocity fields following Morlighem, M. et al.,
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
    mask::BitMatrix,
    normalization::F,
) where {F <: AbstractFloat}
    @assert (minimum(a) >= 0.0) & (minimum(b) >= 0.0)
    return sum((log.((a[mask] .+ lossType.ϵ) ./ (b[mask] .+ lossType.ϵ)).^2)) ./ normalization
end
function backward_loss(
    lossType::LogSum,
    a::Matrix{F},
    b::Matrix{F},
    mask::BitMatrix,
    normalization::F,
) where {F <: AbstractFloat}
    d = zero(a)
    d[mask] = log.((a[mask] .+ lossType.ϵ) ./ (b[mask] .+ lossType.ϵ)) ./ (a[mask] .+ lossType.ϵ)
    return 2.0 .* d ./ normalization
end

function loss(
    lossType::LogSum,
    a::Matrix{F},
    b::Matrix{F},
    normalization::F,
) where {F <: AbstractFloat}
    @assert (minimum(a) >= 0.0) & (minimum(b) >= 0.0)
    return sum(log.((a .+ lossType.ϵ) ./ (b .+ lossType.ϵ)).^2) ./ normalization
end
function backward_loss(
    lossType::LogSum,
    a::Matrix{F},
    b::Matrix{F},
    normalization::F,
) where {F <: AbstractFloat}
    return 2.0 .* log.((a .+ lossType.ϵ) ./ (b .+ lossType.ϵ)) ./ (a .+ lossType.ϵ) ./ normalization
end

function loss(
    lossType::LossH,
    H_pred::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}
    if isnothing(H_ref)
        # That time step has no valid ground truth ice thickness data, so the contribution is zero
        return 0.0
    else
        return loss(lossType.loss, H_pred, H_ref, normalization)*Δt.H
    end
end
function backward_loss(
    lossType::LossH,
    H_pred::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}
    ∂L∂θ = zero(θ)
    ∂L∂H = if isnothing(H_ref)
        # That time step has no valid ground truth ice thickness data, so the contribution is zero
        zero(H_pred)
    else
        backward_loss(lossType.loss, H_pred, H_ref, normalization)
    end
    return ∂L∂H*Δt.H, ∂L∂θ*Δt.H
end

function loss(
    lossType::LossV,
    H_pred::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}
    if isnothing(V_ref)
        # That time step has no valid ground truth ice surface velocity data, so the contribution is zero
        return 0.0
    end

    # Compute the predicted velocity Vx_pred, Vy_pred, V_pred
    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx_pred, Vy_pred, V_pred = Huginn.V_from_H(simulation, H_pred, t, θ)
    # TODO: in the future we should dispatch wrt the iceflow model

    mask = is_in_glacier(H_pred, lossType.loss.distance) .& (V_ref .> 0.0)

    ℓ = if lossType.component == :xy
        loss(lossType.loss, Vx_pred, Vx_ref, mask, normalization) + loss(lossType.loss, Vy_pred, Vy_ref, mask, normalization)
    elseif lossType.component == :abs
        loss(lossType.loss, V_pred, V_ref, mask, normalization)
    else
        @error "Loss type not implemented."
    end

    # Scale loss function
    ℓ_scale = if lossType.scale_loss
        ℓ / mean(Vx_ref[mask].^2 .+ Vy_ref[mask].^2)^0.5
    else
        ℓ
    end

    return ℓ_scale*Δt.V
end
function backward_loss(
    lossType::LossV,
    H_pred::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}
    if isnothing(V_ref)
        # That time step has no valid ground truth ice surface velocity data, so the contribution is zero
        return nothing, nothing
    end

    # Compute the predicted velocity Vx_pred, Vy_pred, V_pred
    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx_pred, Vy_pred, V_pred = Huginn.V_from_H(simulation, H_pred, t, θ)
    # TODO: in the future we should dispatch wrt the iceflow model

    mask = is_in_glacier(H_pred, lossType.loss.distance) .& (V_ref .> 0.0)

    if lossType.component == :xy
        ∂lV∂Vx = backward_loss(lossType.loss, Vx_pred, Vx_ref, mask, normalization)
        ∂lV∂Vy = backward_loss(lossType.loss, Vy_pred, Vy_ref, mask, normalization)
    elseif lossType.component == :abs
        ∂lV∂V = backward_loss(lossType.loss, V_pred, V_ref, mask, normalization)
        # ∂lV∂Vx, ∂lV∂Vy = zero(∂lV∂V), zero(∂lV∂V)
        ∂lV∂Vx = ifelse.(mask, ∂lV∂V .* (Vx_pred .- Vx_ref) ./ (V_pred .- V_ref), 0.0)
        ∂lV∂Vy = ifelse.(mask, ∂lV∂V .* (Vy_pred .- Vy_ref) ./ (V_pred .- V_ref), 0.0)
    end

    ∂lV∂Vx_scale = if lossType.scale_loss
        ∂lV∂Vx / mean(Vx_ref[mask].^2 .+ Vy_ref[mask].^2)^0.5
    else
        ∂lV∂Vx
    end
    ∂lV∂Vy_scale = if lossType.scale_loss
        ∂lV∂Vy / mean(Vx_ref[mask].^2 .+ Vy_ref[mask].^2)^0.5
    else
        ∂lV∂Vy
    end

    ∂L∂H = VJP_λ_∂surface_V∂H(simulation.parameters.UDE.grad.VJP_method, ∂lV∂Vx_scale, ∂lV∂Vy_scale, H_pred, θ, simulation, t)[1]
    ∂L∂θ = VJP_λ_∂surface_V∂θ(simulation.parameters.UDE.grad.VJP_method, ∂lV∂Vx_scale, ∂lV∂Vy_scale, H_pred, θ, simulation, t)[1]

    return ∂L∂H*Δt.V, ∂L∂θ*Δt.V
end

function loss(
    lossType::LossHV,
    H_pred::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}
    lH = loss(lossType.hLoss, H_pred, H_ref, V_ref, Vx_ref, Vy_ref, t, glacier_idx, θ, simulation, normalization, Δt)
    lV = loss(lossType.vLoss, H_pred, H_ref, V_ref, Vx_ref, Vy_ref, t, glacier_idx, θ, simulation, normalization, Δt)
    return lH*Δt.H + lossType.scaling * lV*Δt.V
end
function backward_loss(
    lossType::LossHV,
    H_pred::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}
    ∂lH∂H, ∂lH∂θ = backward_loss(lossType.hLoss, H_pred, H_ref, V_ref, Vx_ref, Vy_ref, t, glacier_idx, θ, simulation, normalization, Δt)
    ∂lV∂H, ∂lV∂θ = backward_loss(lossType.vLoss, H_pred, H_ref, V_ref, Vx_ref, Vy_ref, t, glacier_idx, θ, simulation, normalization, Δt)
    ∂L∂H = isnothing(∂lV∂H) ? ∂lH∂H : ∂lH∂H*Δt.H + lossType.scaling * ∂lV∂H*Δt.V
    ∂L∂θ = if isnothing(∂lV∂θ)
        ∂lH∂θ*Δt.H
    elseif isnothing(∂lH∂θ)
        lossType.scaling * ∂lV∂θ*Δt.V
    else
        ∂lH∂θ*Δt.H + lossType.scaling * ∂lV∂θ*Δt.V
    end
    return ∂L∂H, ∂L∂θ
end

loss_uses_velocity(lossType::LossH) = false
loss_uses_velocity(lossType::Union{LossV, LossHV}) = true
discreteLossSteps(lossType::AbstractLoss, tspan) = Vector{Float64}()
