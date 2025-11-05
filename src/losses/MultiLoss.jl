export MultiLoss

"""
    MultiLoss(; losses = (L2Sum(),), λs = (1.0,))

Combines multiple loss functions into a single weighted objective.

`MultiLoss` enables composing several individual loss terms—each possibly representing
a different physical constraint, data fidelity term, or regularization penalty—into
a single differentiable loss function.

# Keyword Arguments (Constructor)
- `losses::Tuple = (L2Sum(),)`: A tuple of loss objects (each subtype of `AbstractLoss`) to be combined.
- `λs::Tuple = (1.0,)`:  A tuple of scalar weights or hyperparameters corresponding to each loss term.

# Fields (Struct)
- `losses::TL`: Tuple of loss functions.
- `λs::TS`: Tuple of weighting coefficients.
"""
struct MultiLoss{TL, TS} <: AbstractLoss
    losses::TL
    λs::TS
    function MultiLoss(;
        losses = (L2Sum(), ),
        λs = (1.0, ),
    )
        @assert length(losses) == length(λs) "You need to provide an hyperparameter for each loss term defined."
        λs = collect(λs)
        return new{typeof(losses), typeof(λs)}(losses, λs)
    end
end

"""
    loss(lossType::MultiLoss, H_pred::Matrix{F}, H_ref::Matrix{F}, t::F, glacier, θ, simulation, normalization::F) where {F<:AbstractFloat}

Computes the weighted composite loss for a prediction `H_pred` against a reference `H_ref`
using a `MultiLoss` object.

Each individual loss in `lossType.losses` is evaluated and multiplied by its corresponding
weight in `lossType.λs`. The final loss is the sum of these weighted contributions.

# Arguments
- `lossType::MultiLoss`: Composite loss object containing individual losses and weights.
- `H_pred::Matrix{F}`: Predicted ice thickness.
- `H_ref::Matrix{F}`: Reference ice thickness.
- `t::F`: Current time or simulation step.
- `glacier`: Glacier-specific data structure.
- `θ`: Model parameters used in the simulation.
- `simulation`: Simulation object providing necessary context for loss evaluation.
- `normalization::F`: Normalization factor applied within each individual loss.

# Returns
- `F`: The total scalar loss, computed as the sum of weighted individual losses.
"""
function loss(
    lossType::MultiLoss,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation,
    normalization::F,
) where {F <: AbstractFloat}
    losses = map(sub_loss ->
        loss(
            sub_loss,
            H_pred,
            H_ref,
            t,
            glacier,
            θ,
            simulation,
            normalization,
        ), lossType.losses
    )
    # Combine contribution of each loss
    return sum(lossType.λs .* losses)
end

"""
    backward_loss(lossType::MultiLoss, H_pred::Matrix{F}, H_ref::Matrix{F}, t::F, glacier, θ, simulation; normalization::F) where {F<:AbstractFloat}

Computes the gradient of a composite loss defined by a `MultiLoss` object
with respect to both the predicted field `H_pred` and model parameters `θ`.

Each sub-loss's backward gradient is weighted by its corresponding coefficient in `lossType.λs`
and summed to form the total gradient.

# Arguments
- `lossType::MultiLoss`: Composite loss object containing individual losses and weights.
- `H_pred::Matrix{F}`: Predicted ice thickness.
- `H_ref::Matrix{F}`: Reference ice thickness.
- `t::F`: Current time or simulation step.
- `glacier`: Glacier-specific data structure providing context for the loss.
- `θ`: Model parameters used in the simulation.
- `simulation`: Simulation object providing necessary context for gradient computation.
- `normalization::F`: Normalization factor applied within each individual loss.

# Returns
- `(∂L∂H, ∂L∂θ)`: Tuple containing:
  - `∂L∂H::Matrix{F}`: Gradient of the composite loss with respect to `H_pred`.
  - `∂L∂θ`: Gradient of the composite loss with respect to model parameters `θ`.
"""
function backward_loss(
    lossType::MultiLoss,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation,
    normalization::F,
) where {F <: AbstractFloat}
    res_backward_losses = map(sub_loss ->
        backward_loss(
            sub_loss,
            H_pred,
            H_ref,
            t,
            glacier,
            θ,
            simulation,
            normalization
        ), lossType.losses
    )
    # Combine contribution of each gradient
    ∂L∂Hs, ∂L∂θs = map(x -> collect(x), zip(res_backward_losses...))
    ∂L∂H = sum(lossType.λs .* ∂L∂Hs)
    ∂L∂θ = sum(lossType.λs .* ∂L∂θs)
    return ∂L∂H, ∂L∂θ
end

function loss_uses_ref_velocity(lossType::MultiLoss)
    return any(
        map(lossType.losses) do l
            loss_uses_ref_velocity(l)
        end
    )
end
