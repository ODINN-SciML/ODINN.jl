export MultiLoss
export loss, backward_loss

"""
"""
# struct MultiLoss{F <: AbstractFloat} <: AbstractLoss
struct MultiLoss{TL, TS} <: AbstractLoss
    losses::TL
    λs::TS
    # losses::Union{Tuple{Vararg{<:AbstractLoss}}, Nothing} = 
    # λs::Union{Tuple{Vararg{F}}, Nothing} = 
    function MultiLoss(;
        losses = (L2Sum(), ),
        λs = (1.0, ),
        )
        @assert length(losses) == length(λs) "You need to provide an hyperparameter for each loss term defined."
        # return new{Sleipnir.Float}(losses, λs)
        return new{typeof(losses), typeof(λs)}(losses, λs)
    end
end

function loss(
    lossType::MultiLoss,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    losses = map(sub_loss ->
        loss(
            sub_loss,
            H_pred,
            H_ref,
            t::F,
            glacier,
            θ,
            simulation;
            normalization = normalization
        ), lossType.losses
    )
    # Combine contribution of each loss
    return sum(lossType.λs .* losses)
end

function backward_loss(
    lossType::MultiLoss,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    res_backward_losses = map(sub_loss ->
        backward_loss(
            sub_loss,
            H_pred,
            H_ref,
            t::F,
            glacier,
            θ,
            simulation;
            normalization = normalization
        ), lossType.losses
    )
    # Combine contribution of each gradient
    ∂L∂Hs, ∂L∂θs = map(x -> collect(x), zip(res_backward_losses...))
    ∂L∂H = sum(lossType.λs .* ∂L∂Hs)
    ∂L∂θ = sum(lossType.λs .* ∂L∂θs)
    return ∂L∂H, ∂L∂θ
end