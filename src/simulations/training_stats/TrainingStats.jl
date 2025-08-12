export TrainingStats

"""
    mutable struct TrainingStats{F <: AbstractFloat, I <: Integer}

An object with the information of the training.

# Fields
- `retcode::Union{String, Nothing}`: Report code of the optimization.
- `losses::Vector{F}`: Vector storing the value of the loss function at each iteration.
- `niter::I`: Total number of iterations/epochs.
- `θ::Union{<: ComponentVector, Nothing}`: Parameters of neural network after training
- `θ_hist::Vector{<: ComponentVector}`: History of parameters of neural network during training
- `∇θ_hist::Vector{<: ComponentVector}`: History of gradients training
- `lastCall::DateTime`: Last time the callback diagnosis was called.
    This is used to compute the time per iteration.
"""
mutable struct TrainingStats{F <: AbstractFloat, I <: Integer}
    retcode::Union{String, Nothing}
    losses::Vector{F}
    niter::I
    θ::Union{<: ComponentVector, Nothing}
    θ_hist::Vector{<: ComponentVector}
    ∇θ_hist::Vector{<: ComponentVector}
    lastCall::DateTime
end

"""
    TrainingStats(;
        retcode::Union{String, Nothing} = nothing,
        losses::Vector{F} = Float64[],
        niter::I = 0,
        θ::Union{ComponentVector, Nothing} = nothing,
        θ_hist::Union{Vector{ComponentVector}, Nothing} = ComponentVector[],
        ∇θ_hist::Union{Vector{ComponentVector}, Nothing} = ComponentVector[]
    ) where {F <: AbstractFloat, I <: Integer}

Constructor for TrainingStats object used to store important information during training.

# Arguments
- `retcode`: Report code of the optimization.
- `losses`: Vector storing the value of the loss function at each iteration.
- `niter`: Total number of iterations/epochs.
- `θ`: Parameters of neural network after training
- `θ_hist`: History of parameters of neural network during training
- `∇θ_hist`: History of gradients training
"""
function TrainingStats(;
    retcode::Union{String, Nothing} = nothing,
    losses::Vector{F} = Float64[],
    niter::I = 0,
    θ::Union{ComponentVector, Nothing} = nothing,
    θ_hist::Union{Vector{ComponentVector}, Nothing} = ComponentVector[],
    ∇θ_hist::Union{Vector{ComponentVector}, Nothing} = ComponentVector[]
) where {F <: AbstractFloat, I <: Integer}

    @assert length(losses) == niter
    @assert length(θ_hist) == niter

    training_stats = TrainingStats{eltype(losses), typeof(niter)}(
        retcode, losses, niter, θ, θ_hist, ∇θ_hist, DateTime(0,1,1),
    )

    return training_stats
end
