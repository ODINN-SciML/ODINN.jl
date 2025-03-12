export TrainingStats

"""
    mutable struct TrainingStats

An object with the information of the training. 

# Fields
- `retcode`: Report code of the optimization.
- `losses`: Vector storing the value of the loss function at each iteration. 
- `niter`: Total number of iterations/epochs.
- `θ`: Parameters of neural network after training
"""
mutable struct TrainingStats{F <: AbstractFloat, I <: Int}
    retcode::Union{String, Nothing}
    losses::Vector{F}
    niter::I
    θ::Union{ComponentVector, Nothing}
end 

"""
    function TrainingStats(;
        retcode::Union{String, Nothing} = nothing,
        losses::Vector{F} = Float64[],
        niter::I = 0
    ) where {F <: AbstractFloat, I <: Int}

Constructor for TrainingStats object used to store important information during training. 

# Arguments
- `retcode`: Report code of the optimization.
- `losses`: Vector storing the value of the loss function at each iteration. 
- `niter`: Total number of iterations/epochs.
- `θ`: Parameters of neural network after training
"""
function TrainingStats(;
    retcode::Union{String, Nothing} = nothing,
    losses::Vector{F} = Float64[],
    niter::I = 0,
    θ::Union{ComponentVector, Nothing} = nothing
) where {F <: AbstractFloat, I <: Int}

    @assert length(losses) == niter

    training_stats = TrainingStats(retcode, losses, niter, θ)

    return training_stats
end