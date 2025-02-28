export TrainingStats

"""
    mutable struct TrainingStats

An object with the information of the training. 

# Fields
- `retcode`: Report code of the optimization.
- `losses`: Vector storing the value of the loss function at each iteration. 
- `niter`: Total number of iterations/epochs.
"""
mutable struct TrainingStats{F <: AbstractFloat, I <: Int}
    retcode::Union{String, Nothing}
    losses::Vector{F}
    niter::I
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
"""
function TrainingStats(;
    retcode::Union{String, Nothing} = nothing,
    losses::Vector{F} = Float64[],
    niter::I = 0
) where {F <: AbstractFloat, I <: Int}

    @assert length(losses) == niter

    training_stats = TrainingStats(retcode, losses, niter)

    return training_stats
end