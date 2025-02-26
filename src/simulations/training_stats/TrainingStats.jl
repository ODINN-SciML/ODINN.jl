export TrainingStats

mutable struct TrainingStats{F <: AbstractFloat, I <: Int}
    retcode::Union{String, Nothing}
    losses::Union{Vector{F}, Nothing}
    niter::Union{I, Nothing}
end 

function TrainingStats(;
    retcode::Union{String, Nothing} = nothing,
    losses::Union{Vector{F}, Nothing} = Float64[],
    niter::Union{I, Nothing} = 0
) where {F <: AbstractFloat, I <: Int}

    @assert length(losses) == niter

    training_stats = TrainingStats(retcode, losses, niter)

    return training_stats
end