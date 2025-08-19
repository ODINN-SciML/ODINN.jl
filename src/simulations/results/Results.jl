export TrainingStats, Results

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

"""
    mutable struct Results{RES <: Sleipnir.Results, STAT <: TrainingStats}

Mutable struct containing the results of a functional inversion.
This object stores both the results of the optimization and the simulation results of the forward evaluations using the optimized variables.

# Fields
- `simulation::Vector{RES}`: Vector of `Sleipnir.Results` representing the results of a forward evaluation for each glacier.
- `stats::STAT`: Training statistics gathered during the optimization.

    function Results(
        simulation::Vector{<: Sleipnir.Results},
        stats::TrainingStats,
    )

Constructor for the `Results` object used to track statistics during training and the results of the forward evaluations simulated with the optimized variables.

# Arguments
- `simulation::Vector{<: Sleipnir.Results}`: Vector of `Sleipnir.Results` associated to the forward simulation of each glacier.
- `stats::TrainingStats`: Training statistics.
"""
mutable struct Results{RES <: Sleipnir.Results, STAT <: TrainingStats}
    simulation::Vector{RES}
    stats::STAT

    function Results(
        simulation::Vector{<: Sleipnir.Results},
        stats::TrainingStats,
    )
        return new{eltype(simulation), typeof(stats)}(simulation, stats)
    end
end
