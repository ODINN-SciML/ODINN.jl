
@kwdef mutable struct Hyperparameters{F <: AbstractFloat, I <: Int}
    current_epoch::I
    current_minibatch::I
    loss_history::Vector{F}
    optimizer::Union{Optim.FirstOrderOptimizer, Flux.Optimise.AbstractOptimiser}
    epochs::I
    batch_size::I
end

"""
    Hyperparameters(;
        current_epoch::Int64 = nothing,
        current_minibatch::Int64 = nothing,
        loss_history::Vector{Float64} = Vector{Float64}[],
        optimizer::Optim.FirstOrderOptimizer = BFGS(initial_stepnorm=0.001),
        epochs::Int64 = 50,
        batch_size::Int64 = 15
        )
Initialize the hyperparameters of a machine learning model (`Machine`).
Keyword arguments
=================
    - `current_epoch`: Current epoch in training
    - `current_minibatch`: Current minibatch in training
    - `loss_history`: `Vector` storing the loss for each epoch during training
    - `optimizer`: Optimizer to be used for training. Currently supports both `Optim.jl` and `Flux.jl` optimisers.
    - `epochs`: Number of epochs for the training
    - `batch_size`: Batch size for the training
"""
function Hyperparameters(;
            current_epoch::Int64 = nothing,
            current_minibatch::Int64 = nothing,
            loss_history::Vector{Float64} = Vector{Float64}[],
            optimizer::Optim.FirstOrderOptimizer = BFGS(initial_stepnorm=0.001),
            epochs::Int64 = 50,
            batch_size::Int64 = 15
            )
    # Build Hyperparameters based on input values
    hyperparameters = Hyperparameters(current_epoch, current_minibatch,
                                    loss_history, optimizer,
                                    epochs, batch_size)

    return hyperparameters
end