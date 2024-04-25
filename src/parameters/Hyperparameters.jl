export Hyperparameters

  mutable struct Hyperparameters{F <: AbstractFloat, I <: Int} <: AbstractParameters
    current_epoch::I
    current_minibatch::I
    loss_history::Vector{F}
    optimizer::Union{Optim.FirstOrderOptimizer, Flux.Optimise.AbstractOptimiser, Optimisers.AbstractRule}
    loss_epoch::F
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
            current_epoch::Int64 = 1,
            current_minibatch::Int64 = 1,
            loss_history::Vector{Float64} = zeros(Float64, 0),
            optimizer::Union{Optim.FirstOrderOptimizer, Flux.Optimise.AbstractOptimiser, Optimisers.AbstractRule} = BFGS(initial_stepnorm=0.001),
            loss_epoch::Float64 = 0.0,
            epochs::Int64 = 50,
            batch_size::Int64 = 15
            )
    # Build Hyperparameters based on input values
    hyperparameters = Hyperparameters(current_epoch, current_minibatch,
                                    loss_history, optimizer, loss_epoch,
                                    epochs, batch_size)

    return hyperparameters
end

Base.:(==)(a::Hyperparameters, b::Hyperparameters) = a.current_epoch == b.current_epoch && a.current_minibatch == b.current_minibatch && a.loss_history == b.loss_history && 
                                      a.optimizer == b.optimizer && a.epochs == b.epochs && a.loss_epoch == b.loss_epoch && a.batch_size == b.batch_size 

