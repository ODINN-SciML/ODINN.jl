export Hyperparameters

"""
    mutable struct Hyperparameters{F <: AbstractFloat, I <: Int} <: AbstractParameters

A mutable struct that holds hyperparameters for training a machine learning model.

# Keyword arguments
- `current_epoch::I`: The current epoch number.
- `current_minibatch::I`: The current minibatch number.
- `loss_history::Vector{F}`: A vector storing the history of loss values.
- `optimizer::Union{Optim.FirstOrderOptimizer, Flux.Optimise.AbstractOptimiser, Optimisers.AbstractRule}`: The optimizer used for training.
- `loss_epoch::F`: The loss value for the current epoch.
- `epochs::I`: The total number of epochs for training.
- `batch_size::I`: The size of each minibatch.
"""
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
    Hyperparameters(; current_epoch::Int64 = 1, current_minibatch::Int64 = 1, loss_history::Vector{Float64} = Vector{Float64}(), optimizer::Union{Optim.FirstOrderOptimizer, Flux.Optimise.AbstractOptimiser, Optimisers.AbstractRule} = BFGS(initial_stepnorm=0.001), loss_epoch::Float64 = 0.0, epochs::Int64 = 50, batch_size::Int64 = 15)

Constructs a `Hyperparameters` object with the specified parameters.

# Arguments
- `current_epoch::Int64`: The current epoch number. Defaults to 1.
- `current_minibatch::Int64`: The current minibatch number. Defaults to 1.
- `loss_history::Vector{Float64}`: A vector to store the history of loss values. Defaults to an empty vector.
- `optimizer::Union{Optim.FirstOrderOptimizer, Flux.Optimise.AbstractOptimiser, Optimisers.AbstractRule}`: The optimizer to be used. Defaults to `BFGS(initial_stepnorm=0.001)`.
- `loss_epoch::Float64`: The loss value for the current epoch. Defaults to 0.0.
- `epochs::Int64`: The total number of epochs. Defaults to 50.
- `batch_size::Int64`: The size of each minibatch. Defaults to 15.

# Returns
- A `Hyperparameters` object initialized with the provided values.
"""
function Hyperparameters(;
            current_epoch::Int64 = 1,
            current_minibatch::Int64 = 1,
            loss_history::Vector{Float64} = Vector{Float64}(),
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

