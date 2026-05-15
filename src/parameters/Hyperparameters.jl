export Hyperparameters

"""
    mutable struct Hyperparameters{F <: AbstractFloat, I <: Integer} <: AbstractParameters

A mutable struct that holds hyperparameters for training a machine learning model.

# Keyword arguments

  - `current_epoch::I`: The current epoch number.
  - `current_minibatch::I`: The current minibatch number.
  - `loss_history::Vector{F}`: A vector storing the history of loss values.
  - `optimizer::Union{Optim.FirstOrderOptimizer, Optimisers.AbstractRule, Vector{Optim.FirstOrderOptimizer}, Vector{Any}}`: The optimizer used for training.
  - `loss_epoch::F`: The loss value for the current epoch.
  - `epochs::I`: The total number of epochs for training.
  - `batch_size::I`: The size of each minibatch.
"""
mutable struct Hyperparameters{F <: AbstractFloat, I <: Integer} <: AbstractParameters
    current_epoch::I
    current_minibatch::I
    loss_history::Vector{F}
    optimizer::Union{Optim.FirstOrderOptimizer, Optimisers.AbstractRule,
        Vector{Optim.FirstOrderOptimizer}, Vector{Any}}
    loss_epoch::F
    epochs::Union{I, Vector{I}}
    batch_size::I
end

"""
    Hyperparameters(;
        current_epoch::Int64 = 1,
        current_minibatch::Int64 = 1,
        loss_history::Vector{Float64} = Vector{Float64}(),
        optimizer::Union{Optim.FirstOrderOptimizer, Vector{Optim.FirstOrderOptimizer}, Vector{Any}} = BFGS(initial_stepnorm = 0.001),
        loss_epoch::Float64 = 0.0,
        epochs::Union{Int64, Vector{Int64}} = 50,
        batch_size::Int64 = 15

)

Constructs a `Hyperparameters` object with the specified parameters.

# Arguments

  - `current_epoch::Int64`: The current epoch number. Defaults to 1.
  - `current_minibatch::Int64`: The current minibatch number. Defaults to 1.
  - `loss_history::Vector{Float64}`: A vector to store the history of loss values. Defaults to an empty vector.
  - `optimizer::Union{Optim.FirstOrderOptimizer, Vector{Optim.FirstOrderOptimizer}, Vector{Any}}`: The optimizer to be used. Defaults to `BFGS(initial_stepnorm=0.001)`.
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
        optimizer::Union{Optim.FirstOrderOptimizer, Optimisers.AbstractRule,
            Vector{Optim.FirstOrderOptimizer}, Vector{Any}} = BFGS(initial_stepnorm = 0.001),
        loss_epoch::Float64 = 0.0,
        epochs::Union{Int64, Vector{Int64}} = 50,
        batch_size::Int64 = 15
)
    # Build Hyperparameters based on input values
    hyperparameters = Hyperparameters(current_epoch, current_minibatch,
        loss_history, optimizer, loss_epoch,
        epochs, batch_size)

    return hyperparameters
end

function Base.:(==)(a::Hyperparameters, b::Hyperparameters)
    a.current_epoch == b.current_epoch && a.current_minibatch == b.current_minibatch &&
        a.loss_history == b.loss_history &&
        a.optimizer == b.optimizer && a.epochs == b.epochs &&
        a.loss_epoch == b.loss_epoch &&
        a.batch_size == b.batch_size
end

# Display setup
Base.show(io::IO, ::MIME"text/plain", params::Hyperparameters) = Base.show(io, params)
function Base.show(io::IO, params::Hyperparameters)
    label(s) = printstyled(io, rpad(s, 11); color = :light_black)
    sep() = printstyled(io, " · "; color = :light_black)
    field(s) = printstyled(io, s; color = :light_black)
    val(s) = print(io, s)
    hint(s) = printstyled(io, s; color = :light_black)

    println(io, "Hyperparameters")

    # Training
    label("  Training")
    field("epochs");
    print(io, " = ")
    val("$(params.epochs)")
    sep()
    field("batch_size");
    print(io, " = ");
    val("$(params.batch_size)")
    sep()
    field("optimizer");
    print(io, " = ")
    if params.optimizer isa Vector
        opt_names = join([nameof(typeof(o)) for o in params.optimizer], ", ")
        val("[$(opt_names)]")
    else
        val("$(nameof(typeof(params.optimizer)))")
    end
    println(io)

    # State
    label("  State")
    field("epoch");
    print(io, " = ");
    val("$(params.current_epoch)")
    total_epochs = params.epochs isa Vector ? sum(params.epochs) : params.epochs
    hint(" / $total_epochs")
    sep()
    field("minibatch");
    print(io, " = ");
    val("$(params.current_minibatch)")
    sep()
    field("loss");
    print(io, " = ");
    val("$(params.loss_epoch)")
    sep()
    field("loss_history");
    print(io, " = ")
    n = length(params.loss_history)
    n == 0 ? hint("(empty)") : hint("$n $(n == 1 ? "entry" : "entries")")
    println(io)
end
