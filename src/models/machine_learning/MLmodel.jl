export NeuralNetwork

include("ML_utils.jl")
include("Target_utils.jl")

# Abstract type as a parent type for Machine Learning models
abstract type MLmodel <: AbstractModel end

"""
    Model(; iceflow::Union{IFM, Vector{IFM}, Nothing}, mass_balance::Union{MBM, Vector{MBM}, Nothing}, machine_learning::Union{MLM, Nothing}) where {IFM <: IceflowModel, MBM <: MBmodel, MLM <: MLmodel}

Creates a new model instance using the provided iceflow, mass balance, and machine learning components.

# Arguments
- `iceflow::Union{IFM, Vector{IFM}, Nothing}`: The iceflow model(s) to be used. Can be a single model, a vector of models, or `nothing`.
- `mass_balance::Union{MBM, Vector{MBM}, Nothing}`: The mass balance model(s) to be used. Can be a single model, a vector of models, or `nothing`.
- `machine_learning::Union{MLM, Nothing}`: The machine learning model to be used. Can be a single model or `nothing`.
# Returns
- `model`: A new instance of `Sleipnir.Model` initialized with the provided components.
"""
function Model(;
    iceflow::Union{IFM, Vector{IFM}, Nothing},
    mass_balance::Union{MBM, Vector{MBM}, Nothing},
    machine_learning::Union{MLM, Nothing},
    ) where {IFM <: IceflowModel, MBM <: MBmodel, MLM <: MLmodel}

    iceflowType = isa(iceflow, Vector) ? typeof(iceflow[1]) : typeof(iceflow)
    massbalanceType = isa(mass_balance, Vector) ? typeof(mass_balance[1]) : typeof(mass_balance)
    model = Sleipnir.Model{iceflowType, massbalanceType, typeof(machine_learning)}(iceflow, mass_balance, machine_learning)

    return model
end

"""
    NeuralNetwork{
        ChainType <: Lux.Chain,
        ComponentVectorType <: ComponentVector,
        NamedTupleType <: NamedTuple
    } <: MLmodel

Feed-forward neural network.

# Fields
- `architecture::ChainType`: `Flux.Chain` neural network architecture
- `θ::ComponentVectorType`: Neural network parameters
- `st::NamedTupleType`: Neural network status
"""
mutable struct NeuralNetwork{
    ChainType <: Lux.Chain,
    ComponentVectorType <: ComponentVector,
    NamedTupleType <: NamedTuple,
    TAR <: AbstractTarget
} <: MLmodel
    architecture::ChainType
    θ::ComponentVectorType
    st::NamedTupleType
    target::TAR
end

"""
    NN(params::Parameters;
        architecture::Union{Flux.Chain, Nothing} = nothing,
        θ::Union{Vector{AbstractFloat}, Nothing} = nothing)

Creates a new feed-forward neural network.

# Keyword arguments
- `architecture`: `Flux.Chain` neural network architecture (optional)
- `θ`: Neural network parameters (optional)
"""
function NeuralNetwork(
    params::P;
    architecture::Union{ChainType, Nothing} = nothing,
    θ::Union{ComponentArrayType, Nothing} = nothing,
    st::Union{NamedTupleType, Nothing} = nothing,
) where {
    P<:Sleipnir.Parameters, ChainType<:Lux.Chain,
    ComponentArrayType<:ComponentArray, NamedTupleType<:NamedTuple
}

    # Float type
    ft = Sleipnir.Float
    lightNN = params.simulation.test_mode

    _nn_is_provided = [isnothing(architecture), isnothing(θ), isnothing(st)]
    if all(_nn_is_provided)
        if params.UDE.target == :A
            architecture, θ, st = get_NN(θ, ft; lightNN=lightNN)
        elseif params.UDE.target == :D
            # TODO: I shoudl store NN elements in Target
            architecture = Lux.Chain(
                Dense(2, 3, x -> softplus.(x)),
                Dense(3, 1, sigmoid)
            )
            θ, st = Lux.setup(ODINN.rng_seed(), architecture)
            θ = ODINN.ComponentArray(θ=θ)
            if Sleipnir.Float == Float64
                architecture = f64(architecture)
                θ = f64(θ)
                st = f64(st)
            end
        else
            # Contruct default NN
            @warn "Constructing default Neural Network"
            architecture, θ, st = get_NN(θ, ft; lightNN=lightNN)
        end
    elseif any(_nn_is_provided) && !all(_nn_is_provided)
        @warn "To specify the neural network please provide all (architecture, θ, st), not just a subset of them."
    end

    # Build target based on parameters
    target_object = SIA2D_target(name = params.UDE.target)

    # Build the simulation parameters based on input values
    # TODO: I don't think target should be inside NN, but rather having the neural net elements
    # as part of a regressor type.
    neural_net = NeuralNetwork{typeof(architecture), typeof(θ), typeof(st), typeof(target_object)}(
        architecture, θ, st,
        target_object
    )

    return neural_net
end
