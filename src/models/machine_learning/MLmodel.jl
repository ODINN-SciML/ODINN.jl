export NeuralNetwork

include("ML_utils.jl")

# Abstract type as a parent type for Machine Learning models
abstract type MLmodel <: AbstractModel end

"""
function Model(;
    iceflow::Union{IFM, Nothing},
    mass_balance::Union{MBM, Nothing}
    machine_learning::Union{MLM, Nothing},
    ) where {IFM <: IceflowModel, MBM <: MBmodel, MLM <: MLmodel}

Initialize Model at ODINN level (iceflow + mass balance + machine learning).

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
    NamedTupleType <: NamedTuple
} <: MLmodel
    architecture::ChainType
    θ::ComponentVectorType
    st::NamedTupleType
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
function NeuralNetwork(params::P;
            architecture::Union{ChainType, Nothing} = nothing,
            θ::Union{ComponentArrayType, Nothing} = nothing) where {P <: Sleipnir.Parameters, ChainType <: Lux.Chain, ComponentArrayType <: ComponentArray}

    # Float type
    ft = Sleipnir.Float

    if isnothing(architecture)
        architecture, θ, st = get_NN(θ, ft)
    end

    # Build the simulation parameters based on input values
    neural_net = NeuralNetwork{typeof(architecture), typeof(θ), typeof(st)}(architecture, θ, st)

    return neural_net
end
