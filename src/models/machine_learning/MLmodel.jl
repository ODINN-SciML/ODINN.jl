export NeuralNetwork
export ParallelBatch

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

    model = Sleipnir.Model(iceflow, mass_balance, machine_learning)

    return model
end

"""
    NN{F <: AbstractFloat}(architecture::Flux.Chain, NN_f::Optimisers.Restructure, θ::Vector{F})

Feed-forward neural network.

# Fields
- `architecture`: `Flux.Chain` neural network architecture
- `NN_f`: Neural network restructuring
- `θ`: Neural network parameters
"""
mutable struct NeuralNetwork{F <: AbstractFloat} <: MLmodel
    architecture::Lux.Chain
    θ::ComponentArray{F}
    st::NamedTuple
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
function NeuralNetwork(params::Sleipnir.Parameters;
            architecture::Union{Lux.Chain, Nothing} = nothing,
            θ::Union{ComponentArray{F}, Nothing} = nothing) where {F <: AbstractFloat}

    # Float type
    ft = Sleipnir.Float

    if isnothing(architecture)
        # architecture, θ, NN_f = get_NN(θ)
        architecture, θ, st = get_NN(θ, ft)
    end

    # Build the simulation parameters based on input values
    neural_net = NeuralNetwork{ft}(architecture, θ, st)

    return neural_net
end