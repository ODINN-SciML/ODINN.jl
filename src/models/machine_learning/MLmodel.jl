
export NN

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

mutable struct NN{T1, T2, T3} <: MLmodel 
    architecture::T1
    st::T2
    θ::T3
end
(f::NN)(u) = f.architecture(u, f.θ, f.st)

"""
    NN(params::Parameters;
        architecture::Union{Lux.Chain, Nothing} = nothing,
        θ::Union{Vector{AbstractFloat}, Nothing} = nothing)
        
        Feed-forward neural network.

Keyword arguments
=================
    - `architecture`: `Lux.Chain` neural network architecture
    - `θ`: Neural network parameters
"""
function NN(params::Sleipnir.Parameters;
            architecture::Union{Lux.Chain, Nothing} = nothing,
            θ::Union{ComponentArray{F}, Nothing} = nothing) where {F <: AbstractFloat}

    if isnothing(architecture)
        architecture, θ, st = get_NN(θ)
    end

    # Build the simulation parameters based on input values
    ft = params.simulation.float_type
    neural_net = NN(architecture, st, θ)

    return neural_net
end
