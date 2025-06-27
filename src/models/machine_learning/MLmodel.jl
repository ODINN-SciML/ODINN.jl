export NeuralNetwork


# Abstract type as a parent type for Machine Learning models
abstract type MLmodel <: AbstractModel end

"""
    Model(; iceflow::Union{IFM, Nothing}, mass_balance::Union{MBM, Nothing}, machine_learning::Union{MLM, Nothing}) where {IFM <: IceflowModel, MBM <: MBmodel, MLM <: MLmodel}

Creates a new model instance using the provided iceflow, mass balance, and machine learning components.

# Arguments
- `iceflow::Union{IFM, Nothing}`: The iceflow model to be used. Can be a single model or `nothing`.
- `mass_balance::Union{MBM, Nothing}`: The mass balance model to be used. Can be a single model or `nothing`.
- `machine_learning::Union{MLM, Nothing}`: The machine learning model to be used. Can be a single model or `nothing`.
# Returns
- `model`: A new instance of `Sleipnir.Model` initialized with the provided components.
"""
function Model(;
    iceflow::Union{IFM, Nothing},
    mass_balance::Union{MBM, Nothing},
    machine_learning::Union{MLM, Nothing},
    ) where {IFM <: IceflowModel, MBM <: MBmodel, MLM <: MLmodel}

    errMssg = "law must be differentiable in order to be used within ODINN"
    if iceflow.U_is_provided
        @assert is_differentiable(iceflow.U) "U $(errMssg)"
    else
        @assert is_differentiable(iceflow.A) "A $(errMssg)"
        @assert is_differentiable(iceflow.C) "C $(errMssg)"
        @assert is_differentiable(iceflow.n) "n $(errMssg)"
    end

    iceflowType = typeof(iceflow)
    massbalanceType = typeof(mass_balance)
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
    target::Union{TAR, Nothing} = nothing,
    seed::Union{RNG, Nothing} = nothing
) where {
    P<:Sleipnir.Parameters,
    ChainType<:Lux.Chain,
    ComponentArrayType<:ComponentArray,
    NamedTupleType<:NamedTuple,
    TAR<:AbstractTarget,
    RNG<:AbstractRNG
}

    # Float type
    ft = Sleipnir.Float
    lightNN = params.simulation.test_mode

    if isnothing(architecture) & isnothing(θ) & isnothing(st)
        if params.UDE.target == :A
            architecture, θ, st = get_default_NN(θ, ft; lightNN = lightNN, seed = seed)
        elseif params.UDE.target == :D_hybrid
            architecture = build_default_NN(; n_input = 2, lightNN = lightNN)
            architecture, θ, st = set_NN(architecture; ft = ft, seed = seed)
        elseif params.UDE.target == :D
            architecture = build_default_NN(; n_input = 2, lightNN = lightNN)
            architecture, θ, st = set_NN(architecture; ft = ft, seed = seed)
        else
            @warn "Constructing default Neural Network"
            architecture, θ, st = get_default_NN(θ, ft; lightNN = lightNN, seed = seed)
        end
    elseif !isnothing(architecture) & isnothing(θ) & isnothing(st)
        architecture, θ, st = set_NN(architecture; ft = ft, seed = seed)
    elseif !isnothing(architecture) & !isnothing(θ) & isnothing(st)
        architecture, θ, st = set_NN(architecture; θ_trained = θ, ft = ft, seed = seed)
    elseif !isnothing(architecture) & !isnothing(θ) & !isnothing(st)
        # Architecture and setup already provided
        nothing
    else
        @warn "To specify the neural network please provide architecture, (architecture, θ, st), or none of them to create default NN."
    end

    # Build target based on parameters
    if isnothing(target)
        if params.UDE.target == :A
            target = SIA2D_A_target()
        elseif params.UDE.target == :D_hybrid
            target = SIA2D_D_hybrid_target()
        elseif params.UDE.target == :D
            target = SIA2D_D_target()
        else
            @warn "Target object has not been provided during simulation."
        end
    end
    # Build the simulation parameters based on input values
    # TODO: I don't think target should be inside NN, but rather having the neural net elements
    # as part of a regressor type.
    neural_net = NeuralNetwork{typeof(architecture), typeof(θ), typeof(st), typeof(target)}(
        architecture, θ, st, target
    )

    return neural_net
end
