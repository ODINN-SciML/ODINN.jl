export NeuralNetwork

"""
    NeuralNetwork{
        ChainType <: Lux.Chain,
        ComponentVectorType <: ComponentVector,
        NamedTupleType <: NamedTuple,
    } <: FunctionalModel

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
} <: FunctionalModel
    architecture::ChainType
    θ::ComponentVectorType
    st::NamedTupleType

    function NeuralNetwork(
            params::P;
            architecture::Union{ChainType, Nothing} = nothing,
            θ::Union{ComponentArrayType, Nothing} = nothing,
            st::Union{NamedTupleType, Nothing} = nothing,
            seed::Union{RNG, Nothing} = nothing
    ) where {
            P <: Sleipnir.Parameters,
            ChainType <: Lux.Chain,
            ComponentArrayType <: ComponentArray,
            NamedTupleType <: NamedTuple,
            RNG <: AbstractRNG
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

        new{typeof(architecture), typeof(θ), typeof(st)}(
            architecture, θ, st
        )
    end
end
# Note: we could define any other kind of regressor as a subtype of TrainableModel

# Display setup
function Base.show(io::IO, nn_model::NeuralNetwork)
    println(io, "--- NeuralNetwork ---")
    println(io, "    architecture:")
    # Retrieve the printed lines
    iotmp = IOBuffer()
    show(iotmp, "text/plain", nn_model.architecture)
    str = String(take!(iotmp))
    # Add prefix to each line
    prefix = "      "
    prefixed_str = join(prefix .* split(str, '\n'), '\n')
    println(io, prefixed_str)
    print(io, "    θ: ComponentVector of length $(length(nn_model.θ))")
end
