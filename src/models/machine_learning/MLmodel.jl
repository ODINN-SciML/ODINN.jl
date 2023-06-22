
export NN

# Abstract type as a parent type for Machine Learning models
abstract type MLmodel end

struct NN{F <: AbstractFloat} <: MLmodel 
    architecture::Flux.Chain
    θ::Vector{F}
end

"""
    NN(;
        architecture::Union{Flux.Chain, Nothing} = nothing,
        θ::Union{Vector{AbstractFloat}, Nothing} = nothing)
        Temperature-index model with a single degree-day factor.

Keyword arguments
=================
    - `architecture`: `Flux.Chain` neural network architecture
    - `θ`: Neural network parameters
"""
function NN(params::Parameters;
            architecture::Union{Flux.Chain, Nothing} = nothing,
            θ::Union{Vector{F}, Nothing} = nothing) where {F <: AbstractFloat}

    if isnothing(architecture)
        architecture, θ = get_NN(θ)
    end

    # Build the simulation parameters based on input values
    ft = params.simulation.float_type
    neural_net = NN{ft}(architecture, θ)

    return neural_net
end
