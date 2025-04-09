export FunctionalInversion

# Subtype composite type for a prediction simulation
"""
    mutable struct FunctionalInversion <: Simulation

An object representing a functional inversion simulation (i.e. the inversion of a function using some data-driven regressor).

# Fields
- `model::Sleipnir.Model`: The model used for the simulation.
- `glaciers::Vector{Sleipnir.AbstractGlacier}`: A vector of glaciers involved in the simulation.
- `parameters::Sleipnir.Parameters`: The parameters used for the simulation.
- `results::Vector{Results}`: A vector to store the results of the simulation.
"""
mutable struct FunctionalInversion{G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters} <: Simulation
    model::M
    glaciers::Vector{G}
    parameters::P
    results::Vector{Results}
    stats::TrainingStats
end

"""
    function FunctionalInversion(
        model::Sleipnir.Model,
        glaciers::Vector{G},
        parameters::Sleipnir.Parameters
    ) where {G <: Sleipnir.AbstractGlacier}
     
Constructor for FunctionalInversion struct with glacier model information, glaciers, and parameters.

# Arguments
- `model::Sleipnir.Model`: The model used for the simulation.
- `glaciers::Vector{G}`: A vector of glaciers involved in the simulation.
- `parameters::Sleipnir.Parameters`: The parameters used for the simulation.

# Returns
- `FunctionalInversion`: A new instance of the FunctionalInversion struct.
"""
function FunctionalInversion(
    model::M,
    glaciers::Vector{G},
    parameters::P
    ) where {G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters}

    # Generate multiple instances of the models for differentiation compatibility
    if !(model.iceflow isa Vector) || ((model.iceflow isa Vector) && (length(model.iceflow) != length(glaciers)))
        model.iceflow = [deepcopy(model.iceflow) for _ in 1:length(glaciers)]
    end
    if !(model.mass_balance isa Vector) || ((model.mass_balance isa Vector) && (length(model.mass_balance) != length(glaciers)))
        model.mass_balance = [deepcopy(model.mass_balance) for _ in 1:length(glaciers)]
    end

    # Build the results struct based on input values
    functional_inversion = FunctionalInversion(model,
                            glaciers,
                            parameters,
                            Vector{Results}([]),
                            TrainingStats())

    return functional_inversion
end

###############################################
################### UTILS #####################
###############################################

include("functional_inversion_utils.jl")
include("callback_utils.jl")