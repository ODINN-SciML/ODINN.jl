export FunctionalInversion

# Subtype composite type for a prediction simulation
"""
    struct FunctionalInversion{MODEL, CACHE, GLACIER} <: Simulation

An object representing a functional inversion simulation (i.e. the inversion of a function using some data-driven regressor).

# Fields
- `model::Sleipnir.Model`: The model used for the simulation.
- `glaciers::Vector{Sleipnir.AbstractGlacier}`: A vector of glaciers involved in the simulation.
- `parameters::Sleipnir.Parameters`: The parameters used for the simulation.
- `results::Vector{Results}`: A vector to store the results of the simulation.
"""
mutable struct FunctionalInversion{MODEL, CACHE, GLACIER} <: Simulation
    model::MODEL
    cache::Union{CACHE, Nothing}
    glaciers::Vector{GLACIER}
    parameters::Sleipnir.Parameters
    results::Vector{<: Results}
    stats::TrainingStats
end

"""
    function FunctionalInversion(
        model::M,
        glaciers::Vector{G},
        parameters::P
        ) where {G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters}

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

    # We perform this check here to avoid having to provide the parameters when creating the model
    @assert targetType(model.machine_learning.target) == parameters.UDE.target "Target does not match the one provided in the parameters."

    # Build the results struct based on input values
    functional_inversion = FunctionalInversion{M, cache_type(model), G}(model, nothing,
                            glaciers,
                            parameters,
                            Vector{Results{Sleipnir.Float, Sleipnir.Int}}([]),
                            TrainingStats())

    return functional_inversion
end

###############################################
################### UTILS #####################
###############################################

include("sciml_utils.jl")
include("functional_inversion_utils.jl")
include("callback_utils.jl")
