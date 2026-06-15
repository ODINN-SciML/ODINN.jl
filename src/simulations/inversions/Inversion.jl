export Inversion

"""
    mutable struct Inversion{MODEL, CACHE, GLACIER, RES} <: Simulation

An object representing an inversion simulation.
It can involve at the same time a classical inversion and a functional inversion (i.e. the inversion of a function using some data-driven regressor).

# Fields

  - `model::Sleipnir.Model`: The model used for the simulation.
  - `glaciers::Vector{Sleipnir.AbstractGlacier}`: A vector of glaciers involved in the simulation.
  - `parameters::Sleipnir.Parameters`: The parameters used for the simulation.
  - `results::ODINN.Results`: A `ODINN.Results` instance to store the results of the inversion and of the forward simulations.
"""
mutable struct Inversion{MODEL, CACHE, GLACIER, RES} <: Simulation
    model::MODEL
    cache::Union{CACHE, Nothing}
    glaciers::Vector{GLACIER}
    parameters::Sleipnir.Parameters
    results::ODINN.Results
end

"""
    function Inversion(
        model::M,
        glaciers::Vector{G},
        parameters::P
    ) where {G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters}

Constructor for Inversion struct with glacier model information, glaciers, and parameters.

# Arguments

  - `model::Sleipnir.Model`: The model used for the simulation.
  - `glaciers::Vector{G}`: A vector of glaciers involved in the simulation.
  - `parameters::Sleipnir.Parameters`: The parameters used for the simulation.

# Returns

  - `Inversion`: A new instance of the Inversion struct.
"""
function Inversion(
        model::M,
        glaciers::Vector{G},
        parameters::P
) where {G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters}

    # We perform this check here to avoid having to provide the parameters when creating the model
    @assert targetType(model.trainable_components.target) == parameters.UDE.target "Target does not match the one provided in the parameters."

    # Optionally calibrate the mass balance model per glacier (no-op unless the
    # model type defines a calibration routine, e.g. TImodel1).
    if parameters.simulation.use_MB && parameters.simulation.calibrate_MB
        calibrate_MB_model!(model, glaciers, parameters)
    end

    # Build the results struct based on input values
    emptySimulationResults = Vector{Sleipnir.Results{Sleipnir.Float, Sleipnir.Int}}([])
    emptyResults = Results(emptySimulationResults, TrainingStats())
    inversion = Inversion{M, cache_type(model), G, typeof(emptyResults)}(model, nothing,
        glaciers,
        parameters,
        emptyResults)

    return inversion
end

###############################################
################### UTILS #####################
###############################################

include("sciml_utils.jl")
include("inversion_utils.jl")
include("callback_utils.jl")
