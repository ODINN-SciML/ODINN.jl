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
mutable struct FunctionalInversion  <: Simulation 
    model::Sleipnir.Model
    glaciers::Vector{Sleipnir.AbstractGlacier}
    parameters::Sleipnir.Parameters
    results::Vector{Results}
end

"""
    function FunctionalInversion(
        model::Sleipnir.Model,
        glaciers::Vector{G},
        parameters::Sleipnir.Parameters
    ) where {G <: Sleipnir.AbstractGlacier}
Constructor for FunctionalInversion struct with glacier model infomation, glaciers and parameters.
Keyword arguments
=================
"""
function FunctionalInversion(
    model::Sleipnir.Model,
    glaciers::Vector{G},
    parameters::Sleipnir.Parameters
    ) where {G <: Sleipnir.AbstractGlacier}

    # Generate multiple instances of the models for Reverse Diff compatibility
    model.iceflow = [deepcopy(model.iceflow) for _ in 1:length(glaciers)]
    model.mass_balance = [deepcopy(model.mass_balance) for _ in 1:length(glaciers)]

    # Build the results struct based on input values
    functional_inversion = FunctionalInversion(model,
                            glaciers,
                            parameters,
                            Vector{Results}([]))

    return functional_inversion
end

###############################################
################### UTILS #####################
###############################################

include("functional_inversion_utils.jl")