
export Prediction

# Subtype composite type for a prediction simulation
mutable struct Prediction  <: Simulation 
    model::Model
    glaciers::Vector{Glacier}
    parameters::Parameters
    results::Vector{Results}
end

function Prediction(
    model::Model,
    glaciers::Vector{Glacier},
    parameters::Parameters
        )

    # Build the results struct based on input values
    prediction = Prediction(model,
                            glaciers,
                            parameters,
                            Vector{Results}([]))

    return prediction
end

###############################################
################### UTILS #####################
###############################################

include("prediction_utils.jl")
