include("../Simulation.jl")
include("../../models/Model.jl")
include("../../glaciers/Glacier.jl")
include("../../parameters/Parameters.jl")

# Subtype composite type for a prediction simulation
@kwdef struct Prediction  <: Simulation 
    model::Model
    glaciers::Vector{Glacier}
    parameters::Parameters
    results::Results
end

function Prediction(;
    model::Model,
    glaciers::Vector{Glacier},
    parameters::Parameters,
    results::Results = Results()
        )

    # Build the results struct based on input values
    prediction = Prediction(model = model,
                            glaciers = glaciers,
                            parameters = parameters,
                            results = results)

    return prediction
end