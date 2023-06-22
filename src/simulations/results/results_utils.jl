

"""
    store_results!(simulation::SIM, glacier_idx::I, solution) where {SIM <: Simulation, I <: Int}

Store the results of a simulation of a single glacier into a `Results`.
"""
function store_results!(simulation::SIM, glacier_idx::I, solution) where {SIM <: Simulation, I <: Int} # TODO: define type for solution!

    results = Results(simulation.glaciers[glacier_idx], simulation.model.iceflow;
                      H = solution.u,
                      S = simulation.model.iceflow.S,
                      B = simulation.glaciers[glacier_idx].B,
                      V = simulation.model.iceflow.V,
                      Vx = simulation.model.iceflow.Vx,
                      Vy = simulation.model.iceflow.Vy)
                      
    push!(simulation.results, results)
end

"""
    save_results_file(simulation::Prediction)

Save simulation `Results` into a `.jld2` file.
"""
function save_results_file(simulation::Prediction; path::Union{String,Nothing}=nothing)
    # Create path for simulation results
    predictions_path = joinpath(ODINN.root_dir, "data/results/predictions")
    if !ispath(predictions_path)
        mkpath(predictions_path)
    end

    if isnothing(path)
        timestamp = now()
        nglaciers = length(simulation.glaciers)
        jldsave(joinpath(predictions_path, "prediction_$(nglaciers)glaciers_$timestamp.jld2"); simulation.results)
    end
end