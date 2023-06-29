

"""
    store_results!(simulation::SIM, glacier_idx::I, solution) where {SIM <: Simulation, I <: Int}

Store the results of a simulation of a single glacier into a `Results`.
"""
function create_results(simulation::SIM, glacier_idx::I, solution; light=false) where {SIM <: Simulation, I <: Int} # TODO: define type for solution!
    ft = simulation.parameters.simulation.float_type
    H::Vector{Matrix{ft}} = light ? [solution.u[begin],solution.u[end]] : solution.u 
    results = Results(simulation.glaciers[glacier_idx], simulation.model.iceflow;
                      H = H,
                      S = simulation.model.iceflow.S,
                      B = simulation.glaciers[glacier_idx].B,
                      V = simulation.model.iceflow.V,
                      Vx = simulation.model.iceflow.Vx,
                      Vy = simulation.model.iceflow.Vy)
                      
    return results
end

"""
    save_results_file(simulation::Prediction)

Save simulation `Results` into a `.jld2` file.
"""
function save_results_file!(results_list::Vector{Results{F}}, simulation::Prediction; path::Union{String,Nothing}=nothing) where {F <: AbstractFloat}
    # Create path for simulation results
    predictions_path = joinpath(ODINN.root_dir, "data/results/predictions")
    if !ispath(predictions_path)
        mkpath(predictions_path)
    end

    simulation.results = results_list

    if isnothing(path)
        tspan = simulation.parameters.simulation.tspan
        nglaciers = length(simulation.glaciers)
        jldsave(joinpath(predictions_path, "prediction_$(nglaciers)glaciers_$tspan.jld2"); simulation.results)
    end
end