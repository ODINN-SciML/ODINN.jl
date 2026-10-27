"""
This function saves the results of an inversion to a file in JLD2 format. If the `path` argument is not provided, the function will save the file in `data/results/inversions` inside the `working_dir` of the simulation parameters.
"""
function save_inversion_file!(
        sol,
        simulation::SIM;
        path::Union{String, Nothing} = nothing,
        file_name::Union{String, Nothing} = nothing
) where {SIM <: Simulation}

    # Create path for simulation results
    if isnothing(path)
        simulation_path = joinpath(
            simulation.parameters.simulation.working_dir, "data/results/inversions")
    else
        simulation_path = path
    end
    if !ispath(simulation_path)
        mkpath(simulation_path)
    end

    res = ODINN.TrainingResult(
        θ = sol.u,
        θ_hist = simulation.results.stats.θ_hist,
        ∇θ_hist = simulation.results.stats.∇θ_hist,
        losses = simulation.results.stats.losses,
        params = simulation.parameters
    )

    # Save results
    jldsave(joinpath(simulation_path, file_name); res)

    return nothing
end
