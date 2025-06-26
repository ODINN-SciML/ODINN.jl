"""

This function saves the results of an inversion to a file in JLD2 format. If the `path` argument is not provided, the function will create a default path based on the current project directory. The results are saved in a file named `prediction_<nglaciers>glaciers_<tspan>.jld2`, where `<nglaciers>` is the number of glaciers in the simulation and `<tspan>` is the simulation time span.
"""
function save_inversion_file!(
    sol,
    simulation::SIM;
    path::Union{String,Nothing} = nothing,
    file_name::Union{String,Nothing} = nothing
    ) where {SIM <: Simulation}

    # Create path for simulation results
    if isnothing(path)
        simulation_path = joinpath(dirname(Base.current_project()), "data/results/inversions")
    else
        simulation_path = path
    end
    if !ispath(simulation_path)
        mkpath(simulation_path)
    end

    res = ODINN.TrainingResult(
        θ = sol.u,
        θ_hist = simulation.stats.θ_hist,
        ∇θ_hist = simulation.stats.∇θ_hist,
        losses = simulation.stats.losses,
        params = simulation.parameters
    )

    # Save results
    jldsave(joinpath(simulation_path, file_name); res)

    return nothing
end
