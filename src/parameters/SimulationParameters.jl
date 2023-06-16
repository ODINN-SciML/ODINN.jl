

@kwdef struct SimulationParameters
    use_MB::Bool
    plots::Bool
    overwrite_climate::Bool
end

"""
    SimulationParameters(;
        use_MB::Bool = true,
        plots::Bool = true,
        overwrite_climate::Bool = false
        )
Initialize the parameters for a simulation.
Keyword arguments
=================
    - `use_MB`: Determines if surface mass balance should be used.
    - `plots`: Determines if plots should be made.
    - `overwrite_climate`: Determines if climate data should be overwritten
"""
function SimulationParameters(;
            use_MB::Bool = true,
            plots::Bool = true,
            overwrite_climate::Bool = false
            )

    # Build the simulation parameters based on input values
    simulation_parameters = SimulationParameters(use_MB, plots,
                                                overwrite_climate)

    return simulation_parameters
end