

struct SimulationParameters{F <: AbstractFloat}
    use_MB::Bool
    plots::Bool
    velocities::Bool 
    overwrite_climate::Bool
    float_type::DataType
    int_type::DataType
    tspan::Tuple{F, F}
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
            velocities::Bool = true,
            overwrite_climate::Bool = false,
            float_type::DataType = Float64,
            int_type::DataType = Int64,
            tspan::Tuple{F, F} = (2010.0,2015.0)
            ) where {F <: AbstractFloat}

    # Build the simulation parameters based on input values
    simulation_parameters = SimulationParameters(use_MB, plots, velocities,
                                                overwrite_climate,
                                                float_type, int_type,
                                                tspan)

    return simulation_parameters
end