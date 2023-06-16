
include("PhysicalParameters.jl")
include("Hyperparameters.jl")
include("SolverParameters.jl")
include("UDEparameters.jl")
include("OGGMparameters.jl")
include("SimulationParameters.jl")

@kwdef struct Parameters
    physical::PhysicalParameters
    hyper::Hyperparameters
    solver::SolverParameters
    UDE::UDEparameters
    OGGM::OGGMparameters
    simulation::SimulationParameters
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
function Parameters(;
            physical::PhysicalParameters = PhysicalParameters(),
            hyper::Hyperparameters = Hyperparameters(),
            solver::SolverParameters = SolverParameters(),
            UDE::UDEparameters = UDEparameters(),
            OGGM::OGGMparameters = OGGMparameters(),
            simulation::SimulationParameters = SimulationParameters()
            )

    # Build the parameters based on all the subtypes of parameters
    parameters = Parameters(physical, hyper, solver,
                            UDE, OGGM, simulation)

    return parameters
end