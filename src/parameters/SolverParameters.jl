
@kwdef struct SolverParameters{F <: AbstractFloat}
    solver::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm
    reltol::F
    tspan::Tuple{F, F}
    step::F
end

"""
    SolverParameters(;
        solver::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm = RDPK3Sp35(),
        reltol::Float64 = 1e-7
        )
Initialize the parameters for the numerical solver.
Keyword arguments
=================
    - `solver`: solver to use from DifferentialEquations.jl
    - `reltol`: Relative tolerance for the solver
"""
function SolverParameters(;
            solver::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm = RDPK3Sp35(),
            reltol::Float64 = 1e-7,
            tspan::Tuple{Float64, Float64} = (2010.0, 2015.0),
            step::Float64 = 1.0/12.0
            )
    # Build the solver parameters based on input values
    solver_parameters = SolverParameters(solver, reltol, 
                                        tspan, step)

    return solver_parameters
end