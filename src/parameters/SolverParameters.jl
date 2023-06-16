
@kwdef struct SolverParameters
    solver::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm
    reltol::Float64
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
            reltol::Float64 = 1e-7
            )
    # Build the solver parameters based on input values
    solver_parameters = SolverParameters(solver, reltol)

    return solver_parameters
end