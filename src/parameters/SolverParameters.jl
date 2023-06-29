
  mutable struct SolverParameters{F <: AbstractFloat}
    solver::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm
    reltol::F
    step::F
    tstops::Union{Nothing,Vector{F}} 
    save_everystep::Bool
    progress::Bool
    progress_steps::Int
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
            reltol::F = 1e-12,
            step::F = 1.0/12.0,
            save_everystep = false,
            tstops::Union{Nothing,Vector{F}} = nothing,
            progress = true,
            progress_steps = 10
            ) where {F <: AbstractFloat}
    # Build the solver parameters based on input values
    solver_parameters = SolverParameters(solver, reltol, 
                                        step, tstops,
                                        save_everystep, progress, progress_steps)

    return solver_parameters
end

"""
    define_callback_steps(tspan::Tuple{Float64, Float64}, step::Float64)

Defines the times to stop for the DiscreteCallback given a step
"""
function define_callback_steps(tspan::Tuple{Float64, Float64}, step::Float64)
    tmin_int = Int(tspan[1])
    tmax_int = Int(tspan[2])+1
    tstops = range(tmin_int+step, tmax_int, step=step) |> collect
    tstops = filter(x->( (Int(tspan[1])<x) & (x<=Int(tspan[2])) ), tstops)
    return tstops
end