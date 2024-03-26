export InversionParameters

mutable struct InversionParameters <: AbstractParameters
    initial_conditions::Vector{Float64}
    lower_bound::Vector{Float64}
    upper_bound::Vector{Float64}
    regions_split::Vector{Int}
    x_tol::Float64
    f_tol::Float64
    solver::Any  
end

"""
    InversionParameters(;
        initial_conditions::Vector{Float64} = [1.0],
        lower_bound::Vector{Float64} = [0.0],
        upper_bound::Vector{Float64} = [Inf],
        regions_split::Vector{Int} = [1, 1],
        x_tol::Float64 = 1.0e-3,
        f_tol::Float64 = 1.0e-3,
        solver = BFGS()
    )

Initialize the parameters for the inversion process.

# Arguments
- `initial_conditions`: Starting point for optimization.
- `lower_bound`: Lower bounds for optimization variables.
- `upper_bound`: Upper bounds for optimization variables.
- `regions_split`: Defines amount of region split based on altitude and distance to border for the inversion process.
- `x_tol`: Tolerance for variables convergence.
- `f_tol`: Tolerance for function value convergence.
- `solver`: Optimization solver to be used.
"""
function InversionParameters(;
        initial_conditions::Vector{Float64} = [1.0],
        lower_bound::Vector{Float64} = [0.0],
        upper_bound::Vector{Float64} = [Inf],
        regions_split::Vector{Int} = [1, 1],
        x_tol::Float64 = 1.0e-3,
        f_tol::Float64 = 1.0e-3,
        solver = BFGS()
    )
    inversionparameters = InversionParameters(initial_conditions, lower_bound, upper_bound, regions_split, x_tol, f_tol, solver)
    
    return inversionparameters
end

Base.:(==)(a::InversionParameters, b::InversionParameters) = 
    a.initial_conditions == b.initial_conditions &&
    a.lower_bound == b.lower_bound &&
    a.upper_bound == b.upper_bound &&
    a.regions_split == b.regions_split &&
    a.x_tol == b.x_tol &&
    a.f_tol == b.f_tol