export InversionParameters

"""
    InversionParameters{F<:AbstractFloat}

A mutable struct that holds parameters for inversion processes. This struct is a subtype of `AbstractParameters`.

# Fields
- `initial_conditions::Vector{F}`: A vector of initial conditions.
- `lower_bound::Vector{F}`: A vector specifying the lower bounds for the parameters.
- `upper_bound::Vector{F}`: A vector specifying the upper bounds for the parameters.
- `regions_split::Vector{Int}`: A vector indicating how the regions are split.
- `x_tol::F`: The tolerance for the solution's x-values.
- `f_tol::F`: The tolerance for the function values.
- `solver::Any`: The solver to be used for the inversion process.
"""
mutable struct InversionParameters{F<:AbstractFloat} <: AbstractParameters
    initial_conditions::Vector{F}
    train_initial_conditions::Bool
    lower_bound::Vector{F}
    upper_bound::Vector{F}
    regions_split::Vector{Int}
    x_tol::F
    f_tol::F
    solver::Any
end

"""
    InversionParameters{F<:AbstractFloat}(;
        initial_conditions::Vector{F} = [1.0],
        lower_bound::Vector{F} = [0.0],
        upper_bound::Vector{F} = [Inf],
        regions_split::Vector{Int} = [1, 1],
        x_tol::F = 1.0e-3,
        f_tol::F = 1.0e-3,
        solver = BFGS()
    )

Initialize the parameters for the inversion process.

# Arguments
- `initial_conditions::Vector{F}`: Starting point for optimization.
- `lower_bound::Vector{F}`: Lower bounds for optimization variables.
- `upper_bound::Vector{F}`: Upper bounds for optimization variables.
- `regions_split::Vector{Int}`: Defines the amount of region split based on altitude and distance to border for the inversion process.
- `x_tol::F`: Tolerance for variables convergence.
- `f_tol::F`: Tolerance for function value convergence.
- `solver`: Optimization solver to be used.
"""
function InversionParameters{}(;
        initial_conditions::Vector{F} = [1.0],
        train_initial_conditions::Bool = false,
        lower_bound::Vector{F} = [0.0],
        upper_bound::Vector{F} = [Inf],
        regions_split::Vector{Int} = [1, 1],
        x_tol::F = 1.0e-3,
        f_tol::F = 1.0e-3,
        solver = BFGS()
    ) where F <: AbstractFloat
    inversionparameters = InversionParameters{F}(
        initial_conditions,
        train_initial_conditions,
        lower_bound,
        upper_bound,
        regions_split,
        x_tol,
        f_tol,
        solver
        )
    return inversionparameters
end

Base.:(==)(a::InversionParameters, b::InversionParameters) =
    a.initial_conditions == b.initial_conditions &&
    a.train_initial_conditions == b.train_initial_conditions &&
    a.lower_bound == b.lower_bound &&
    a.upper_bound == b.upper_bound &&
    a.regions_split == b.regions_split &&
    a.x_tol == b.x_tol &&
    a.f_tol == b.f_tol