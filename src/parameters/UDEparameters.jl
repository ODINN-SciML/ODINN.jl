export UDEparameters

"""
A mutable struct that holds parameters for a UDE (Universal Differential Equation).

    UDEparameters{ADJ <: AbstractAdjointMethod} <: AbstractParameters

# Fields
- `sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm`: The sensitivity algorithm used for adjoint sensitivity analysis.
- `optimization_method::String`: The optimization method to be used.
- `target::Symbol`: The target variable for the optimization.
"""
mutable struct UDEparameters{ADJ<:AbstractAdjointMethod} <: AbstractParameters
    sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm
    optim_autoAD::AbstractADType
    grad::Union{ADJ, Nothing}
    optimization_method::String
    empirical_loss_function::AbstractLoss
    target::Union{Symbol, Nothing}
end

Base.:(==)(a::UDEparameters, b::UDEparameters) = a.sensealg == b.sensealg &&
    a.optim_autoAD == b.optim_autoAD && a.grad == b.grad &&
    a.optimization_method == b.optimization_method &&
    a.empirical_loss_function == b.empirical_loss_function &&
    a.target == b.target


"""
    UDEparameters(;
        sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = GaussAdjoint(autojacvec=SciMLSensitivity.EnzymeVJP()),
        optim_autoAD::AbstractADType = Optimization.AutoEnzyme(),
        grad::ADJ = SciMLSensitivityAdjoint(),
        optimization_method::String = "AD+AD",
        empirical_loss_function::AbstractLoss = LossH(),
        target::Union{Symbol, Nothing} = :A
    ) where {ADJ <: AbstractAdjointMethod}

Create a `UDEparameters` object for configuring the sensitivity analysis and optimization of a Universal Differential Equation (UDE).

# Keyword Arguments
- `sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm`: The sensitivity algorithm to use for adjoint calculations. Defaults to `GaussAdjoint(autojacvec=SciMLSensitivity.EnzymeVJP())`.
- `optim_autoAD::AbstractADType`: The automatic differentiation type for optimization. Defaults to `Optimization.AutoEnzyme()`.
- `grad::ADJ`: The adjoint gradient computation method. Defaults to `SciMLSensitivityAdjoint()`.
- `optimization_method::String`: The optimization method to use. Must be either `"AD+AD"` or `"AD+Diff"`. Defaults to `"AD+AD"`.
- `empirical_loss_function::AbstractLoss`: The loss function to use for optimization. Defaults to `LossH()`.
- `target::Union{Symbol, Nothing}`: The target variable for optimization. Defaults to `:A`.

# Returns
- A `UDEparameters` object configured with the specified sensitivity, optimization, and loss settings.

# Description
This function creates a `UDEparameters` object that encapsulates the configuration for sensitivity analysis, optimization, and loss computation in a Universal Differential Equation (UDE) framework. It verifies that the provided `optimization_method` is valid and constructs the solver parameters accordingly.

# Notes
- The `optimization_method` must be either `"AD+AD"` (automatic differentiation for both forward and backward passes) or `"AD+Diff"` (automatic differentiation combined with finite differences).
- The `empirical_loss_function` determines how the loss is computed during optimization.
"""
function UDEparameters(;
        sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = GaussAdjoint(autojacvec=SciMLSensitivity.EnzymeVJP()),
        optim_autoAD::AbstractADType = Optimization.AutoEnzyme(),
        grad::ADJ = SciMLSensitivityAdjoint(),
        optimization_method::String = "AD+AD",
        empirical_loss_function::AbstractLoss = LossH(),
        target::Union{Symbol, Nothing} = :A
    ) where {ADJ <: AbstractAdjointMethod}

    # Verify that the optimization method is correct
    @assert ((optimization_method == "AD+AD") || (optimization_method == "AD+Diff")) "Wrong optimization method! Needs to be either `AD+AD` or `AD+Diff`"

    # Build the solver parameters based on input values
    UDE_parameters = UDEparameters{typeof(grad)}(
        sensealg, optim_autoAD, grad, optimization_method,
        empirical_loss_function, target
    )

    return UDE_parameters
end

"""
    UDEparameters(
        params::UDEparameters;
        grad::Union{<: AbstractAdjointMethod, Nothing} = nothing,
    )

Copy a `UDEparameters` and replace its `grad` field by the provided value.

# Arguments
- `params::UDEparameters`: Object to copy.
- `grad::Union{<: AbstractAdjointMethod, Nothing}`: Adjoint flavor to replace by in the `UDEparameters` struct.
    Defaults to `nothing` which keeps the existing value.

# Returns
- A new `UDEparameters` object.
"""
function UDEparameters(
    params::UDEparameters;
    grad::Union{<: AbstractAdjointMethod, Nothing} = nothing,
)
    grad = isnothing(grad) ? params.grad : grad
    return UDEparameters{typeof(grad)}(
        params.sensealg,
        params.optim_autoAD,
        grad,
        params.optimization_method,
        params.empirical_loss_function,
        params.target,
    )
end

include("InversionParameters.jl")

"""
Constructor for the `Parameters` type. Since some of the subtypes of parameters are defined
in different packages of the ODINN ecosystem, this constructor will call the constructors of
the different subtypes and return a `Parameters` object with the corresponding subtypes. 
The `Parameters` mutable struct is defined in `Sleipnir.jl` using abstract types, which are
later on defined in the different packages of the ODINN ecosystem.

    Parameters(;
        physical::PhysicalParameters = PhysicalParameters(),
        simulation::SimulationParameters = SimulationParameters(),
        solver::SolverParameters = SolverParameters(),
        hyper::Hyperparameters = Hyperparameters(),
        UDE::UDEparameters = UDEparameters(),
        inversion::InversionParameters = InversionParameters(),
    )


Keyword arguments
=================
  - `physical::PhysicalParameters`: Physical parameters for the simulation.
  - `simulation::SimulationParameters`: Parameters related to the simulation setup.
  - `solver::SolverParameters`: Parameters for the solver configuration.
  - `hyper::Hyperparameters`: Hyperparameters for the model.
  - `UDE::UDEparameters`: Parameters specific to the UDE (Universal Differential Equation).
  - `inversion::InversionParameters`: Parameters for inversion processes.
"""
function Parameters(;
    physical::PhysicalParameters = PhysicalParameters(),
    simulation::SimulationParameters = SimulationParameters(),
    solver::SolverParameters = SolverParameters(),
    hyper::Hyperparameters = Hyperparameters(),
    UDE::UDEparameters = UDEparameters(),
    inversion::InversionParameters = InversionParameters(),
)
    parameters = Sleipnir.Parameters(physical, simulation, hyper, solver, UDE, inversion)

    enable_multiprocessing(parameters)

    return parameters
end

"""
    Parameters(
        params::Sleipnir.Parameters;
        physical::Union{<: PhysicalParameters, Nothing} = nothing,
        simulation::Union{<: SimulationParameters, Nothing} = nothing,
        solver::Union{<: SolverParameters, Nothing} = nothing,
        hyper::Union{<: Hyperparameters, Nothing} = nothing,
        UDE::Union{<: UDEparameters, Nothing} = nothing,
        inversion::Union{<: InversionParameters, Nothing} = nothing,
    )

Copy a `Parameters` struct and replace its fields by the provided values.

Arguments
=================
- params::Sleipnir.Parameters;
- `physical::Union{<: PhysicalParameters, Nothing}`: New `physical` value to replace by in the copied `Parameters` struct.
    Defaults to `nothing` which keeps the existing value.
- `simulation::Union{<: SimulationParameters, Nothing}`: New `simulation` value to replace by in the copied `Parameters` struct.
    Defaults to `nothing` which keeps the existing value.
- `solver::Union{<: SolverParameters, Nothing}`: New `solver` value to replace by in the copied `Parameters` struct.
    Defaults to `nothing` which keeps the existing value.
- `hyper::Union{<: Hyperparameters, Nothing}`: New `hyper` value to replace by in the copied `Parameters` struct.
    Defaults to `nothing` which keeps the existing value.
- `UDE::Union{<: UDEparameters, Nothing}`: New `UDE` value to replace by in the copied `Parameters` struct.
    Defaults to `nothing` which keeps the existing value.
- `inversion::Union{<: InversionParameters, Nothing}`: New `inversion` value to replace by in the copied `Parameters` struct.
    Defaults to `nothing` which keeps the existing value.
"""
function Parameters(
    params::Sleipnir.Parameters;
    physical::Union{<: PhysicalParameters, Nothing} = nothing,
    simulation::Union{<: SimulationParameters, Nothing} = nothing,
    solver::Union{<: SolverParameters, Nothing} = nothing,
    hyper::Union{<: Hyperparameters, Nothing} = nothing,
    UDE::Union{<: UDEparameters, Nothing} = nothing,
    inversion::Union{<: InversionParameters, Nothing} = nothing,
)
    physical = isnothing(physical) ? params.physical : physical
    simulation = isnothing(simulation) ? params.simulation : simulation
    solver = isnothing(solver) ? params.solver : solver
    hyper = isnothing(hyper) ? params.hyper : hyper
    UDE = isnothing(UDE) ? params.UDE : UDE
    inversion = isnothing(inversion) ? params.inversion : inversion

    return Sleipnir.Parameters(physical, simulation, hyper, solver, UDE, inversion)
end
