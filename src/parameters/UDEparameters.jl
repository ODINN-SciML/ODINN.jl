export UDEparameters

"""
A mutable struct that holds parameters for a UDE (Universal Differential Equation).

    UDEparameters{ADJ <: AbstractAdjointMethod} <: AbstractParameters

# Fields
- `sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm`: The sensitivity algorithm used for adjoint sensitivity analysis.
- `optimization_method::String`: The optimization method to be used.
- `loss_type::String`: The type of loss function to be used.
- `scale_loss::Bool`: A boolean indicating whether to scale the loss.
- `target::Symbol`: The target variable for the optimization.
"""
mutable struct UDEparameters{ADJ<:AbstractAdjointMethod} <: AbstractParameters
    sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm
    optim_autoAD::AbstractADType
    grad::Union{ADJ, Nothing}
    optimization_method::String
    loss_type::String
    empirical_loss_function::AbstractLoss
    scale_loss::Bool
    target::Union{Symbol, Nothing}
end

Base.:(==)(a::UDEparameters, b::UDEparameters) = a.sensealg == b.sensealg &&
    a.optim_autoAD == b.optim_autoAD && a.grad == b.grad &&
    a.optimization_method == b.optimization_method && a.loss_type == b.loss_type &&
    a.empirical_loss_function == b.empirical_loss_function && a.scale_loss == b.scale_loss &&
    a.target == b.target


"""
    UDEparameters(; sensealg, optim_autoAD, grad, optimization_method, loss_type, empirical_loss_function, scale_loss, target) where {ADJ <: AbstractAdjointMethod}

Create a `UDEparameters` object for configuring the sensitivity analysis and optimization of a Universal Differential Equation (UDE).

# Keyword Arguments
- `sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm`: The sensitivity algorithm to use for adjoint calculations. Defaults to `GaussAdjoint(autojacvec=SciMLSensitivity.EnzymeVJP())`.
- `optim_autoAD::AbstractADType`: The automatic differentiation type for optimization. Defaults to `Optimization.AutoEnzyme()`.
- `grad::ADJ`: The adjoint gradient computation method. Defaults to `SciMLSensitivityAdjoint()`.
- `optimization_method::String`: The optimization method to use. Must be either `"AD+AD"` or `"AD+Diff"`. Defaults to `"AD+AD"`.
- `loss_type::String`: The type of loss function to use. Must be either `"V"` (velocity) or `"H"` (thickness). Defaults to `"V"`.
- `empirical_loss_function::AbstractLoss`: The loss function to use for optimization. Defaults to `L2Sum()`.
- `scale_loss::Bool`: Whether to scale the loss function. Defaults to `true`.
- `target::Union{Symbol, Nothing}`: The target variable for optimization. Defaults to `:A`.

# Returns
- A `UDEparameters` object configured with the specified sensitivity, optimization, and loss settings.

# Description
This function creates a `UDEparameters` object that encapsulates the configuration for sensitivity analysis, optimization, and loss computation in a Universal Differential Equation (UDE) framework. It verifies that the provided `optimization_method` and `loss_type` are valid and constructs the solver parameters accordingly.

# Notes
- The `optimization_method` must be either `"AD+AD"` (automatic differentiation for both forward and backward passes) or `"AD+Diff"` (automatic differentiation combined with finite differences).
- The `loss_type` must be either `"V"` (velocity-based loss) or `"H"` (thickness-based loss).
- The `empirical_loss_function` determines how the loss is computed during optimization.
"""
function UDEparameters(;
        sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = GaussAdjoint(autojacvec=SciMLSensitivity.EnzymeVJP()),
        optim_autoAD::AbstractADType = Optimization.AutoEnzyme(),
        grad::ADJ = SciMLSensitivityAdjoint(),
        optimization_method::String = "AD+AD",
        loss_type::String = "V",
        empirical_loss_function::AbstractLoss = L2Sum(),
        scale_loss::Bool = true,
        target::Union{Symbol, Nothing} = :A
    ) where {ADJ <: AbstractAdjointMethod}

    # Verify that the optimization method is correct
    @assert ((optimization_method == "AD+AD") || (optimization_method == "AD+Diff")) "Wrong optimization method! Needs to be either `AD+AD` or `AD+Diff`"
    @assert ((loss_type == "V") || (loss_type == "H")) "Wrong loss type! Needs to be either `V` or `H`"

    # target_object = SIA2D_target(name = target)

    # Build the solver parameters based on input values
    UDE_parameters = UDEparameters{typeof(grad)}(
        sensealg, optim_autoAD, grad, optimization_method,
        loss_type, empirical_loss_function, scale_loss, target
    )

    return UDE_parameters
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
            UDE::UDEparameters = UDEparameters()
            inversion::InversionParameters = InversionParameters()
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
    inversion::InversionParameters = InversionParameters()
    )

    parameters = Sleipnir.Parameters(physical, simulation, hyper, solver, UDE, inversion)

    if simulation.multiprocessing
        enable_multiprocessing(parameters)
    end

    return parameters
end



