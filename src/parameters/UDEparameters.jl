export UDEparameters
mutable struct UDEparameters{ADJ <: AbstractAdjointMethod} <: AbstractParameters
    sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm
    optim_autoAD::AbstractADType
    grad::Union{ADJ, Nothing}
    optimization_method::String
    loss_type::String
    empirical_loss_function::Lux.AbstractLossFunction
    scale_loss::Bool
    target::Union{String, Nothing}
end

Base.:(==)(a::UDEparameters, b::UDEparameters) = a.sensealg == b.sensealg &&
    a.optim_autoAD == b.optim_autoAD && a.grad == b.grad &&
    a.optimization_method == b.optimization_method && a.loss_type == b.loss_type &&
    a.empirical_loss_function == b.empirical_loss_function && a.scale_loss == b.scale_loss &&
    a.target == b.target


"""
    UDEparameters(;
        sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = GaussAdjoint(autojacvec=EnzymeVJP()),
        optimization_method::String = "AD+AD",
        loss_type::String = "V",
        scale_loss::Bool = true
        target::Union{String, Nothing} = "D"
        )
Initialize the parameters for the training of the UDE.
Keyword arguments
=================
    - `sensealg`: Sensitivity algorithm from SciMLSensitivity.jl to be used.
    - `optimization_method`: Optimization method for the UDE.
    - `loss_type`: Type of loss function to be used. Can be either `V` (ice velocities), or `H` (ice thickness).
    - `scale_loss`: Determines if the loss function should be scaled or not.
"""
function UDEparameters(;
            sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = GaussAdjoint(autojacvec=EnzymeVJP()),
            optim_autoAD::AbstractADType = Optimization.AutoEnzyme(),
            grad::ADJ = SciMLSensitivityAdjoint(),
            optimization_method::String = "AD+AD",
            loss_type::String = "V",
            empirical_loss_function::Lux.AbstractLossFunction = Lux.MSELoss(; agg=mean),
            scale_loss::Bool = true,
            target::Union{String, Nothing} = "D"
            ) where {ADJ <: AbstractAdjointMethod}
    # Verify that the optimization method is correct
    @assert ((optimization_method == "AD+AD") || (optimization_method == "AD+Diff")) "Wrong optimization method! Needs to be either `AD+AD` or `AD+Diff`"
    @assert ((loss_type == "V") || (loss_type == "H")) "Wrong loss type! Needs to be either `V` or `H`"

    # Build the solver parameters based on input values
    UDE_parameters = UDEparameters{typeof(grad)}(sensealg, optim_autoAD, grad, optimization_method,
                                    loss_type, empirical_loss_function, scale_loss, target)

    return UDE_parameters
end

include("InversionParameters.jl")

"""
Parameters(;
        physical::PhysicalParameters = PhysicalParameters(),
        simulation::SimulationParameters = SimulationParameters(),
        solver::SolverParameters = SolverParameters(),
        hyper::Hyperparameters = Hyperparameters(),
        UDE::UDEparameters = UDEparameters()
        inversion::InversionParameters = InversionParameters()
        )
Initialize ODINN parameters

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



