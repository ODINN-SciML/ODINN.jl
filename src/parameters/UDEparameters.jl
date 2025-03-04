export UDEparameters

"""
A mutable struct that holds parameters for a UDE (Universal Differential Equation).

    UDEparameters <: AbstractParameters

# Keyword arguments
- `sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm`: The sensitivity algorithm used for adjoint sensitivity analysis.
- `optimization_method::String`: The optimization method to be used.
- `loss_type::String`: The type of loss function to be used.
- `scale_loss::Bool`: A boolean indicating whether to scale the loss.
- `target::String`: The target variable for the optimization.
"""
mutable struct UDEparameters <: AbstractParameters
    sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm
    optimization_method::String
    loss_type::String
    scale_loss::Bool
    target::String
end

Base.:(==)(a::UDEparameters, b::UDEparameters) = a.sensealg == b.sensealg && a.optimization_method == b.optimization_method && a.loss_type == b.loss_type && 
                                      a.scale_loss == b.scale_loss 


"""
Function to initialize the parameters for the training of a UDE.

    UDEparameters(;
        sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = GaussAdjoint(autojacvec=EnzymeVJP()),
        optimization_method::String = "AD+AD",
        loss_type::String = "V",
        scale_loss::Bool = true
        target::String = "D"
        )

Keyword arguments
=================
    - `sensealg`: Sensitivity algorithm from SciMLSensitivity.jl to be used.
    - `optimization_method`: Optimization method for the UDE.
    - `loss_type`: Type of loss function to be used. Can be either `V` (ice velocities), or `H` (ice thickness).
    - `scale_loss`: Determines if the loss function should be scaled or not.
"""
function UDEparameters(;
            sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = GaussAdjoint(autojacvec=EnzymeVJP()),
            optimization_method::String = "AD+AD",
            loss_type::String = "V",
            scale_loss::Bool = true,
            target::String = "D"
            )
    # Verify that the optimization method is correct
    @assert ((optimization_method == "AD+AD") || (optimization_method == "AD+Diff")) "Wrong optimization method! Needs to be either `AD+AD` or `AD+Diff`"
    @assert ((loss_type == "V") || (loss_type == "H")) "Wrong loss type! Needs to be either `V` or `H`"

    # Build the solver parameters based on input values
    UDE_parameters = UDEparameters(sensealg, optimization_method,
                                    loss_type, scale_loss, target)

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



