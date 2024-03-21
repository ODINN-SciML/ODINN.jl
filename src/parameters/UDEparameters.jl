
  mutable struct UDEparameters <: AbstractParameters
    sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm
    optimization_method::String
    loss_type::String
    scale_loss::Bool
end

Base.:(==)(a::UDEparameters, b::UDEparameters) = a.sensealg == b.sensealg && a.optimization_method == b.optimization_method && a.loss_type == b.loss_type && 
                                      a.scale_loss == b.scale_loss 


"""
    UDEparameters(;
        sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
        optimization_method::String = "AD+AD",
        loss_type::String = "V",
        scale_loss::Bool = true
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
            sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
            optimization_method::String = "AD+AD",
            loss_type::String = "V",
            scale_loss::Bool = true
            )
    # Verify that the optimization method is correct
    @assert ((optimization_method == "AD+AD") || (optimization_method == "AD+Diff")) "Wrong optimization method! Needs to be either `AD+AD` or `AD+Diff`"
    @assert ((loss_type == "V") || (loss_type == "H")) "Wrong loss type! Needs to be either `V` or `H`"

    # Build the solver parameters based on input values
    UDE_parameters = UDEparameters(sensealg, optimization_method,
                                    loss_type, scale_loss)

    return UDE_parameters
end

include("Inversionparameters.jl")

"""
Parameters(;
        physical::PhysicalParameters = PhysicalParameters(),
        simulation::SimulationParameters = SimulationParameters(),
        OGGM::OGGMparameters = OGGMparameters(),
        solver::SolverParameters = SolverParameters(),
        hyper::Hyperparameters = Hyperparameters(),
        UDE::UDEparameters = UDEparameters()
        inversion::InversionParameters = InversionParameters()
        )
Initialize ODINN parameters

Keyword arguments
=================
    
"""
function Parameters(;
    physical::PhysicalParameters = PhysicalParameters(),
    simulation::SimulationParameters = SimulationParameters(),
    OGGM::OGGMparameters = OGGMparameters(),
    solver::SolverParameters = SolverParameters(),
    hyper::Hyperparameters = Hyperparameters(),
    UDE::UDEparameters = UDEparameters(),
    inversion::InversionParameters = InversionParameters()  
    ) 

    
    parameters = Sleipnir.Parameters(physical, simulation, OGGM, hyper, solver, UDE, inversion)  

    if simulation.multiprocessing
        enable_multiprocessing(parameters)
    end
    # Config OGGM *after* setting multirprocessing 
    # to ensure it is initialized in all workers
    oggm_config(OGGM.working_dir; oggm_processes=OGGM.workers)


    return parameters
end



