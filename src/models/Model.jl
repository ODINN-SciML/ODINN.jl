
include("iceflow/IceflowModel.jl")
include("mass_balance/MBmodel.jl")
include("machine_learning/MLmodel.jl")

# Composite type as a representation of ODINN models
@kwdef struct Model
    iceflow::IceflowModel
    mass_balance::MBmodel
    machine_learning::MLmodel
end

"""
    Model(;
        iceflow::IceflowModel = SIA2Dmodel(),
        mass_balance::MBmodel = TImodel1(),
        machine_learning::MLmodel = NN()
        )
Constructs an ODINN model based on an iceflow model, a mass balance model and a machine learning model.

Keyword arguments
=================
    - `iceflow`: `IceflowModel` to be used for the ice flow dynamics simulations.
    - `mass_balance`: `MBmodel` to be used for the surface mass balance simulations.
    - `machine_learning`: Machine learning model to be used for the UDEs.
"""
function Model(;
            iceflow::IceflowModel = SIA2Dmodel(),
            mass_balance::MBmodel = TImodel1(),
            machine_learning::MLmodel = NN()
            )

    # Build the simulation parameters based on input values
    model = Model(iceflow = iceflow, 
                  mass_balance = mass_balance,
                  machine_learning = machine_learning)

    return model
end