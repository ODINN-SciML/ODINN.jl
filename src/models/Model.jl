
export Model

## Model subtypes
include("iceflow/IceflowModel.jl")
include("mass_balance/MBmodel.jl")
include("machine_learning/MLmodel.jl")

# Composite type as a representation of ODINN models
struct Model{IF <: IceflowModel, MB <: MBmodel, ML <: MLmodel}
    iceflow::IF
    mass_balance::MB
    machine_learning::ML
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
            iceflow::IF,
            mass_balance::MB,
            machine_learning::ML
            ) where {IF <: IceflowModel, MB <: MBmodel, ML <: MLmodel}

    # Build an ODINN model based on the iceflow, MB and ML models
    model = Model(iceflow, mass_balance, machine_learning)

    return model
end

###############################################
################### UTILS #####################
###############################################

include("iceflow/iceflow_utils.jl")
include("mass_balance/mass_balance_utils.jl")
include("machine_learning/ML_utils.jl")
