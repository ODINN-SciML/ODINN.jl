
import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using ODINN
using Infiltrator

function API_test()

    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

    # Get ODINN parameters for the simulation
    parameters = Parameters()
    # Create an ODINN model based on a 2D Shallow Ice Approximation, 
    # a TI model with 1 DDF, and a neural network
    # @infiltrate
    model = Model(iceflow = SIA2Dmodel(parameters),
                  mass_balance = TImodel1(parameters),
                  machine_learning = NN(parameters))


    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, parameters)

    # We create an ODINN simulation
    prediction = Prediction(model, glaciers, parameters)

    # We run the simulation
    @time run!(prediction)
end

API_test()