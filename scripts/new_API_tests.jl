
import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using ODINN
using Infiltrator
using TimerOutputs

tspan=(2010.0, 2015.0)
nglaciers = 1

function API_test(tspan)

    rgi_paths = get_rgi_paths()

    to = get_timer("ODINN")
    reset_timer!(to)

    # rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170"]
    rgi_ids = ["RGI60-11.01450"]

    # Get ODINN parameters for the simulation
    parameters = Parameters(simulation=SimulationParameters(tspan=tspan,
                                                            use_MB=true,
                                                            use_iceflow=true,
                                                            multiprocessing=true,
                                                            workers=1,
                                                            rgi_paths=rgi_paths),
                            physical=PhysicalParameters(A=2e-17),
                            solver=SolverParameters(reltol=1e-7))
    # Create an ODINN model based on a 2D Shallow Ice Approximation,
    # a TI model with 1 DDF, and a neural network
    model = Model(iceflow = SIA2Dmodel(parameters),
                  mass_balance = TImodel1(parameters; DDF=8.0/1000.0, acc_factor=1.0/1000.0),
                  machine_learning = NN(parameters))


    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, parameters)

    # We create an ODINN simulation
    prediction = Prediction(model, glaciers, parameters)

    # We run the simulation
    @timeit get_timer("ODINN") "Run prediction" run!(prediction)
    @show to

    display(ODINN.Makie.heatmap(prediction.results[1].S))
end

API_test(tspan)

include("dhdt_plots.jl")
make_plots(tspan, nglaciers)