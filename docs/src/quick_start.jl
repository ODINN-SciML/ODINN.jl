# # Quick start

# Before going into model details, it is always better to have a quick overview with a simple example on how `ODINN.jl` 
# is used from a user point of view. This is the simplest example of how to create and run a simple glacier forward simulation:

using ODINN

working_dir = joinpath(ODINN.root_dir, "demos")
mkpath(working_dir)

rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-11.02346", "RGI60-08.00203"]
rgi_paths = get_rgi_paths()

params = Parameters(
    simulation = SimulationParameters(
        working_dir = working_dir,
        tspan = (2010.0, 2015.0),
        multiprocessing=false,
        rgi_paths = rgi_paths
    )
)

model = Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
)

glaciers = initialize_glaciers(rgi_ids, params)
prediction = Prediction(model, glaciers, params)
Huginn.run!(prediction)

# This code will run a forward simulation for the glaciers defined in `rgi_ids` from 2010 to 2015, 
# using the specified ice flow and mass balance models. The results will be stored in the `working_dir` directory.

# Then, we can easily visualize the results of the simulation, e.g. the difference in ice thickness between 2010 to 2015 for Argentière glacier:

pdiff = plot_glacier(prediction.results[1], "evolution difference", [:H]; metrics=["difference"])