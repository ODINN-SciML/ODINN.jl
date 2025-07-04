# # Forward simulation tutorial

# This tutorial provides a simple example on how to perform a forward simulation using ODINN.jl.

# ## Running the whole code

using ODINN

## Define the working directory
working_dir = joinpath(ODINN.root_dir, "demos")

## Ensure the working directory exists
mkpath(working_dir)

## Define which glacier RGI IDs we want to work with
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-11.02346", "RGI60-08.00203"]
rgi_paths = get_rgi_paths()

## Create the necessary parameters
params = Parameters(
    simulation = SimulationParameters(
        working_dir = working_dir,
        tspan = (2010.0, 2015.0),
        multiprocessing = true,
        workers = 5,
        rgi_paths = rgi_paths
    )
)

## Specify a model based on an iceflow model, a mass balance model, and a machine learning model
model = Huginn.Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
)

## We initialize the glaciers with all the necessary data
glaciers = initialize_glaciers(rgi_ids, params)

## We specify the type of simulation we want to perform
prediction = Prediction(model, glaciers, params)

## And finally, we just run the simulation
Huginn.run!(prediction)

## Then we can visualize the results of the simulation, e.g. the difference in ice thickness between 2010 to 2015 for Argentière glacier
pdiff = plot_glacier(prediction.results[1], "evolution difference", [:H]; metrics=["difference"])

# ## Step-by-step explanation of the tutorial

# Here we will cover in detail each one of the steps that lead us to run the
# `Prediction` from the previous example (i.e. a forward run). This first tutorial keeps things simple, and since
# we are not using machine learning models, we will only use the `Model` type to specify the iceflow and mass balance models. These functionalities
# are mainly covered by `Huginn.jl`.


# ### Step 1: Parameter initialization

# The first step is to initialize and specify all the necessary parameters. In ODINN.jl
# we have many different types of parameters, specifying different aspects of the model.
# All the parameter types come with a default constructor, which will provide default
# values in case you don't want to tune those. The main types of parameters are:

# - *Simulation parameters*: `SimulationParameters` includes all the parameters related to the
#                              ODINN.jl simulation, including the number of workers, the timespan
#                               of the simulation or the working directory.
# - *Hyperparameters*: `Hyperparameters` includes all the necessary hyperparameters for a machine learning model.
# - *UDEparameters*: `UDEparameters` contains the parameters related to the training of a Universal Differential Equation.

# All these sub-types of parameters are held in a `Parameters` struct, a general
# parameters structure to be passed to an ODINN simulation.

# First we need to specify a list of RGI IDs of the glacier we want to work with. Specifying an RGI
# region is also possible. From these RGI IDs, we will look for the necessary files inside the workspace.

rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-11.02346", "RGI60-08.00203"]
rgi_paths = get_rgi_paths()

params = Parameters(
    simulation = SimulationParameters(
        working_dir = working_dir,
        tspan = (2010.0, 2015.0),
        multiprocessing = true,
        workers = 5,
        rgi_paths = rgi_paths
    )
)


# ### Step 2: Model specification

# The next step is to specify which model(s) we want to use for our simulation. In ODINN
# we have three different types of model, which are encompassed in a `Model` structure:

# - *Iceflow model*: `IceflowModel` is the ice flow dynamics model that will be used to simulate
#                       iceflow. It defaults to a 2D Shallow Ice Approximation.
# - *Surface mass balance model*: `MassBalanceModel` is the mass balance model that will be used for
#                               simulations. Options here include temperature-index models, or
#                               machine learning models coming from `MassBalanceMachine`.
# - *Machine learning model*: `MLmodel` is the machine learning model (e.g. a neural network) which will
#                               be used as part of a hybrid model based on a Universal Differential Equation.

# The model is initialized using the `Model` constructor:

model = Huginn.Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
)


# ### Step 3: Glacier initialization

# The third step is to fetch and initialize all the necessary data for our glaciers of interest.
# This is strongly built on top of OGGM, mostly providing a Julia interface to automatize this. The package
# Gungnir is used to fetch the necessary data from the RGI and other sources. The data is then stored in servers
# and fetched and read using `Rasters.jl` directly by `Sleipnir.jl` when needed.

# Then, we initialize those glaciers based on those RGI IDs and the parameters we previously specified.
glaciers = initialize_glaciers(rgi_ids, params)


# ### Step 4: Creating and running a simulation

# The final step of the pipeline, is to create an ODINN simulation based on all the previous
# steps, and then to run it. There are different types of simulations that we can do with ODINN:

# - `Prediction`: This is a forward simulation, where the initial glacier conditions are run forward in
#                   in time based on specified parameters and climate data.

# This is as simple as doing:

prediction = Prediction(model, glaciers, params)

# And once we have the `Prediction` object, we can run it using `Huginn.run!`:
Huginn.run!(prediction)

# There we go, we have successfully simulated the evolution of 3 glaciers for 5 years in around 1-2 seconds!


# ### Step 5: Visualizing the results

# Finally, we can use the plotting functions of `ODINN.jl` to visualize the results of the simulation. Like the glacier ice thickness evolution
plot_glacier(prediction.results[1], "evolution difference", [:H]; metrics=["difference"])

# Or the initial glacier ice thickness and the resulting ice surface velocities
plot_glacier(prediction.results[1], "heatmaps", [:H, :V])

