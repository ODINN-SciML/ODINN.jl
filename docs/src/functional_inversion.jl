# ## Forward simulation tutorial

## This tutorial provides a simple example on how to perform a functional inversion using Universal Differential Equations (UDEs) in ODINN.jl.
## For this, we will generate a synthetic dataset using a forward simulation, and then we will use this dataset to perform the functional inversion.
## The goal of this functional inversion will be to learn a synthetic law that maps `A`, i.e. the ice rigidity, to long-term changes in atmospheric surface temperature. 

using ODINN

## Define the working directory
working_dir = joinpath(homedir(), "ODINN_simulations")

## We fetch the paths with the files for the available glaciers on disk
rgi_paths = get_rgi_paths()

## Ensure the working directory exists
mkpath(working_dir)

## Define which glacier RGI IDs we want to work with
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-11.02346", "RGI60-07.00065", "RGI60-08.00147","RGI60-07.00042"]
## Filter out glaciers that are not used to avoid having references that depend on all the glaciers processed in Gungnir
rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)
## Define the time step for the simulation output and for the adjoint calculation. In this case, a month. 
δt = 1/12

params = Parameters(simulation = SimulationParameters(working_dir=working_dir,
                                                    use_MB=false,
                                                    velocities=true,
                                                    tspan=(2010.0, 2015.0),
                                                    step=δt,
                                                    multiprocessing=false,
                                                    workers=1,
                                                    light=false, 
                                                    test_mode=false,
                                                    rgi_paths=rgi_paths),
                    hyper = Hyperparameters(batch_size=length(rgi_ids), # We set batch size equals all datasize so we test gradient
                                            epochs=[50,50],
                                            optimizer=[ODINN.ADAM(0.005), ODINN.LBFGS()]),
                    physical = PhysicalParameters(minA = 8e-21,
                                                  maxA = 8e-17),
                    UDE = UDEparameters(optim_autoAD=ODINN.NoAD(),
                                        grad=ContinuousAdjoint(),
                                        optimization_method="AD+AD",
                                        target = "A"),
                    solver = Huginn.SolverParameters(step=δt,
                                                     save_everystep=true,
                                                     progress=true)
                    )

model = Model(iceflow = SIA2Dmodel(params),
                mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
                machine_learning = NeuralNetwork(params))

## We initialize the glaciers with all the necessary data 
glaciers = initialize_glaciers(rgi_ids, params)

## Time snapshots for transient inversion
tstops = collect(2010:δt:2015)

## We define a synthetic law to generate the synthetic dataset. For this, we use some tabular data from Cuffey and Paterson (2010).
A_poly = ODINN.A_law_PatersonCuffey()
fakeA(T) = A_poly(T)

## We generate the synthetic dataset using the forward simulation. This will generate a dataset with the ice thickness and surface velocities
## for each glacier at each time step. The dataset will be used to train the machine learning model.
ODINN.generate_ground_truth(glaciers, :PatersonCuffey, params, model, tstops)

## After this forward simulation, we restart the iceflow model to be ready for the inversions
model.iceflow = SIA2Dmodel(params)

## We specify the type of simulation we want to perform
functional_inversion = FunctionalInversion(model, glaciers, params)

## And finally, we just run the simulation
run!(functional_inversion)


# # ### Step-by-step explanation of the tutorial

# # Here we will cover in detail each one of the steps that lead us to run the 
# # `Prediction` from the previous example (i.e. a forward run). This first tutorial keeps things simple, and since 
# # we are not using machine learning models, we will only use the `Model` type to specify the iceflow and mass balance models. These functionalities
# # are mainly covered by `Huginn.jl`. 

# # #### Step 1: Parameter initialization

# # The first step is to initialize and specify all the necessary parameters. In ODINN.jl
# # we have many different types of parameters, specifying different aspects of the model.
# # All the parameter types come with a default constructor, which will provide default
# # values in case you don't want to tune those. The main types of parameters are:

# # - *Simulation parameters*: `SimulationParameters` includes all the parameters related to the
# #                              ODINN.jl simulation, including the number of workers, the timespan
# #                               of the simulation or the working directory.
# # - *PhysicalParameters*: `PhysicalParameters` includes all the necessary physical parameters for the model.
# # - *SolverParameters*: `SolverParameters` includes all the necessary parameters for the solver.
# # - *Hyperparameters*: `Hyperparameters` includes all the necessary hyperparameters for a machine learning model.
# # - *UDEparameters*: `UDEparameters` contains the parameters related to the training of a Universal Differential Equation.

# # All these sub-types of parameters are held in a `Parameters` struct, a general 
# # parameters structure to be passed to an ODINN simulation.

# # First we need to specify a list of RGI IDs of the glacier we want to work with. Specifying an RGI
# # region is also possible. From these RGI IDs, we will look for the necessary files inside the workspace.

# rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-11.02346", "RGI60-07.00065", "RGI60-08.00147","RGI60-07.00042"]
# rgi_paths = get_rgi_paths()
# # Filter out glaciers that are not used to avoid having references that depend on all the glaciers processed in Gungnir
# rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)
# ## Define the time step for the simulation output and for the adjoint calculation. In this case, a month. 
# δt = 1/12

# params = Parameters(simulation = SimulationParameters(working_dir=working_dir,
#                                                     use_MB=false,
#                                                     velocities=true,
#                                                     tspan=(2010.0, 2015.0),
#                                                     step=δt,
#                                                     multiprocessing=true,
#                                                     workers=7,
#                                                     light=false, 
#                                                     test_mode=false,
#                                                     rgi_paths=rgi_paths),
#                     hyper = Hyperparameters(batch_size=length(rgi_ids), # We set batch size equals all datasize so we test gradient
#                                             epochs=[50,50],
#                                             optimizer=[ODINN.ADAM(0.005), ODINN.LBFGS()]),
#                     physical = PhysicalParameters(minA = 8e-21,
#                                                   maxA = 8e-17),
#                     UDE = UDEparameters(optim_autoAD=ODINN.NoAD(),
#                                         grad=ContinuousAdjoint(),
#                                         optimization_method="AD+AD",
#                                         target = "A"),
#                     solver = Huginn.SolverParameters(step=δt,
#                                                      save_everystep=true,
#                                                      progress=true)
#                     )

# # #### Step 2: Model specification

# # The next step is to specify which model(s) we want to use for our simulation. In ODINN
# # we have three different types of model, which are encompassed in a `Model` structure:

# # - *Iceflow model*: `IceflowModel` is the ice flow dynamics model that will be used to simulate
# #                       iceflow. It defaults to a 2D Shallow Ice Approximation.
# # - *Surface mass balance model*: `MassBalanceModel` is the mass balance model that will be used for 
# #                               simulations. Options here include temperature-index models, or 
# #                               machine learning models coming from `MassBalanceMachine`.
# # - *Machine learning model*: `MLmodel` is the machine learning model (e.g. a neural network) which will
# #                               be used as part of a hybrid model based on a Universal Differential Equation.

# # Generally, a model can be initialized directly using the `Model` constructor:

# model = Model(iceflow = SIA2Dmodel(params),
#                 mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
#                 machine_learning = NeuralNetwork(params))

# # #### Step 3: Glacier initialization

# # The third step is to fetch and initialize all the necessary data for our glaciers of interest.
# # This is strongly built on top of OGGM, mostly providing a Julia interface to automatize this. The package
# # Gungnir is used to fetch the necessary data from the RGI and other sources. The data is then stored in servers 
# # and fetched and read using `Rasters.jl` directly by `Sleipnir.jl` when needed.

# # Then, we initialize those glaciers based on those RGI IDs and the parameters we previously specified.
# glaciers = initialize_glaciers(rgi_ids, params)

# # #### Step 4: Running a forward simulation as a synthetic ground truth 

# # The next step is to generate a synthetic dataset using a forward simulation. This will generate a dataset with the ice thickness and surface velocities
# # for each glacier at each time step. The dataset will be used to train the machine learning model.

# # We define the time snapshots for transient inversion, i.e. the time steps at which we want to save the results, which will be used
# # to compute the adjoint in reverse mode.
# tstops = collect(2010:δt:2015)

# prediction = Prediction(model, glaciers, params)

# # We define a synthetic law to generate the synthetic dataset. For this, we use the data from a table in Cuffey and Paterson (2010).
# A_poly = ODINN.A_law_PatersonCuffey()
# fakeA(T) = A_poly(T)

# # We generate the synthetic dataset using the forward simulation. This will generate a dataset with the ice thickness and surface velocities
# # for each glacier at each time step. The dataset will be used to train the machine learning model. This will run under the hood
# # a `Prediction` using `Huginn.jl`. 
# ODINN.generate_ground_truth(glaciers, :PatersonCuffey, params, model, tstops)

# # After this forward simulation, we restart the iceflow model to be ready for the inversions
# model.iceflow = SIA2Dmodel(params)

# # #### Step 5: Train a Universal Differential Equation via a functional inversion
# # The next step is to specify the type of simulation we want to perform. In this case, we will use a `FunctionalInversion` simulation,
# # which will use the synthetic dataset generated in the previous step to train a Universal Differential Equation (UDE) model.

# functional_inversion = FunctionalInversion(model, glaciers, params)

# # And finally, we just run the simulation. This will run the adjoint method to compute the gradients and then use the ADAM optimizer
# # to train the UDE model. 
# run!(functional_inversion)

