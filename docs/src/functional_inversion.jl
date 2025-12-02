# # Functional inversion tutorial

# This tutorial provides a simple example on how to perform a functional inversion using Universal Differential Equations (UDEs) in ODINN.jl. For this, we will generate a synthetic dataset using a forward simulation, and then we will use this dataset to perform the functional inversion. The goal of this functional inversion will be to learn a synthetic law that maps `A`, i.e. the ice rigidity, to long-term changes in atmospheric surface temperature.

# For more details on the functional inversion concept, please refer to the [Functional Inversion section in the Inversion types page](./inversions.md#Functional-inversions).

# ## Running the whole code

using ODINN

## Define the working directory
working_dir = joinpath(ODINN.root_dir, "demos")

## We fetch the paths with the files for the available glaciers on disk
rgi_paths = get_rgi_paths()

## Ensure the working directory exists
mkpath(working_dir)

## Define which glacier RGI IDs we want to work with
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-11.02346", "RGI60-07.00065"]
## Define the time step for the simulation output and for the adjoint
## calculation. In this case, a month.
δt = 1/12

params = Parameters(
    simulation = SimulationParameters(
        working_dir=working_dir,
        use_MB=false,
        use_velocities=true,
        tspan=(2010.0, 2015.0),
        multiprocessing=true,
        workers=4,
        test_mode=false,
        rgi_paths=rgi_paths,
        gridScalingFactor=4), # Downscale the glacier grid to speed-up this example
    hyper = Hyperparameters(
        batch_size=length(rgi_ids), # Set batch size equals size of the dataset
        epochs=[15,10],
        optimizer=[
            ODINN.ADAM(0.01),
            ODINN.LBFGS(
                linesearch = ODINN.LineSearches.BackTracking(iterations = 5)
            )
        ]),
    physical = PhysicalParameters(
        minA = 8e-21,
        maxA = 8e-17),
    UDE = UDEparameters(
        optim_autoAD=ODINN.NoAD(),
        grad=ContinuousAdjoint(),
        optimization_method="AD+AD",
        empirical_loss_function = LossH() # Loss function based on ice thickness
    ),
    solver = Huginn.SolverParameters(
        step=δt,
        progress=true)
)

## We define a synthetic law to generate the synthetic dataset.
## For this, we use some tabular data from Cuffey and Paterson (2010).
A_law = CuffeyPaterson(scalar=true)

model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
)

## We initialize the glaciers with all the necessary data
glaciers = initialize_glaciers(rgi_ids, params)

## Time snapshots for transient inversion
tstops = collect(2010:δt:2015)

## We generate the synthetic dataset using the forward simulation.
## This will generate a dataset with the ice thickness and surface velocities for each
## glacier at each time step. The dataset will be used to train the machine learning model.
glaciers = generate_ground_truth(glaciers, params, model, tstops)

## After this forward simulation, we restart the iceflow model to be ready for the inversions
nn_model = NeuralNetwork(params)
A_law = LawA(nn_model, params)
model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
    regressors = (; A=nn_model)
)

## We specify the type of simulation we want to perform
functional_inversion = Inversion(model, glaciers, params)

## And finally, we just run the simulation
run!(functional_inversion)

# ## Step-by-step explanation of the tutorial

# Here we will cover in detail each one of the steps that lead us to run the
# `Inversion` from the previous example.
# The goal of this simple example is to learn a mapping of a law for `A`, the creep
# coefficient of ice. Mathematically, we make `A` depends on the long term air
# temperature `T` through a neural network `A=NN(T, θ)` and we optimize `θ` so that
# the generated solution matches some ice thickness reference.
# This reference is generated using the relation of the book from Cuffey and Paterson (2010) [cuffey_physics_2010](@cite).

# ### Step 1: Parameter and glacier initialization

# First we need to specify a list of RGI IDs of the glacier we want to work with.
# From these RGI IDs, we will look for the necessary files inside the workspace.
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-11.02346", "RGI60-07.00065"]
rgi_paths = get_rgi_paths()

# Define the time step for the simulation output and for the adjoint calculation. In this case, a month.
δt = 1/12

# Then we need to define the parameters of the simulation we want to perform.
# The arguments are very similar to the ones used in the [forward simulation tutorial](./forward_simulation.md) and for a complete explanation, the reader should refer to this tutorial.
# The main difference with the forward simulation tutorial here is that we need to specify the parameters for the functional inversion through the `Hyperparameters` and the `UDEparameters`.
# The `Hyperparameters` structure contains information about the optimization algorithm.
# The `UDEparameters` define how the Universal Differential Equation (UDE) is solved and how its gradient is computed.
params = Parameters(
    simulation = SimulationParameters(
        working_dir=working_dir,
        use_MB=false,
        use_velocities=true,
        tspan=(2010.0, 2015.0),
        multiprocessing=true,
        workers=4,
        test_mode=false,
        rgi_paths=rgi_paths,
        gridScalingFactor=4), # Downscale the glacier grid to speed-up this example
    hyper = Hyperparameters(
        batch_size=length(rgi_ids), # Set batch size equals size of the dataset
        epochs=[15,10],
        optimizer=[
            ODINN.ADAM(0.01),
            ODINN.LBFGS(
                linesearch = ODINN.LineSearches.BackTracking(iterations = 5)
            )
        ]),
    physical = PhysicalParameters(
        minA = 8e-21,
        maxA = 8e-17),
    UDE = UDEparameters(
        optim_autoAD=ODINN.NoAD(),
        grad=ContinuousAdjoint(),
        optimization_method="AD+AD",
        target = :A),
    solver = Huginn.SolverParameters(
        step=δt,
        progress=true)
)

# Then, we initialize those glaciers based on those RGI IDs and the parameters we previously specified.
glaciers = initialize_glaciers(rgi_ids, params)

# ### Step 2: Generate synthetic ground truth data with a forward simulation

# The next step is to generate a synthetic dataset using a forward simulation. 
# This will generate a dataset with the ice thickness and surface velocities for
# each glacier at each time step. The dataset will be used to train the machine
# learning model. We define a synthetic law to generate the synthetic dataset.
# For this, we use some tabular data from Cuffey and Paterson (2010) [cuffey_physics_2010](@cite). The REPL
# shows that it maps the long term air temperature `T` to the creep coefficient `A`.
A_law = CuffeyPaterson(scalar=true)

# The model is initialized using the `Model` constructor:
model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
)

# We define the time snapshots for transient inversion, i.e. the time steps at which we want to save the results, which will be used to compute the adjoint in reverse mode.
tstops = collect(2010:δt:2015)

prediction = Prediction(model, glaciers, params)

# We generate the synthetic dataset using the forward simulation. This will generate
# a dataset with the ice thickness and surface velocities for each glacier at each
# time step. The dataset will be used to train the machine learning model. This will
#  run under the hood a `Prediction` using [`Huginn.jl`](https://github.com/ODINN-SciML/Huginn.jl/).
glaciers = generate_ground_truth(glaciers, params, model, tstops)

# The results of this simulation are stored in the `thicknessData` field of each glacier, which we can inspect:

glaciers[1].thicknessData

# ### Step 3: Model specification to perform a functional inversion

# After this forward simulation, we define a new iceflow model to be ready for the
# inversions. The first step is to define a simple neural network that takes as
# input a scalar and returns a scalar.
nn_model = NeuralNetwork(params)

# Then we define a law that uses this neural network to map the long term air temperature `T` to the creep coefficient `A`.
# ODINN comes with a set of already defined laws. Only a few of them support functional inversion as the computation of the gradient needs to be carefully handled.
# More information about these laws can be found in the [laws tutorial](./laws.md).
law_input = (; T = iAvgScalarTemp())
A_law = LawA(nn_model, params, law_input)

# Then we define an iceflow and ODINN tells us how the law is used in the iceflow equation.
iceflow = SIA2Dmodel(params; A=A_law)

# Finally we define the model which needs to know the iceflow and mass balance models, and in comparison to Huginn, there is a third argument `regressors`.
# This `regressors` argument tells how each regressor relates into the SIA. Although we already defined this in the law, this definition is mandatory for technical reasons.
# This argument will probably disappear in the future once the code becomes more mature.
# It must match how the laws are defined in the iceflow model.
model = Model(
    iceflow = iceflow,
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
    regressors = (; A=nn_model)
)

# ### Step 4: Train a Universal Differential Equation via a functional inversion

# The next step is to specify the type of simulation we want to perform. In this case, we will use an `Inversion` simulation, which will use the synthetic dataset generated in the previous step to train a Universal Differential Equation (UDE).
functional_inversion = Inversion(model, glaciers, params)

# And finally, we just run the simulation. This will run the adjoint method to compute the gradients and then use the ADAM optimizer
# to train the UDE model.
run!(functional_inversion)

# The optimized parameters can be found in the results:
θ = functional_inversion.results.stats.θ

# And then we can visualize the learnt law by plotting the neural network mapping from temperature to A.
fig = plot_law(functional_inversion.model.iceflow.A, functional_inversion, law_input, θ)

# And then we compare it to the ground truth law used to generate the synthetic dataset.
fig = plot_law(prediction.model.iceflow.A, functional_inversion, law_input, nothing)

# Since we have a very limited range of temperature coverage with these 4 glaciers, we have only captured a small portion of the ground truth law, which has an almost linear form. If we wanted to learn the full dynamics of the Cuffey and Paterson synthetic law, we would need to have glaciers that cover a wider range of temperatures.
