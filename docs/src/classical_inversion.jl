# # Classical inversion tutorial

# This tutorial provides a simple example on how to perform a classical gridded inversion in `ODINN.jl`.
# For this, we generate a synthetic dataset using a forward simulation, and then we use this dataset to perform the classical inversion.
# The goal of this classical inversion is to retrieve the matrix of `A` values, i.e. the ice rigidity, that was used to generate the results of a forward simulation.

# ## Step 1: Parameter and glacier initialization

using ODINN

# We fetch the paths with the files for the available glaciers on disk

rgi_paths = get_rgi_paths()

# Define which glacier RGI IDs we want to work with

rgi_ids = ["RGI60-11.03638"]

# Define the time step for the simulation output and for the adjoint calculation. In this case, a month.

δt = 1/12

params = Parameters(
    simulation = SimulationParameters(
        use_MB=false,
        tspan=(2010.0, 2015.0),
        test_mode=false,
        multiprocessing=false, # We are processing only one glacier
        rgi_paths=rgi_paths,
        gridScalingFactor=4), # Downscale the glacier grid to speed-up this example for the GitHub servers
    hyper = Hyperparameters(
        batch_size=length(rgi_ids), # Set batch size equals size of the dataset
        epochs=[2,2], # [35,30]
        optimizer=[
            ODINN.ADAM(0.02),
            ODINN.LBFGS(
                linesearch = ODINN.LineSearches.BackTracking(iterations = 5)
            )
        ]),
    physical = PhysicalParameters(
        minA = 8e-21,
        maxA = 8e-17),
    UDE = UDEparameters(
        optim_autoAD=ODINN.NoAD(),
        empirical_loss_function = LossH() # Loss function based on ice thickness
    ),
    solver = Huginn.SolverParameters(step=δt),
)

# ## Step 2: Generate synthetic ground truth data with a forward simulation

# We define a synthetic law to generate the synthetic dataset. For this, we use some tabular data from Cuffey and Paterson (2010) [cuffey_physics_2010](@cite) in a law that we have already available in `ODINN.jl`, which we specify to be in a gridded format which means that it varies spatially (i.e. non-scalar).

A_law = CuffeyPaterson(scalar=false)

model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
)

# We initialize the glaciers with all the necessary data

glaciers = initialize_glaciers(rgi_ids, params)

# Time snapshots where to store data for the inversion

tstops = collect(2010:δt:2015)

# We generate the synthetic dataset using the forward simulation. This will generate a dataset with the ice thickness and surface velocities for each glacier at each time step. The dataset will be used to make the inversion hereafter.

prediction = generate_ground_truth_prediction(glaciers, params, model, tstops)

glaciers = prediction.glaciers

# Now we compute the spatially varying `A` to have a ground truth for the comparison at the end of this tutorial.

A_ground_truth = zeros(size(prediction.glaciers[1].H₀))
inn1(A_ground_truth) .= eval_law(prediction.model.iceflow.A, prediction, 1, (;T=get_input(iAvgGriddedTemp(), prediction, 1, tstops[1])), nothing)
A_ground_truth[prediction.glaciers[1].H₀.==0] .= NaN;

# ## Step 3: Model specification to perform a classical inversion

# After this forward simulation, we restart the iceflow model to be ready for the inversions

trainable_model = GriddedInv(params, glaciers, :A)
A_law = LawA(params; scalar=false)
model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
    regressors = (; A=trainable_model)
)

# ## Step 4: Perform the inversion by optimizing the model

# We specify the type of simulation we want to perform

inversion = Inversion(model, glaciers, params)

# And finally, we just run the simulation

run!(inversion)

# Now that the model has been optimized, we retrieve the inverted parameters. These parameters do not correspond directly to the values of `A`. What it defines instead is a parameterization of `A` to ensure positiveness through a tanh function.

θ = inversion.results.stats.θ

# We map the parameters to the values of `A` by evaluating the law.

A = zeros(size(inversion.glaciers[1].H₀))
inn1(A) .= eval_law(inversion.model.iceflow.A, inversion, 1, (;), θ)
A[inversion.glaciers[1].H₀.==0] .= NaN;

# ## Step 5: Compare the inverted parameter to the synthetic ground truth

# Finally we visualize the inverted `A`.

plot_gridded_data(A, inversion.results.simulation[1]; colormap=:YlGnBu, logPlot=true)

# We can compare it to the ground truth `A` values:

plot_gridded_data(A_ground_truth, inversion.results.simulation[1]; colormap=:YlGnBu, logPlot=true)

# Unsurprisingly the inverted A is noisy in comparison to the ground truth. This is because the inversion requires regularization. For more information on how to define regularizations, see the [Optimization](./optimization.md) section.
