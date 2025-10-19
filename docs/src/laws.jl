# # Laws tutorial

# This tutorial provides simple examples on how to create learnable and non learnable 
# laws and how to inject them into the iceflow model.

# Let's say we have followed the classical workflow from ODINN, shown in the [Forward simulation](./forward_simulation.md) and [Functional inversion](./functional_inversion.md) tutorials. When we declare the `Model` type, we can specify the laws that we want to use in the iceflow model. Here we will briefly show how to do it. For more details you can check the [Understanding the Law interface section](./inversions.md).

using ODINN

## Dummy parameters, only specifying the type of loss function to be used
params = Parameters(UDE = UDEparameters(empirical_loss_function=LossH()))

# ## Learnable laws

# Learnable laws are laws that can be trained using a regressor. They are used to map input variables to a target variable in the iceflow model.
# In ODINN, we have implemented several learnable laws that can be used in the iceflow model.

nn_model = NeuralNetwork(params)

A_law = LawA(nn_model, params)

# The output of the law definition above states that it maps the long term air temperature `T` to a float value which corresponds to the creep coefficient `A`.
# It is defined as a neural network that takes as input the long term air temperature `T` and returns the creep coefficient `A`.
# The parameters `θ` of the neural network are learned during the inversion process, by minimizing the loss function given some target data (for this case the ice thickness).

# As explained in the [Sensitivity analysis](./sensitivity.md) section, ODINN needs to compute the vector-Jacobian products (VJPs).
# The part of the VJP concerning the law can be computed from different ways and it is possible to customize this, or use a default automatic differentiation backend.
# For this specific law the VJPs are already customized to have an efficient implementation and the user does not have to worry about this.
# The [VJP law customization](./vjp_laws.md) section provides a complete description of how this VJP computation can be customized.

# The ouput above shows that the law is applied at each iteration of the iceflow PDE.
# Additionally it says that custom VJPs are used to compute the gradient and that these VJPs are precomputed as the inputs of the law do not depend on the glacier state.

# It is then possible to visualize how the law integrates into the iceflow PDE:

model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
    regressors = (; A=nn_model)
)

# ## Non learnable laws

# Non learnable laws are laws that are not trained using a regressor. They are used to map input variables to a target variable in the iceflow model, but they do not have any learnable parameters.

# ### Example 1: Cuffey and Paterson (2010) 1-dimensional law

# Here is a quick example also drawn from the [Functional inversion](./functional_inversion.md) tutorial. We define a synthetic law to generate the synthetic dataset. For this, we use some tabular data from Cuffey and Paterson (2010).

A_law = CuffeyPaterson()

# Note that this time since there is no learnable parameter, ODINN does not need to compute the VJPs.

model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
)

# In this ice flow model, the ice rigidity `A` is defined by the `CuffeyPaterson` law, which is a non-learnable law that maps the long term air temperature `T` to the creep coefficient `A`. 

# ### Example 2: Synthetic C (sliding) 2-dimensional law

# In this example, we present a synthetic non-learnable law, that maps the basal sliding coefficient `C` to the surface topographical roughness and cumulative positive degree days (CPDDs).

using ODINN
using Plots
using Dates
using PlotlyJS

rgi_paths = get_rgi_paths()

## Retrieving simulation data for the following glaciers
rgi_ids = ["RGI60-11.03638"]
δt = 1/12

# The key part here is the definition of the law inputs, which are the variables that will be used to compute the basal sliding coefficient `C`. In this case, we use the CPDD and the topographical roughness as inputs. As you can see, there are different options to customize the way the inputs are computed. For exampe, for the CPDD, we can specify a time window over which the CPDD is integrated. For the topographical roughness, we can specify a spatial window and the type of curvature to be used.

law_inputs = (; CPDD=iCPDD(window=Week(1)), topo_roughness=iTopoRough(window=200.0, curvature_type=:variability))

# Then, we define the parameters as for any other simulation.

params = Parameters(
    simulation = SimulationParameters(
        use_MB = false,
        use_velocities = false,
        tspan = (2010.0, 2015.0),
        step = δt,
        rgi_paths = rgi_paths,
        gridScalingFactor = 4 # We reduce the size of glacier for simulation
        ),
    solver = Huginn.SolverParameters(
        step = δt,
        save_everystep = true,
        progress = true
        )
    )

# When declaring the model, we will indicate that the basal sliding coefficient `C` will be simulated by the `SyntheticC` law, which takes as input the parameters and the law inputs we defined before.

model = Huginn.Model(
    iceflow = SIA2Dmodel(params; C=SyntheticC(params; inputs=law_inputs)),
    mass_balance = nothing,
)

# We retrieve some glaciers for the simulation

glaciers = initialize_glaciers(rgi_ids, params)

# Time snapshots for transient inversion

tstops = collect(2010:δt:2015)

# Then, we can run the `generate_ground_truth_prediction` function to simulate the glacier evolution using the defined law.

prediction = generate_ground_truth_prediction(glaciers, params, model, tstops)

# Importantly, we provide the `plot_law` function to visualize 2-dimensional laws in 3D.
# This is especially useful when exploring the behaviour of laws with respect to different proxies, and to better understand learnable laws and their drivers.

fig = plot_law(prediction.model.iceflow.C, prediction, law_inputs, 1, nothing);

# Since we are in the documentation it is not possible to have an interactive plot but if you reproduce this example locally, you can run the line above without ";" and you can skip the lines hereafter. This will open an interactive window with a 3D plot that you can rotate.

folder = "laws_plots"
mkpath(folder)
filepath = joinpath(folder, "3d_plot.png")
PlotlyJS.savefig(fig, filepath);

# ```@raw html
# <img src="./laws_plots/3d_plot.png" width="500"/>
# ```