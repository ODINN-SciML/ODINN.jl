# # Transient inversion tutorial

# This tutorial shows how to run a transient inversion with real observations.
# "Transient" means that we compare the whole trajectory of the glacier with observations made at different dates, not only the final state.
# We use Aletsch, the largest glacier of the European Alps (`RGI60-11.02773`).
# It has a rich observation record: ice thickness surveys from GlaThiDa, dated DEMs of the surface, and one surface velocity map from [millan_ice_2022](@cite).

# We invert the initial state of the glacier: the initial ice thickness and, with it, the bed of the glacier.
# The surface of a glacier is well observed (DEMs cover all of it), but the bed is only seen through a few thickness surveys.
# So we treat the surface as an observation and the bed as an unknown: this is the "free bed" of [Inversion types](./inversions.md).
# The ice creep coefficient `A` is an unknown too: the inversion adjusts it, together with the initial state, to fit all the observations at once.

# ## Step 1: The loss and the parameters

using ODINN

# We fetch the paths with the files for the available glaciers on disk

rgi_paths = get_rgi_paths()

# Define which glacier RGI IDs we want to work with

rgi_ids = ["RGI60-11.02773"]

# The loss defines the misfit between the simulation and the observations. We combine several terms with a `MultiLoss`:
# - `LossH`: the ice thickness surveys, compared at their dates. The surveys are sparse, so the loss uses every observed pixel, without any margin from the glacier border (`distance = 0`),
# - `LossAvgV`: the time average of the simulated surface velocities, compared with the velocity map over its acquisition window,
# - `LossS`: the DEMs, compared with the model surface at the date of each DEM,
# - `InitialThicknessRegularization`: a prior that keeps the changes to the initial thickness smooth,
# - `BedRegularization`: a prior that keeps the bed smooth.
# Each term is a mean of `(misfit / σ)²`, where `σ` is the uncertainty of the observation, so the terms are comparable and the weights are 1 by default.

loss = MultiLoss(
    losses = (
    LossH(loss = L2Sum(distance = 0)),
    LossAvgV(),
    LossS(),
    InitialThicknessRegularization(),
    BedRegularization()
)
)

# We now define the parameters used for the simulation

params = Parameters(
    simulation = SimulationParameters(
        use_MB = true, # the surface mass balance is on: the glacier can grow and shrink
        use_velocities = true, # load the surface velocity map
        use_glathida_data = true, # load the GlaThiDa ice thickness surveys
        tspan = (2013.0, 2018.0), # the window that contains the observations we use; the start is moved to the first observation below
        multiprocessing = false, # we are processing only one glacier
        rgi_paths = rgi_paths,
        gridScalingFactor = 4, # coarser grid: the example stays cheap to run
        velocity_product = :Millan22,
        ice_thickness_source = :Farinotti19 # starting thickness: the consensus estimate of [farinotti_consensus_2019](@cite)
    ),
    hyper = Hyperparameters(
        batch_size = length(rgi_ids), # one glacier per batch
        epochs = [5, 5], # [40, 60] give a better fit, at the cost of a longer run
        optimizer = [
            ODINN.Adam(0.05),
            ODINN.LBFGS(
                linesearch = ODINN.LineSearches.BackTracking(iterations = 5)
            )
        ]
    ),
    physical = PhysicalParameters(),
    UDE = UDEparameters(
        grad = SciMLSensitivityAdjoint(), # the automatic adjoint: the free bed needs it
        optim_autoAD = ODINN.Optimization.AutoZygote(),
        empirical_loss_function = loss,
        initial_condition_filter = :Zang1980 # keeps the initial ice thickness non-negative
    ),
    solver = Huginn.SolverParameters(
        step = 1 / 12, # save the simulation every month
        solver = Huginn.ROCK2() # a stabilized explicit solver: the backward pass of the adjoint needs one
    )
)

# ## Step 2: The glacier and its observations

# We initialize the glacier with all its data: the climate (W5E5), the thickness product, the observations. The data comes from the preprocessing of the glacier, see [Glaciers](./glaciers.md).

glaciers = initialize_glaciers(rgi_ids, params)
glacier = glaciers[1]

# The observations attached to the glacier are listed by `observations`, sorted in time:

observations(glacier)

# Each row is one observation. The field `kind` says what it is: `:H` for an ice thickness survey, `:V` for a surface velocity map, `:S` for a DEM and `:dhdt` for the glacier-wide surface elevation change. The field `t` is its date, in decimal years.
# For the velocity, `t_start` and `t_end` are the acquisition window: the Millan22 map is an average over 2017-2018, which is why `LossAvgV` compares the average of the simulated velocity over that window.

# The time window of the glacier is, by default, the `tspan` of the parameters:

glacier_tspan(params, glacier)

# It is best to start the simulation on an observation: the initial state is then constrained by data directly, and not only through the ice flow of the first time steps.
# `align_start!` moves the start of each glacier to its first observation inside the window (the glacier-wide `:dhdt` does not count), and writes the new window in `params.simulation.glacier_tspans`.
# Here the first observation inside the window is the Copernicus DEM, so the simulation starts at its date and the DEM constrains the initial surface directly:

align_start!(params, glaciers)

glacier_tspan(params, glacier)

# The `Inversion` below also does this on its own, but only when the first observation is at most one mass balance step after the start.

# ## Step 3: The model: what we invert

# The creep coefficient `A` is one of the unknowns: the inversion adjusts it as a map, cell by cell, so the rheology of the ice can vary in space.
# The law behind it, `LawA`, maps the optimized parameter to a physical `A`: a `tanh` map keeps every value between `minA` and `maxA`, the bounds of the physical parameters.
# The inversion starts from the reference `A` of the glacier.
# A law like `CuffeyPaterson` maps the long-term air temperature to `A` through a polynomial fit of the tabulated flow-rate values [cuffey_physics_2010](@cite).
# It is a good first guess for a forward simulation, but it has no parameter to optimize: an inverted `A` needs `LawA`.

trainable_A = GriddedInv(params, glaciers, :A)
A_law = LawA(params; scalar = false)

# The other trainable part is the initial condition `ic`. It starts from the Farinotti19 thickness and from the DEM of the glacier. With `bed = :free` (the default), the unknowns are the initial ice thickness and the initial surface of each cell, and the bed is what is left: `B = S₀ - H₀`. A thickness survey at any date then constrains the bed, and the DEMs keep the surface in place.
# The outline of the glacier is a hard limit for the ice: no ice outside it, at least `min_thickness` (1 m) inside it.

ic = InitialCondition(params, glaciers, :Farinotti19) # bed = :free

# The model combines the ice flow, the mass balance and the two trainable parts.
# The mass balance is a temperature-index (degree-day) model with one degree-day factor, driven by the W5E5 climate of the glacier.
# We do not set its parameters by hand: with `calibrate_MB = true` (the default), the `Inversion` constructor below calibrates the model of each glacier against its geodetic mass balance, given by the surface elevation changes of [hugonnet_accelerated_2021](@cite).

model = Model(
    iceflow = SIA2Dmodel(params; A = A_law),
    mass_balance = TImodel1(params),
    regressors = (; A = trainable_A, IC = ic)
)

# ## Step 4: Run the inversion

# We specify the type of simulation we want to perform

inversion = Inversion(model, glaciers, params)

# The construction checks the setup and prints a warning about the outline of the glacier: the RGI outline of Aletsch is from 2003, while the simulation starts in 2013.
# The outline is a hard limit for the ice, so an outline ten years older than the start is a real caveat, and the warning is here to remind us of it.

# And we run it. The optimizer runs in two phases: Adam first, then LBFGS. Each iteration evaluates the loss and its gradient, which needs one forward simulation and one adjoint (backward) pass per glacier.

run!(inversion)

# ## Step 5: The results

# The optimized parameters are in `inversion.results.stats`. Inside them, the initial thickness is stored divided by a scale, so read it in metres with `ODINN.initial_thickness`:

θ = inversion.results.stats.θ
H₀ = ODINN.initial_thickness(θ, inversion, 1)

# The results of the simulation hold the inverted bed, `B = S₀ - H₀`:

results = inversion.results.simulation[1]
B_inverted = results.B

# `plot_inverted_fields` draws the maps of the inverted fields: the initial ice thickness, the bed, and the inverted `A`:

plot_inverted_fields(inversion)

# The value of the loss at each iteration is in `inversion.results.stats.losses`. The loss decreases as the fit improves. Without the priors, the inversion could fit the observations at the cost of an unphysical bed; the priors prevent this.

losses = inversion.results.stats.losses
