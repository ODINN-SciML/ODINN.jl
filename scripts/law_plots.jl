using Pkg
# Activate the "scripts" environment, this works both if the user is in "ODINN/", in "ODINN/scripts/" or in any subfolder
odinn_folder = split(Base.source_dir(), "scripts")[1]
Pkg.activate(odinn_folder*"/scripts/")
Pkg.develop(Pkg.PackageSpec(path = odinn_folder)) # Set ODINN in dev mode to use local version, you might do as well for Huginn, Muninn and Sleipnir

using Revise
using ODINN
using Dates

rgi_paths = get_rgi_paths()

# The value of this does not really matter, it is hardcoded in Sleipnir right now.
working_dir = joinpath(homedir(), ".OGGM/ODINN_tests")
# Re-set global constant for working directory
# const global Sleipnir.prepro_dir = joinpath(homedir(),  ".OGGM/ODINN_tests")

## Retrieving simulation data for the following glaciers
# rgi_ids = collect(keys(rgi_paths))
rgi_ids = ["RGI60-11.03638"]

# TODO: Currently there are two different steps defined in params.simulationa and params.solver which need to coincide for manual discrete adjoint
δt = 1/12
time_window = Week(1)
topo_window = 200.0
# curvature_type = :scalar
curvature_type = :variability
law_inputs = (; CPDD = iCPDD(window = time_window),
    topo_roughness = iTopoRough(window = topo_window, curvature_type = curvature_type))

params = Parameters(
    simulation = SimulationParameters(
        working_dir = working_dir,
        use_MB = false,
        use_velocities = false,
        tspan = (2010.0, 2015.0),
        multiprocessing = false,
        workers = 1,
        test_mode = false,
        rgi_paths = rgi_paths,
        gridScalingFactor = 4 # We reduce the size of glacier for simulation
    ),
    # hyper = Hyperparameters(
    #     batch_size = length(rgi_ids), # We set batch size equals all datasize so we test gradient
    #     epochs = [100,50],
    #     optimizer = [ODINN.ADAM(0.005), ODINN.LBFGS()]
    #     ),
    physical = PhysicalParameters(
        minA = 8e-21,
        maxA = 8e-17,
        minC = 8e-21,
        maxC = 8e-17 # This is the default value in Sleipnir
    ),
    # UDE = UDEparameters(
    #     optim_autoAD = ODINN.NoAD(),
    #     grad = ContinuousAdjoint(),
    #     optimization_method  ="AD+AD",
    #     target = :A
    #     ),
    solver = Huginn.SolverParameters(
        step = δt,
        progress = true
    )
)

model = Model(
    iceflow = SIA2Dmodel(params; C = SyntheticC(params; inputs = law_inputs)),
    mass_balance = nothing #TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
)

# We retrieve some glaciers for the simulation
glaciers = initialize_glaciers(rgi_ids, params)

# Time snapshots for transient inversion
tstops = collect(2010:δt:2015)

prediction = generate_ground_truth_prediction(glaciers, params, model, tstops)

### Figures

ODINN.connect_electron_backend() # If you want to visualize the plots below properly on Ubuntu 24.04

fig = plot_law(prediction.model.iceflow.C, prediction, law_inputs, nothing)
ODINN.PlotlyJS.display(fig)

fig = plot_law(
    prediction.model.iceflow.C, prediction, law_inputs, nothing; idx_fixed_input = 1)
ODINN.PlotlyJS.display(fig)

fig = plot_law(
    prediction.model.iceflow.C, prediction, law_inputs, nothing; idx_fixed_input = 2)
ODINN.PlotlyJS.display(fig)
