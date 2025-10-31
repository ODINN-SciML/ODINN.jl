using Pkg
# Activate the "scripts" environment, this works both if the user is in "ODINN/", in "ODINN/scripts/" or in any subfolder
odinn_folder = split(Base.source_dir(), "scripts")[1]
Pkg.activate(odinn_folder*"/scripts/")
Pkg.develop(Pkg.PackageSpec(path = odinn_folder)) # Set ODINN in dev mode to use local version, you might do as well for Huginn, Muninn and Sleipnir

using Revise
using ODINN

rgi_paths = get_rgi_paths()

# The value of this does not really matter, it is hardcoded in Sleipnir right now.
working_dir = joinpath(homedir(), ".OGGM/ODINN_tests")
# Re-set global constant for working directory
# const global Sleipnir.prepro_dir = joinpath(homedir(),  ".OGGM/ODINN_tests")


## Retrieving simulation data for the following glaciers
# rgi_ids = collect(keys(rgi_paths))
# rgi_ids = ["RGI60-11.03638"]
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]
# rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]
# rgi_ids = ["RGI60-11.03638",
#             "RGI60-11.01450",
#             "RGI60-08.00213",
#             "RGI60-04.04351",
#             "RGI60-01.02170",
#             "RGI60-02.05098",
#             "RGI60-01.01104",
#             "RGI60-01.09162",
#             "RGI60-01.00570", # This one does not have millan_v data
#             "RGI60-04.07051",
#             "RGI60-07.00274",
#             "RGI60-07.01323",#],
#             "RGI60-01.17316"] # This one does not have millan_v data

# TODO: Currently there are two different steps defined in params.simulationa and params.solver which need to coincide for manual discrete adjoint
δt = 1/12

params = Parameters(
    simulation = SimulationParameters(
        working_dir = working_dir,
        use_MB = false,
        use_velocities = false,
        tspan = (2010.0, 2015.0),
        step = δt,
        multiprocessing = false,
        workers = 1,
        test_mode = false,
        rgi_paths = rgi_paths,
        gridScalingFactor = 4 # We reduce the size of glacier for simulation
        ),
    hyper = Hyperparameters(
        batch_size = length(rgi_ids), # We set batch size equals all datasize so we test gradient
        epochs = [2, 2],
        optimizer = [ODINN.ADAM(0.005), ODINN.LBFGS()]
        ),
    physical = PhysicalParameters(
        minA = 8e-21,
        maxA = 8e-17
        ),
    UDE = UDEparameters(
        optim_autoAD = ODINN.NoAD(),
        grad = ContinuousAdjoint(),
        optimization_method = "AD+AD",
        empirical_loss_function = MultiLoss(
            losses = (LossH(), InitialThicknessRegularization()),
            λs = (1.0, 1e-4)
            ),
        target = :A,
        initial_condition_filter = :Zang1980
        ),
    solver = Huginn.SolverParameters(
        step = δt,
        save_everystep = true,
        progress = true
        )
    )

model = Model(
    iceflow = SIA2Dmodel(params; A=CuffeyPaterson()),
    mass_balance = nothing, #TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
)

# We retrieve some glaciers for the simulation
glaciers = initialize_glaciers(rgi_ids, params)

# Time snapshots for transient inversion
tstops = collect(2010:δt:2015)

glaciers = generate_ground_truth(glaciers, params, model, tstops)

nn_model = NeuralNetwork(params)

# Decide if we want or not to learn initial condition
train_initial_conditions = true

if train_initial_conditions
    ic = InitialCondition(params, glaciers, :Farinotti2019)
    model = Model(
        iceflow = SIA2Dmodel(params; A = LawA(nn_model, params)),
        mass_balance = nothing,
        regressors = (; A = nn_model, IC = ic)
    )
else
    # ic = ODINN.emptyIC()
    model = Model(
        iceflow = SIA2Dmodel(params; A = LawA(nn_model, params)),
        mass_balance = nothing,
        regressors = (; A = nn_model)
    )
end



# We create an ODINN prediction
functional_inversion = FunctionalInversion(model, glaciers, params)

# We run the simulation with ADAM and then LBFGS
# run!(functional_inversion)

### Figures
