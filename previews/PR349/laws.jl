# # Laws tutorial

## This tutorial provides simple examples on how to create laws and how to inject them into the iceflow model.

using ODINN

## Define the working directory
working_dir = joinpath(ODINN.root_dir, "demos")

## We fetch the paths with the files for the available glaciers on disk
rgi_paths = get_rgi_paths()

## Ensure the working directory exists
mkpath(working_dir)

## Define which glacier RGI IDs we want to work with
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-11.02346", "RGI60-07.00065"]
## Define the time step for the simulation output and for the adjoint calculation. In this case, a month.
δt = 1/12

params = Parameters(
    simulation = SimulationParameters(
        working_dir=working_dir,
        use_MB=false,
        velocities=true,
        tspan=(2010.0, 2015.0),
        step=δt,
        multiprocessing=true,
        workers=5,
        test_mode=false,
        rgi_paths=rgi_paths),
    hyper = Hyperparameters(
        batch_size=length(rgi_ids), # We set batch size equals all datasize so we test gradient
        epochs=[2,1],
        optimizer=[ODINN.ADAM(0.01), ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5))]),
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
        save_everystep=true,
        progress=true)
)

# ## Non learnable laws

# We define a synthetic law to generate the synthetic dataset. For this, we use some tabular data from Cuffey and Paterson (2010).
A_law = CuffeyPaterson()

model = Huginn.Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
)

# ## Learnable laws

