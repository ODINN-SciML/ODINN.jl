function inversion_test(;
    use_MB = false,
    multiprocessing = false,
    grad = ContinuousAdjoint(),
    functional_inv = true,
)

    rgi_paths = get_rgi_paths()
    # The value of this does not really matter, it is hardcoded in Sleipnir right now.
    working_dir = joinpath(homedir(), ".OGGM/ODINN_tests")

    ## Retrieving simulation data for the following glaciers
    if multiprocessing && functional_inv
        workers = 3 # Two processes for the two glaciers + one for main
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]
        # Multiprocessing is especially slow in the CI, so we perform a very short optimization
        epochs = 3
        optimizer = ODINN.ADAM(0.01)
    elseif functional_inv
        workers = 1
        rgi_ids = ["RGI60-11.03638"]
        epochs = [20,20]
        optimizer = [ODINN.ADAM(0.005), ODINN.LBFGS()]
    else
        workers = 1
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]
        epochs = [5,7]
        optimizer = [ODINN.ADAM(0.01), ODINN.LBFGS()]
    end

    # TODO: Currently there are two different steps defined in params.simulationa and params.solver which need to coincide for manual discrete adjoint
    δt = 1/12
    tspan = (2010.0, 2012.0)

    params = Parameters(
        simulation = SimulationParameters(
            working_dir = working_dir,
            use_MB = use_MB,
            use_velocities = false,
            tspan = tspan,
            step_MB = δt,
            multiprocessing = multiprocessing,
            workers = workers,
            test_mode = true,
            rgi_paths = rgi_paths,
            gridScalingFactor = 4 # We reduce the size of glacier for simulation
            ),
        hyper = Hyperparameters(
            batch_size = length(rgi_ids), # We set batch size equals all datasize so we test gradient
            epochs = epochs,
            optimizer = optimizer
            ),
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-17
            ),
        UDE = UDEparameters(
            optim_autoAD = ODINN.NoAD(),
            grad = grad,
            optimization_method  ="AD+AD",
            target = :A
            ),
        solver = Huginn.SolverParameters(
            step = δt,
            progress = true
            )
        )

    MB_model = use_MB ? TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0) : nothing
    model = Model(
        iceflow = SIA2Dmodel(params; A=CuffeyPaterson()),
        mass_balance = MB_model,
    )

    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, params)

    # Time snapshots for transient inversion
    tstops = collect(tspan[1]:δt:tspan[2])

    A_poly = Huginn.polyA_PatersonCuffey()

    glaciers = generate_ground_truth(glaciers, params, model, tstops)

    optimizable_model = functional_inv ? NeuralNetwork(params) : GlacierWideInv(params, glaciers, :A)
    A_law = functional_inv ? LawA(optimizable_model, params) : LawA(params)
    model = Model(
        iceflow = SIA2Dmodel(params; A=A_law),
        mass_balance = MB_model,
        regressors = (; A=optimizable_model))

    # We create an ODINN prediction
    functional_inversion = Inversion(model, glaciers, params)

    # We run the simulation
    path = mktempdir()
    file_name = functional_inv ? "functional_inversion_test.jld2" : "classical_inversion_test.jld2"
    run!(
        functional_inversion;
        path = path,
        file_name = file_name
    )

    res_load = load(joinpath(path, file_name), "res")

    losses = res_load.losses
    θ = res_load.θ

    # Compute estimated values of A

    Temps = Float64[]
    As_optim = Float64[]

    t = tstops[end]
    for (i, glacier) in enumerate(glaciers)
        # Initialize the cache to make predictions with the law
        functional_inversion.cache = init_cache(functional_inversion.model, functional_inversion, i, θ)
        functional_inversion.model.machine_learning.θ = θ

        T = get_input(iTemp(), functional_inversion, i, t)
        apply_law!(functional_inversion.model.iceflow.A, functional_inversion.cache.iceflow.A, functional_inversion, i, t, θ)
        push!(Temps, T)
        push!(As_optim, functional_inversion.cache.iceflow.A.value[1])
    end

    if !multiprocessing || !functional_inv
        # Reference value of A
        As_fake = A_poly.(Temps)
        @show As_fake
        @show As_optim

        # Loss did not decrease enough during inversion training
        @test losses[end] < 1e-6
        # Loss did not decrease enough during inversion training
        @test losses[end] < 1e-6 * losses[begin]

        rel_error = abs.(As_optim .- As_fake) ./ As_fake

        # Worse case inversion error is not small enough
        @test maximum(rel_error) < 1e-3
        # Inversion not working even for the best behaved glacier
        @test minimum(rel_error) < 1e-4
    end
end
