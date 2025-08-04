function inversion_test(;
    use_MB = false,
    multiprocessing = false
    )

    rgi_paths = get_rgi_paths()
    # The value of this does not really matter, it is hardcoded in Sleipnir right now.
    working_dir = joinpath(homedir(), ".OGGM/ODINN_tests")

    ## Retrieving simulation data for the following glaciers
    if multiprocessing
        workers = 3 # Two processes for the two glaciers + one for main
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]
        # Multiprocessing is especially slow in the CI, so we perform a very short optimization
        epochs = 3
        optimizer = ODINN.ADAM(0.01)
    else
        workers = 1
        rgi_ids = ["RGI60-11.03638"]
        epochs = [20,20]
        optimizer = [ODINN.ADAM(0.005), ODINN.LBFGS()]
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
            step = δt,
            multiprocessing = multiprocessing,
            workers = workers,
            test_mode = false,
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
            grad = ContinuousAdjoint(),
            optimization_method  ="AD+AD",
            target = :A
            ),
        solver = Huginn.SolverParameters(
            step = δt,
            save_everystep = true,
            progress = true
            )
        )

    if use_MB
        MB_model = TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0)
    else
        MB_model = nothing
    end

    model = Huginn.Model(
        iceflow = SIA2Dmodel(params; A=CuffeyPaterson()),
        mass_balance = MB_model,
    )

    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, params)

    # Time snapshots for transient inversion
    tstops = collect(tspan[1]:δt:tspan[2])

    A_poly = Huginn.polyA_PatersonCuffey()

    generate_ground_truth!(glaciers, params, model, tstops)

    nn_model = NeuralNetwork(params)
    A_law = LawA(nn_model, params)
    model = Model(
        iceflow = SIA2Dmodel(params; A=A_law),
        mass_balance = MB_model,
        regressors = (; A=nn_model))

    # We create an ODINN prediction
    functional_inversion = FunctionalInversion(model, glaciers, params)

    # We run the simulation
    path = mktempdir()
    run!(
        functional_inversion;
        path = path,
        file_name = "inversion_test.jld2"
    )

    res_load = load(joinpath(path, "inversion_test.jld2"), "res")

    losses = res_load.losses
    θ = res_load.θ

    # Compute estimated values of A

    Temps = Float64[]
    As_pred = Float64[]

    t = tstops[end]
    for (i, glacier) in enumerate(glaciers)
        # Initialize the cache to make predictions with the law
        functional_inversion.cache = init_cache(functional_inversion.model, functional_inversion, i, params)
        functional_inversion.model.machine_learning.θ = θ

        T = get_input(InpTemp(), functional_inversion, i, t)
        apply_law!(functional_inversion.model.iceflow.A, functional_inversion.cache.iceflow.A, functional_inversion, i, t, θ)
        push!(Temps, T)
        push!(As_pred, functional_inversion.cache.iceflow.A[1])
    end

    if !multiprocessing
        # Reference value of A
        As_fake = A_poly.(Temps)
        @show As_fake
        @show As_pred

        # Loss did not decrease enough during inversion training
        @test losses[end] < 1e-6
        # Loss did not decrease enough during inversion training
        @test losses[end] < 1e-6 * losses[begin]

        rel_error = abs.(As_pred .- As_fake) ./ As_fake

        # Worse case inversion error is not small enough
        @test maximum(rel_error) < 1e-3
        # Inversion not working even for the best behaved glacier
        @test minimum(rel_error) < 1e-4
    end
end
