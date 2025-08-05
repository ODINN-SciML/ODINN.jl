function save_simulation_test!()

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()
    working_dir = joinpath(ODINN.root_dir, "test/data")
    δt = 1/12
    tspan = (2010.0, 2012.0)

    params = Parameters(
        simulation = SimulationParameters(
            use_MB = false,
            use_velocities = true,
            tspan = tspan,
            step = δt,
            working_dir = working_dir,
            multiprocessing = false,
            workers = 1,
            test_mode = false,
            rgi_paths = rgi_paths,
        ),
        hyper = Hyperparameters(
            batch_size = length(rgi_ids), # We set batch size equals all datasize so we test gradient
            epochs = 1,
            optimizer = ODINN.ADAM(0.01),
            ),
        UDE = UDEparameters(
            optim_autoAD = ODINN.NoAD(),
            grad = ContinuousAdjoint(),
            target = :A
        ),
        solver = Huginn.SolverParameters(
            step = δt,
            save_everystep = true,
            progress = true
            )
    )

    model = Huginn.Model(
        iceflow = SIA2Dmodel(params, A=CuffeyPaterson()),
        mass_balance = TImodel1(params),
    )

    glaciers = initialize_glaciers(rgi_ids, params)

    tstops = collect(tspan[1]:δt:tspan[2])
    glaciers = generate_ground_truth(glaciers, params, model, tstops)

    nn_model = NeuralNetwork(params)
    model = Model(
        iceflow = SIA2Dmodel(params, A=LawA(nn_model, params)),
        mass_balance = TImodel1(params),
        regressors = (; A=nn_model)
    )

    functional_inversion = FunctionalInversion(model, glaciers, params)

    path = mktempdir()
    file_name = "simulation_results.jld2"
    run!(functional_inversion; path = path, file_name = file_name)

    # Load results
    res_load = load(joinpath(ODINN.root_dir, path, file_name), "res")

    # Obtain parameters of model
    θ_load = res_load.θ
    losses_load = res_load.losses
    params_load = res_load.params

    @test !isnothing(res_load)
    @test θ_load ≈ functional_inversion.stats.θ
    @test length(losses_load) == functional_inversion.stats.niter
    @test params_load.hyper.epochs == params.hyper.epochs
end


