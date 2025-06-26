function save_simulation_test!()

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()
    working_dir = joinpath(ODINN.root_dir, "test/data")
    δt = 1/12

    params = Parameters(
        simulation = SimulationParameters(
            use_MB = false,
            velocities = true,
            tspan = (2010.0, 2015.0),
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

    model = Model(
        iceflow = SIA2Dmodel(params, C=0.0),
        mass_balance = TImodel1(params),
        machine_learning = NeuralNetwork(params)
    )

    glaciers = initialize_glaciers(rgi_ids, params)

    tstops = collect(2010:δt:2015)
    ODINN.generate_ground_truth(glaciers, :PatersonCuffey, params, model, tstops)

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


