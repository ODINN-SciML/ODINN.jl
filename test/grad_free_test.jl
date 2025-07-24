function grad_free_test(;use_MB::Bool=false)

    println("> Testing dummy gradient.")

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()

    working_dir = joinpath(ODINN.root_dir, "test/data")

    δt = 1/12
    tspan = (2010.0, 2012.0)

    params = Parameters(
        simulation = SimulationParameters(
            working_dir=working_dir,
            use_MB=use_MB,
            velocities=true,
            tspan=tspan,
            step=δt,
            multiprocessing=false,
            workers=1,
            test_mode=true,
            rgi_paths=rgi_paths),
        hyper = Hyperparameters(
            batch_size=length(rgi_ids), # We set batch size equals all datasize so we test gradient
            epochs=10,
            optimizer=ODINN.ADAM(0.005)),
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=DummyAdjoint(),
            optimization_method="AD+AD",
            target = :A),
        solver = Huginn.SolverParameters(
            step=δt,
            save_everystep=true,
            progress=true)
    )

    nn_model = NeuralNetwork(params)
    JET.@test_opt broken=true target_modules=(Sleipnir, Muninn, Huginn, ODINN) NeuralNetwork(params)

    # Use a constant A for testing
    A_law = ConstantA(2.21e-18)
    JET.@test_opt target_modules=(Sleipnir, Muninn, Huginn, ODINN) ConstantA(2.21e-18)
    model = Huginn.Model(
        iceflow = SIA2Dmodel(params; A=A_law),
        mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
    )
    JET.@test_opt target_modules=(Sleipnir, Muninn, Huginn, ODINN) Huginn.Model(
        iceflow = SIA2Dmodel(params; A=A_law),
        mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
    )

    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, params)

    # Time stanpshots for transient inversion
    tstops = collect(tspan[1]:δt:tspan[2])

    generate_ground_truth!(glaciers, params, model, tstops)
    # Do a clean restart
    A_law = LawA(nn_model, params)
    JET.@test_opt target_modules=(Sleipnir, Muninn, Huginn, ODINN) LawA(nn_model, params)

    model = ODINN.Model(
        iceflow = SIA2Dmodel(params; A=A_law),
        mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
        regressors = (; A=nn_model)
    )
    JET.@test_opt broken=true target_modules=(Sleipnir, Muninn, Huginn, ODINN) ODINN.Model(
        iceflow = SIA2Dmodel(params; A=A_law),
        mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
        regressors = (; A=nn_model)
    )

    # We create an ODINN prediction
    functional_inversion = FunctionalInversion(model, glaciers, params)
    JET.@test_opt target_modules=(Sleipnir, Muninn, Huginn, ODINN) FunctionalInversion(model, glaciers, params)

    # Run simulation
    run!(functional_inversion)

    # Check losses have been stored
    @test length(functional_inversion.stats.losses) > 2
    # Check that losses change over iterations
    @test any(!=(first(functional_inversion.stats.losses)), functional_inversion.stats.losses)
    # Check parameter has changed
    @test any(!=(first(functional_inversion.model.machine_learning.θ)),  functional_inversion.model.machine_learning.θ)
end
