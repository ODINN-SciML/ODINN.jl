function inversion_test(;steady_state = false)
    rgi_ids = ["RGI60-11.01450", "RGI60-11.00638"]
    working_dir = joinpath(ODINN.root_dir, "test/data")

    params = Parameters(
        OGGM = OGGMparameters(
            working_dir = working_dir,
            multiprocessing = true,
            workers = 1,
            ice_thickness_source = "Farinotti19",
            DEM_source = "DEM3"
        ),
        simulation = SimulationParameters(
            use_MB = true,
            use_iceflow = true,
            velocities = true,
            use_glathida_data = true,
            tspan = (2014.0, 2017.0),
            working_dir = working_dir,
            multiprocessing = true,
            workers = 1
        ),
        solver = SolverParameters(reltol = 1e-8)
    )

    model = Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = TImodel1(params),
        machine_learning = nothing
    )

    glaciers = initialize_glaciers(rgi_ids, params)
    inversion = Inversion(model, glaciers, params)

    if steady_state
        run_ss(inversion)
        ss = inversion.inversion
        @test ss !== nothing 
    end
end


