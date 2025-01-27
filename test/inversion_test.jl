function inversion_test(;steady_state = false, save_refs = false)
    rgi_ids = ["RGI60-07.00042", "RGI60-07.00065"]
    rgi_paths = get_rgi_paths()
    working_dir = joinpath(ODINN.root_dir, "test/data")

    params = ODINN.Parameters(
        simulation = SimulationParameters(
            use_MB = true,
            use_iceflow = true,
            velocities = true,
            use_glathida_data = true,
            tspan = (2014.0, 2017.0),
            working_dir = working_dir,
            multiprocessing = true,
            workers = 1,
            rgi_paths = rgi_paths,
            ice_thickness_source = "Farinotti19",
        ),
        solver = SolverParameters(reltol = 1e-8)
    )

    model = ODINN.Model(
        iceflow = SIA2Dmodel(params, C=0.),
        mass_balance = TImodel1(params),
        machine_learning = nothing
    )

    glaciers = initialize_glaciers(rgi_ids, params)
    inversion = Inversion(model, glaciers, params)

    if steady_state
        run₀(inversion)
        ss = inversion.inversion

        if save_refs
            # Save the ss object to a JLD2 file when save_refs is true
            jldsave(joinpath(ODINN.root_dir, "test/data/PDE_refs_MB.jld2"); ss)
        end
        # Load the reference ss object from the JLD2 file for comparison
        ss_ref = load(joinpath(ODINN.root_dir, "test/data/PDE_refs_MB.jld2"), "ss")

        # Perform the comparison test between ss and ss_ref
        @test ss[1].H_pred == ss_ref[1].H_pred

    end

end



