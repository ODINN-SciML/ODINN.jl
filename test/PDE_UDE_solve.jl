function ude_solve_test(; MB=false, fast=true)

    rgi_paths = get_rgi_paths()

    working_dir = joinpath(homedir(), "OGGM/ODINN_tests")

    # params = Parameters(simulation = SimulationParameters(use_MB=MB,
    #                                                     velocities=false,
    #                                                     tspan=(2010.0, 2015.0),
    #                                                     workers=3,
    #                                                     multiprocessing=true,
    #                                                     test_mode=true),
    #                     hyper = Hyperparameters(batch_size=2,
    #                                             epochs=10),
    #                     UDE = UDEparameters()
    #                     )

    params = Parameters(simulation = SimulationParameters(working_dir=working_dir,
                                                        use_MB=MB,
                                                        velocities=true,
                                                        tspan=(2010.0, 2015.0),
                                                        multiprocessing=false,
                                                        workers=5,
                                                        test_mode=true,
                                                        rgi_paths=rgi_paths),
                        hyper = Hyperparameters(batch_size=4,
                                                epochs=4,
                                                optimizer=ODINN.ADAM(0.01)),
                        UDE = UDEparameters(
                            optim_autoAD = Optimization.AutoEnzyme(; mode=set_runtime_activity(EnzymeCore.Reverse)),
                            target = "A"
                        )
                        )

    ## Retrieving simulation data for the following glaciers
    ## Fast version includes less glacier to reduce the amount of downloaded files and computation time on GitHub CI
    if fast
        rgi_ids = ["RGI60-11.03638"]#, "RGI60-11.01450"]#, "RGI60-08.00213", "RGI60-04.04351"]
    else
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170",
        "RGI60-02.05098", "RGI60-01.01104", "RGI60-01.09162", "RGI60-01.00570", "RGI60-04.07051",
        "RGI60-07.00274", "RGI60-07.01323",  "RGI60-01.17316"]
    end

    # # Load reference values for the simulation
    # if MB
    #     PDE_refs = load(joinpath(ODINN.root_dir, "test/data/PDE_refs_MB.jld2"))["PDE_preds"]
    # else
    #     PDE_refs = load(joinpath(ODINN.root_dir, "test/data/PDE_refs_noMB.jld2"))["PDE_preds"]
    # end

    model = Model(iceflow = SIA2Dmodel(params),
                  mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
                  machine_learning = NeuralNetwork(params))

    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, params)

    # We create an ODINN prediction
    functional_inversion = FunctionalInversion(model, glaciers, params)

    # We run the simulation
    @time run!(functional_inversion)

end