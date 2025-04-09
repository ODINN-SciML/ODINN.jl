function grad_free_test(;use_MB::Bool=false)

    # The choice of sensitivity algorithm should not matter
    sensealg = SciMLSensitivity.ZygoteAdjoint()
    adtype = ODINN.NoAD()

    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]
    rgi_paths = get_rgi_paths()
    working_dir = joinpath(homedir(), "OGGM/ODINN_tests")

    # Define dummy grad
    # This is just a random gradient, and it just serves to test that the optimization pilelien
    # works independenly of the gradient calculation.
    dummy_grad = function (du, u; simulation::Union{FunctionalInversion, Nothing}=nothing)
        du .= maximum(abs.(u)) .* rand(Float64, size(u))
    end

    params = Parameters(simulation = SimulationParameters(working_dir=working_dir,
                                                        use_MB=use_MB,
                                                        velocities=true,
                                                        tspan=(2010.0, 2015.0),
                                                        multiprocessing=false,
                                                        workers=1,
                                                        test_mode=true,
                                                        rgi_paths=rgi_paths),
                        hyper = Hyperparameters(batch_size=4,
                                                epochs=10,
                                                optimizer=ODINN.ADAM(0.01)),
                        UDE = UDEparameters(sensealg=sensealg,
                                            optim_autoAD=adtype,
                                            grad=DummyAdjoint(grad=dummy_grad),
                                            optimization_method="AD+AD",
                                            target = "A")
                        )

    model = Model(iceflow = SIA2Dmodel(params),
    mass_balance = mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
    machine_learning = NeuralNetwork(params))

    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, params)

    # We create an ODINN prediction
    functional_inversion = FunctionalInversion(model, glaciers, params)

    run!(functional_inversion)

    # Check losses have been stored
    @test length(functional_inversion.stats.losses) > 2
    # Check that losses change over iterations
    @test any(!=(first(functional_inversion.stats.losses)), functional_inversion.stats.losses)
    # Check parameter has change
    # @test any(!=(first(simulation.model.machine_learning.θ)),  simulation.model.machine_learning.θ)

end