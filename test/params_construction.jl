
function params_constructor_specified(save_refs::Bool = false)

    rgi_paths = Sleipnir.get_rgi_paths()

    solver_params = SolverParameters(solver = Ralston(),
                                    reltol = 1e-8,
                                    step= 1.0/12.0,
                                    save_everystep = false,
                                    tstops = nothing,
                                    progress = true,
                                    progress_steps = 10)

    hyparams = Hyperparameters(current_epoch=1,
                              current_minibatch=1,
                              loss_history=Vector{Float64}(),
                              optimizer=BFGS(),
                              epochs=10,
                              batch_size=15)

    physical_params = PhysicalParameters(ρ = 900.0,
                                        g = 9.81,
                                        ϵ = 1e-3,
                                        η₀ = 1.0,
                                        maxA = 8e-17,
                                        minA = 8.5e-20,
                                        maxTlaw = 1.0,
                                        minTlaw = -25.0,
                                        noise_A_magnitude = 5e-18)

    simulation_params = SimulationParameters(use_MB = true,
                                            use_iceflow = true,
                                            plots = false,
                                            velocities = false,
                                            overwrite_climate = false,
                                            tspan = (2010.0,2015.0),
                                            multiprocessing = false,
                                            workers = 10,
                                            rgi_paths = rgi_paths)

    ude_params = UDEparameters(sensealg = InterpolatingAdjoint(autojacvec=EnzymeVJP()),
                                optimization_method = "AD+AD",
                                loss_type = "V",
                                scale_loss = true)

    params = Parameters(physical=physical_params,
                        hyper=hyparams,
                        solver=solver_params,
                        UDE=ude_params,
                        simulation=simulation_params)

    if save_refs
        jldsave(joinpath(Sleipnir.root_dir, "test/data/params/solver_params.jld2"); solver_params)
        jldsave(joinpath(Sleipnir.root_dir, "test/data/params/hyparams.jld2"); hyparams)
        jldsave(joinpath(Sleipnir.root_dir, "test/data/params/physical_params.jld2"); physical_params)
        jldsave(joinpath(Sleipnir.root_dir, "test/data/params/simulation_params.jld2"); simulation_params)
        jldsave(joinpath(Sleipnir.root_dir, "test/data/params/ude_params.jld2"); ude_params)
        jldsave(joinpath(Sleipnir.root_dir, "test/data/params/params.jld2"); params)
    end


end

function params_constructor_default()


end