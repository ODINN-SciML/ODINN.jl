
export run!

function run!(simulation::Prediction)

    println("Running forward PDE ice flow model...\n")
    results_list = @showprogress pmap((glacier_idx) -> batch_iceflow_PDE(glacier_idx, simulation), 1:length(simulation.glaciers))

    save_results_file!(results_list, simulation)

    @everywhere GC.gc() # run garbage collector

end

"""
    batch_iceflow_PDE(glacier_idx::Int, simulation::Prediction) 

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series batch
"""
function batch_iceflow_PDE(glacier_idx::Int, simulation::Prediction) 
    
    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]
    println("Processing glacier: ", glacier.gdir.rgi_id)

    # Initialize glacier ice flow model
    initialize_iceflow_model!(model.iceflow, glacier_idx, glacier, params)
    params.solver.tstops = define_callback_steps(params.simulation.tspan, params.solver.step)
    stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, params.solver.tstops) #closure
    function action!(integrator)
        if params.simulation.use_MB 
            # Compute mass balance
            @timeit get_timer("ODINN") "Mass balance" begin
            MB_timestep!(model, glacier, params.solver, integrator.t)
            apply_MB_mask!(integrator.u, glacier, model.iceflow)
            end
        end
        # # Recompute A value
        # A = context[1]
        # A_noise = context[23]
        # A[] = A_fake(mean(climate.longterm_temps), A_noise, noise)[1]
    end
    cb_MB = DiscreteCallback(stop_condition, action!)

    # Run iceflow PDE for this glacier
    du = params.simulation.use_iceflow ? SIA2D! : noSIA2D!
    results = @timeit get_timer("ODINN") "Simulate iceflow PDE" simulate_iceflow_PDE!(simulation, model, params, cb_MB; du = du)

    return results
end

"""
    simulate_iceflow_PDE!(simulation::Simulation, model::Model, params::Parameters, cb::DiscreteCallback; du = SIA2D!)

Make forward simulation of the iceflow PDE determined in `du`.
"""
function simulate_iceflow_PDE!(simulation::SIM, model::Model, params::Parameters, cb::DiscreteCallback; du = SIA2D!) where {SIM <: Simulation}
    # Define problem to be solved
    iceflow_prob = ODEProblem{true,SciMLBase.FullSpecialize}(du, model.iceflow.H, params.simulation.tspan, tstops=params.solver.tstops, simulation)
    @timeit get_timer("ODINN") "Solve" begin
    iceflow_sol = solve(iceflow_prob, 
                        params.solver.solver, 
                        callback=cb, 
                        tstops=params.solver.tstops, 
                        reltol=params.solver.reltol, 
                        save_everystep=params.solver.save_everystep, 
                        progress=params.solver.progress, 
                        progress_steps=params.solver.progress_steps)
    end
    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    @timeit get_timer("ODINN") "Postprocessing" begin
    model.iceflow.H .= iceflow_sol.u[end]
    map!(x -> ifelse(x>0.0,x,0.0), model.iceflow.H, model.iceflow.H)
    
    @timeit get_timer("ODINN") "Surface v"  avg_surface_V!(iceflow_sol[begin], simulation) # Average velocity with average temperature
    glacier_idx = simulation.model.iceflow.glacier_idx
    glacier::Glacier = simulation.glaciers[glacier_idx[]]
    model.iceflow.S .= glacier.B .+ model.iceflow.H # Surface topography
    end

    # Update simulation results
    @timeit get_timer("ODINN") "Results" begin
    results = create_results(simulation, glacier_idx[], iceflow_sol; light=true)
    end

    return results
end

