
export run!

function run!(simulation::Prediction)

    println("Running forward PDE ice flow model...\n")
    @showprogress pmap((glacier_idx) -> batch_iceflow_PDE(glacier_idx, simulation), 1:length(simulation.glaciers))

    save_results_file(simulation)

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
            MB_timestep!(model, glacier, params.solver, integrator.t)
            apply_MB_mask!(integrator.u, glacier, model.iceflow)
        end
        # # Recompute A value
        # A = context[1]
        # A_noise = context[23]
        # A[] = A_fake(mean(climate.longterm_temps), A_noise, noise)[1]
    end
    cb_MB = DiscreteCallback(stop_condition, action!)

    # Run iceflow PDE for this glacier
    simulate_iceflow_PDE!(simulation, model, params, cb_MB; du = SIA2D!)
end

"""
    simulate_iceflow_PDE!(simulation::Simulation, model::Model, params::Parameters, cb::DiscreteCallback; du = SIA2D!)

Make forward simulation of the iceflow PDE determined in `du`.
"""
function simulate_iceflow_PDE!(simulation::SIM, model::Model, params::Parameters, cb::DiscreteCallback; du = SIA2D!) where {SIM <: Simulation}
    # Define problem to be solved
    iceflow_prob = ODEProblem{true,SciMLBase.FullSpecialize}(du, model.iceflow.H, params.simulation.tspan, tstops=params.solver.tstops, simulation)
    iceflow_sol = solve(iceflow_prob, 
                        params.solver.solver, 
                        callback=cb, 
                        tstops=params.solver.tstops, 
                        reltol=params.solver.reltol, 
                        save_everystep=params.solver.save_everystep, 
                        progress=params.solver.progress, 
                        progress_steps=params.solver.progress_steps)
    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    model.iceflow.H .= iceflow_sol.u[end]
    model.iceflow.H[model.iceflow.H.<0.0] .= 0.0 # remove remaining negative values
    avg_surface_V!(iceflow_sol[begin], simulation) # Average velocity with average temperature
    glacier_idx = simulation.model.iceflow.glacier_idx[]
    glacier::Glacier = simulation.glaciers[glacier_idx]
    model.iceflow.S .= glacier.B .+ model.iceflow.H # Surface topography

    # Update simulation results
    store_results!(simulation, glacier_idx, iceflow_sol)
end


# """
# generate_ref_dataset(gdirs_climate, tspan; solver = RDPK3Sp35(), random_MB=nothing)

# Generate reference dataset based on the iceflow PDE
# """
# function generate_ref_dataset(gdirs, tspan, mb_model; solver = RDPK3Sp35(), velocities=true)
#     # Generate climate data if necessary
#     @timeit to "generate raw climate files" begin
#     pmap((gdir) -> generate_raw_climate_files(gdir, tspan), gdirs)
#     end
#     # Perform reference simulation with forward model 
#     println("Running forward PDE ice flow model...\n")
#     # Run batches in parallel
#     A_noises = randn(rng_seed(), length(gdirs)) .* noise_A_magnitude
#     refs = @showprogress pmap((gdir, A_noise) -> batch_iceflow_PDE(gdir, A_noise, tspan, solver, mb_model; run_spinup=false, velocities=velocities), gdirs, A_noises)

#     # Gather information per gdir
#     gdir_refs = get_gdir_refs(refs, gdirs)

#     @everywhere GC.gc() # run garbage collector 

#     return gdir_refs
# end



