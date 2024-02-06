export run₀

"""
    run_inversion!(simulation::Inversion)

Out-place run of classic inversion model. 
"""
function run₀(simulation::Inversion)

    enable_multiprocessing(simulation.parameters)

    println("Running forward out-place PDE ice flow inversion model...\n")
    inversion_params_list = @showprogress pmap((glacier_idx) -> invert_iceflow(glacier_idx, simulation), 1:length(simulation.glaciers))

    simulation.inversion = inversion_params_list

    @everywhere GC.gc() # run garbage collector

end

function invert_iceflow(glacier_idx::Int, simulation::Inversion) 
    
    # Retrieve parameters
    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]
    
    glacier_id = isnothing(glacier.gdir) ? "unnamed" : glacier.rgi_id
    println("Processing glacier: ", glacier_id)
    
    # Initialization
    Huginn.initialize_iceflow_model(model.iceflow, glacier_idx, glacier, params)

    params.solver.tstops = Huginn.define_callback_steps(params.simulation.tspan, params.solver.step)
    stop_condition(u,t,integrator) = Sleipnir.stop_condition_tstops(u,t,integrator, params.solver.tstops) #closure
    
    function action!(integrator)
        if params.simulation.use_MB 
            # Compute mass balance
            MB_timestep!(model, glacier, params.solver.step, integrator.t)
            apply_MB_mask!(integrator.u, glacier, model.iceflow)
        end
    end

    cb_MB = DiscreteCallback(stop_condition, action!)

    # Objective function
    function objfun(x, simulation, realsol)
        simulation.model.iceflow.A[]=x[1]*1e-17  #scaling for optimization
        
        iceflow_prob = ODEProblem{false}(Huginn.SIA2D, model.iceflow.H, params.simulation.tspan, tstops=params.solver.tstops, simulation)
        iceflow_sol = solve(iceflow_prob, 
                        params.solver.solver, 
                        callback=cb_MB, 
                        tstops=params.solver.tstops, 
                        reltol=params.solver.reltol, 
                        abstol=params.solver.reltol,
                        save_everystep=params.solver.save_everystep, 
                        progress=params.solver.progress, 
                        progress_steps=params.solver.progress_steps)

        
        sol = iceflow_sol.u[end]        
        map!(x -> ifelse(x>0.0,x,0.0), sol, sol)
        
        ofv = 0.0
        if any((s.retcode != :Success for s in iceflow_sol))
            ofv = 1e12
        else
            ofv = mean((sol[realsol .!= 0] .- realsol[realsol .!= 0]) .^ 2)
            
        end
        
        println("A = $x")
        println("MSE = $ofv")
        
        return ofv
    end

    # Optimization
    fn(x, p) = objfun(x, p[1], p[2])

    realsol = glacier.H_glathida
    
    lower_bound = [0.0085]    
    upper_bound = [8.0] 
    
    optfun = OptimizationFunction(fn, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfun, [4.0], (simulation, realsol), lb = lower_bound,ub = upper_bound)
    sol = solve(optprob, BFGS(), x_tol=1.0e-3, f_tol = 1e-3)

    optimized_n = 3.0
    optimized_C = 0.0

    inversion_params = InversionParams(sol[1], optimized_n, optimized_C)
    
    return inversion_params
end

