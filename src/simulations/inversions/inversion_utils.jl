export run₀

"""
    run_inversion!(simulation::Inversion)

Out-place run of classic inversion model. 
"""
function run₀(simulation::Inversion)

    enable_multiprocessing(simulation.parameters)

    println("Running out-place PDE ice flow inversion model...\n")
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

        H_obs = realsol[1]
        V_obs = realsol[2]
        
        H_pred = iceflow_sol.u[end]        
        map!(x -> ifelse(x>0.0,x,0.0), H_pred, H_pred)
        
        V_pred = Huginn.avg_surface_V(H_pred, simulation)[3] 

        ofv = 0.0
        if any((s.retcode != :Success for s in iceflow_sol))
            ofv = 1e12
        else
            mask_H = H_obs .!= 0
            mask_V = V_pred .!= 0
            
            ofv_H = mean((H_pred[mask_H] .- H_obs[mask_H]) .^ 2)
            ofv_V = mean((V_obs[mask_V] .- V_pred[mask_V]) .^ 2)

            count_H = sum(mask_H)
            count_V = sum(mask_V)
    
            ofv = (count_H * ofv_H + count_V * ofv_V) / (count_H + count_V)
            
        end
        
        println("A = $x")
        println("MSE = $ofv")
        
        return ofv
    end

    # Optimization
    fn(x, p) = objfun(x, p[1], p[2])

    realsol = (glacier.H_glathida, glacier.V[1:end-1, 1:end-1])
    
    lower_bound = [0.0085]    
    upper_bound = [8.0] 
    
    optfun = OptimizationFunction(fn, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfun, [4.0], (simulation, realsol), lb = lower_bound,ub = upper_bound)
    sol = solve(optprob, BFGS(), x_tol=1.0e-3, f_tol = 1e-3)

    optimized_n = 3.0
    optimized_C = 0.0
    
    count_H = sum(glacier.H_glathida .!= 0)
    count_V = sum(glacier.V[1:end-1, 1:end-1].!= 0)
    
    println("weight H:", count_H/(count_H + count_V))
    println("weight V:", count_V/(count_H + count_V))
    
    inversion_params = InversionParams(sol[1], optimized_n, optimized_C)
    
    return inversion_params
end

