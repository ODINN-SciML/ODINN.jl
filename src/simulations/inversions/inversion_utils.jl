export run₀

"""
    run_inversion!(simulation::Inversion)

Out-place run of classic inversion model. 
"""
function run₀(simulation::Inversion)

    enable_multiprocessing(simulation.parameters)

    println("Running out-place ice flow inversion model...\n")

    inversion_params_list = @showprogress pmap((glacier_idx) -> invert_iceflow_transient(glacier_idx, simulation), 1:length(simulation.glaciers))

    simulation.inversion = inversion_params_list

    @everywhere GC.gc() # run garbage collector

end

function invert_iceflow_transient(glacier_idx::Int, simulation::Inversion) 
    
    ####### Retrieve parameters #######
    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]
    
    glacier_id = isnothing(glacier.gdir) ? "unnamed" : glacier.rgi_id
    println("Processing glacier: ", glacier_id)
    
    ####### Glacier Initialization #######
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

    ####### Objective function #######
    function objfun(x, simulation, realsol)
        
        simulation.model.iceflow.A[]=x[1]*1e-17  #scaling for optimization
        #simulation.model.iceflow.n[]=x[2]

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
            mask_V = V_obs .!= 0
            
            ofv_H = mean((H_pred[mask_H] .- H_obs[mask_H]) .^ 2)
            ofv_V = mean((V_obs[mask_V] .- V_pred[mask_V]) .^ 2)

            count_H = 50 #sum(mask_H)
            count_V = 50 #sum(mask_V)
    
            ofv = (count_H * ofv_H + count_V * ofv_V) / (count_H + count_V)
            
            #ofv = x[3] * ofv_H + x[4] * ofv_V
            
        end
        
        println("A = $x")
        println("MSE = $ofv")
        
        return ofv
    end

    ####### Optimization #######
    fn(x, p) = objfun(x, p[1], p[2])
    #cons(res, x, p) = (res .= x[2] + x[3] - 1.0)
    
    realsol = (glacier.H_glathida, glacier.V[1:end-1, 1:end-1])
    
    # Bounds for parameters
    initial_conditions = [4.0] #3.0,0.5,0.5] #A, n, weight_H, weight_V  
    lower_bound = [0.0085] #2.7,0.0, 0.0]   
    upper_bound = [800.0] #3.3, 1.0, 1.0] 
    
    optfun = OptimizationFunction(fn, Optimization.AutoForwardDiff(),) #cons=cons)
    optprob = OptimizationProblem(optfun, initial_conditions, (simulation, realsol), lb = lower_bound,ub = upper_bound,) #lcons = [0.0], ucons = [0.0])
    sol = solve(optprob, BFGS(), x_tol=1.0e-1, f_tol = 1e1)

    # TODO: Add optimization of n and C
    optimized_n = 3.0
    optimized_C = 0.0
    
    ####### Setup returning solution #######
    optimized_A = sol[1] * 1e-17  
    simulation.model.iceflow.A[] = optimized_A 
    
    # Determine H and V with optimized values
    iceflow_prob = ODEProblem{false}(Huginn.SIA2D, simulation.model.iceflow.H, simulation.parameters.simulation.tspan, simulation)
    iceflow_sol = solve(iceflow_prob, 
                        params.solver.solver, 
                        callback=cb_MB, 
                        tstops=params.solver.tstops, 
                        reltol=params.solver.reltol, 
                        abstol=params.solver.reltol,
                        save_everystep=params.solver.save_everystep, 
                        progress=params.solver.progress, 
                        progress_steps=params.solver.progress_steps)

    H_pred = iceflow_sol.u[end]
    map!(x -> ifelse(x>0.0,x,0.0), H_pred, H_pred)
    V_pred = Huginn.avg_surface_V(H_pred, simulation)[3]  

    # Extract observed H and V from realsol
    H_obs = realsol[1]
    V_obs = realsol[2]

    mask_H = H_obs .!= 0
    mask_V = V_obs .!= 0

    # Absolute difference between observed and predicted
    H_diff = abs.(H_pred .- H_obs)  
    V_diff = abs.(V_pred .- V_obs)  

    H_diff[H_obs .== 0] .= 0 # Set differences to zero where observations are zero
    V_diff[V_obs .== 0] .= 0

    # Compute differences and MSE  #TODO: change into extracted MSE from solution
    MSE_H = mean((H_pred[mask_H] .- H_obs[mask_H]) .^ 2)  
    MSE_V = mean((V_pred[mask_V] .- V_obs[mask_V]) .^ 2)  
    MSE = (MSE_H  + MSE_V) / 2  # Simple average of H and V MSEs for overall MSE

    # Create InversionMetrics instance with all the necessary fields
    inversion_metrics = InversionMetrics(optimized_A, optimized_n, optimized_C, 
                                        H_pred, H_obs, H_diff, V_pred, V_obs, V_diff, 
                                        MSE, 
                                        simulation.glaciers[glacier_idx].Δx,              
                                        simulation.glaciers[glacier_idx].Δy,
                                        Sleipnir.safe_getproperty(simulation.glaciers[glacier_idx].gdir, :cenlon),
                                        Sleipnir.safe_getproperty(simulation.glaciers[glacier_idx].gdir, :cenlat))

return inversion_metrics
end
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

