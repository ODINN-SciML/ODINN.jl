export run₀, run_ss

"""
    run_inversion!(simulation::Inversion)

Out-place run of classic inversion model. 
"""
function run₀(simulation::Inversion)

    enable_multiprocessing(simulation.parameters)

    println("Running out-place transient ice flow inversion model...\n")

    inversion_params_list = @showprogress pmap((glacier_idx) -> invert_iceflow_transient(glacier_idx, simulation), 1:length(simulation.glaciers))

    simulation.inversion = inversion_params_list

    @everywhere GC.gc() # run garbage collector

end

function run_ss(simulation::Inversion)

    enable_multiprocessing(simulation.parameters)

    println("Running out-place steady state ice flow inversion model...\n")

    inversion_params_list = @showprogress pmap((glacier_idx) -> invert_iceflow_ss(glacier_idx, simulation), 1:length(simulation.glaciers))

    simulation.inversion = inversion_params_list

    @everywhere GC.gc() # run garbage collector

end

# === [Begin] Transient Inversion ===  # Frame work is here, however function is not physically correct
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
    inversion_metrics = InversionMetrics(glacier.rgi_id,optimized_A, optimized_n, optimized_C, 
                                        H_pred, H_obs, H_diff, V_pred, V_obs, V_diff, 
                                        MSE_V, #change
                                        simulation.glaciers[glacier_idx].Δx,              
                                        simulation.glaciers[glacier_idx].Δy,
                                        Sleipnir.safe_getproperty(simulation.glaciers[glacier_idx].gdir, :cenlon),
                                        Sleipnir.safe_getproperty(simulation.glaciers[glacier_idx].gdir, :cenlat))

return inversion_metrics
end
# === [End] Transient Inversion ===

# === [Begin] Steady State Inversion ===
function invert_iceflow_ss(glacier_idx::Int, simulation::Inversion)
    # === Retrieve Parameters ===
    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]
    glacier_id = isnothing(glacier.gdir) ? "unnamed" : glacier.rgi_id
    println("Processing glacier: ", glacier_id)

    # === Glacier Initialization ===
    Huginn.initialize_iceflow_model(model.iceflow, glacier_idx, glacier, params)
    simulation.model.iceflow.A[] = 5.0e-17
    
    # === Objective Function Definition ===
    function objfun(x, simulation, realsol)
        # Apply parameter transformations for optimization
         simulation.model.iceflow.C[] = x[1]

        # Extract and predict H values based on V observations
        H_obs = realsol[1][1:end-1, 1:end-1]
        V_obs = realsol[2]
        H_pred = Huginn.H_from_V(V_obs, simulation)

        # Calculate and return the weighted mean squared error
        mask_H = H_obs .!= 0
        ofv_H = weighted_mse(H_obs[mask_H], H_pred[mask_H], 1.0)
        return ofv_H
    end

    # === Optimization Setup ===
    initial_conditions = [1.0]
    lower_bound = [0.0]
    upper_bound = [Inf]
    optfun = OptimizationFunction((x, p) -> objfun(x, p[1], p[2]), Optimization.AutoForwardDiff())

    # === Region-based Optimization and Prediction ===
    regions = split_regions(glacier.H_glathida, glacier.dist_border, 6, 6)
    total_H_pred = zeros(size(glacier.H_glathida[1:end-1, 1:end-1]))
    C_values = fill(NaN, size(glacier.H_glathida))

    
    for region_H_obs in regions
        realsol = (region_H_obs, glacier.V)
        optprob = OptimizationProblem(optfun, initial_conditions, (simulation, realsol), lb = lower_bound, ub = upper_bound)
        sol = solve(optprob, BFGS(), x_tol = 1.0e-1, f_tol = 1.0e-1)
        
        # Apply optimized parameters and predict H values
        simulation.model.iceflow.C[] = sol[1]
        
        H_pred = Huginn.H_from_V(realsol[2], simulation)
        H_pred[realsol[1][1:end-1, 1:end-1] .== 0] .= 0
        total_H_pred += H_pred

        # Create parameter matrices
        C_values[realsol[1] .!= 0].= sol[1] 
    end

    # === Post-Optimization Analysis ===
    # Prepare observed values for comparison
    H_obs, V_obs = glacier.H_glathida, glacier.V
    Vx, Vy = Huginn.surface_V(H_obs, simulation)
    V_pred = sqrt.(Vx.^2 + Vy.^2)
    H_obs = H_obs[1:end-1, 1:end-1]
    V_obs = V_obs[1:end-1, 1:end-1]

    # Compute absolute differences
    H_diff = abs.(total_H_pred - H_obs)
    V_diff = abs.(V_pred - V_obs)
    H_diff[H_obs .== 0] .= 0
    V_diff[V_obs .== 0] .= 0

    # === Results Compilation ===
    # Initialize mean squared error (MSE) for potential calculation or adjustment
    MSE = 0.0

    # Compile inversion metrics
    inversion_metrics = InversionMetrics(
        glacier.rgi_id,
        simulation.model.iceflow.A[],
        simulation.model.iceflow.n[],
        C_values,
        total_H_pred, H_obs, H_diff,
        V_pred, V_obs, V_diff,
        MSE,
        simulation.glaciers[glacier_idx].Δx,
        simulation.glaciers[glacier_idx].Δy,
        Sleipnir.safe_getproperty(simulation.glaciers[glacier_idx].gdir, :cenlon),
        Sleipnir.safe_getproperty(simulation.glaciers[glacier_idx].gdir, :cenlat)
    )

    return inversion_metrics
end

function weighted_mse(H_obs, H_pred, weight_underprediction=2.0)
    errors = H_pred .- H_obs
    weighted_errors = map(errors) do error
        if error < 0
            # Underprediction
            return weight_underprediction * error^2
        else
            # Overprediction or accurate prediction
            return error^2
        end
    end
    return mean(weighted_errors)
end

function split_regions(H_obs, dist_border, n_splits_H_obs, n_splits_dist_border)
    max_value_H_obs = maximum(H_obs)
    regions = []

    for i in 1:n_splits_H_obs
        # Initial split based on H_obs
        region_mask_H_obs = if i == n_splits_H_obs
            (H_obs .>= (i-1)/n_splits_H_obs * max_value_H_obs)
        else
            (H_obs .>= (i-1)/n_splits_H_obs * max_value_H_obs) .& (H_obs .< i/n_splits_H_obs * max_value_H_obs)
        end

        # Find the maximum of dist_border within the current H_obs region
        max_value_dist_border = maximum(dist_border[region_mask_H_obs])

        for j in 1:n_splits_dist_border
            # Further split each H_obs region based on dist_border
            region_mask_dist_border = if j == n_splits_dist_border
                (dist_border .>= (j-1)/n_splits_dist_border * max_value_dist_border) .& region_mask_H_obs
            else
                (dist_border .>= (j-1)/n_splits_dist_border * max_value_dist_border) .& (dist_border .< j/n_splits_dist_border * max_value_dist_border) .& region_mask_H_obs
            end

            # Create a copy of H_obs where only the current sub-region's values are kept
            region_H_obs = zeros(size(H_obs))  # Initialize with zeros
            region_H_obs[region_mask_dist_border] = H_obs[region_mask_dist_border]

            push!(regions, region_H_obs)
        end
    end
    return regions
end
# === [End] Steady State Inversion ===

