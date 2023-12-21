export run_inversion!

"""
    run_inversion!(simulation::Inversion)

In-place run of classic inversion model. 
"""
function run_inversion!(simulation::Inversion)

    enable_multiprocessing(simulation.parameters)

    println("Running forward in-place PDE ice flow inversion model...\n")
    inversion_params_list = @showprogress pmap((glacier_idx) -> invert_iceflow!(glacier_idx, simulation), 1:length(simulation.glaciers))

    simulation.inversion = inversion_params_list

    @everywhere GC.gc() # run garbage collector

end

function invert_iceflow!(glacier_idx::Int, simulation::Inversion) 
    #model = simulation.model
    #params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]
    
    glacier_id = isnothing(glacier.gdir) ? "unnamed" : glacier.rgi_id
    println("Processing glacier: ", glacier_id)
      
    # Trim observed velocity to match size predicted velocity 
    #glacier.V=glacier.V[1:end-1, 1:end-1]
    
    function objective_function(x,p)
        glacier_idx = p[1]
        model = p[2].model 
        glacier = p[2].glaciers[glacier_idx] 
        params = p[2].parameters
        
        model.iceflow.A=x[1]
        model.iceflow.n=x[2]

        # TODO: fix appropiate H and V
        h_dim1, h_dim2 = size(glacier.H_glathida)
        v_dim1, v_dim2 = size(glacier.V)

        H = [1 * x[1] for i in 1:h_dim1, j in 1:h_dim2]
        V = [1 * x[2] for i in 1:v_dim1, j in 1:v_dim2]
       
        # Extract the non-zero elements from observed data
        non_zero_indices_H = glacier.H_glathida .!= 0
        non_zero_indices_V = glacier.V .!= 0

        # Calculate the MSE for H using only the non-zero elements
        mse_H = mean(((glacier.H_glathida[non_zero_indices_H] .- H[non_zero_indices_H]) .^ 2))
        
        # Calculate the MSE for V
        mse_V = mean(((glacier.V[non_zero_indices_V] .- V[non_zero_indices_V]) .^ 2))
        
        # Calculate weighted average of MSE's
        count_H = sum(non_zero_indices_H)
        count_V = sum(non_zero_indices_V)
    
        weighted_average_mse = (count_H * mse_H + count_V * mse_V) / (count_H + count_V)
        
        return weighted_average_mse
        
    end
    
    initial_values = [4.0e-17,3.0]  
    lower_bound = [8.5e-20, 2.5]    # Adjust as needed
    upper_bound = [8.0e-17, 4.2]    # Adjust as needed
    p = (glacier_idx,simulation)

    # TODO: fix right optimizing method
    optf = OptimizationFunction(objective_function,  Optimization.AutoZygote())
    prob = OptimizationProblem(optf, initial_values, p,  lb = lower_bound,ub = upper_bound)

    # Import a solver package and solve the optimization problem
    sol = Optimization.solve(prob, Optim.BFGS())  

    optimized_A = sol[1]
   
    # TODO: change these to optimized values
    optimized_n = sol[2]
    optimized_C = 0.0

    inversion_params = InversionParams(optimized_A, optimized_n, optimized_C)
    return inversion_params
end




