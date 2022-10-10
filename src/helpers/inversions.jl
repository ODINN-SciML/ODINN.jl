export train_iceflow_inversion

"""
invert_iceflow(gdirs, train_settings, θ_trained=[], loss_history=[])

Performs an inversion on the SIA to train a NN on the ice flow law
"""
function train_iceflow_inversion(rgi_ids, glathida, tspan, train_settings, θ_trained=[], loss_history=[])
    println("Training ice rheology inversion...")
    # Initialize gdirs with ice thickness data
    gdirs = init_gdirs(rgi_ids, force=false)
    # Process climate data for glaciers
    gdirs_climate = get_gdirs_with_climate(gdirs, tspan, overwrite=false, massbalance=false, plot=false)

    # Perform inversion with the given gdirs and climate data
    invert_iceflow(glathida, gdirs_climate, tspan, train_settings, θ_trained, loss_history)

end

"""
invert_iceflow(gdirs, train_settings, θ_trained=[], loss_history=[])

Performs an inversion on the SIA to train a NN on the ice flow law
"""
function invert_iceflow(glathida, gdirs_climate, tspan, train_settings, θ_trained, loss_history)
    if length(θ_trained) == 0
        global current_epoch = 1 # reset epoch count
    end
    optimizer = train_settings[1]
    epochs = train_settings[2]
    UD, θ = get_NN_inversion_D(θ_trained)
    gdirs = gdirs_climate[2]

    # Build context for all the batches before training
    println("Building context...")
    context_batches = pmap(gdir -> build_UDE_context(gdir, tspan), gdirs)
    loss(θ) = loss_iceflow_inversion(θ, UD, gdirs_climate, context_batches) # closure
    
    println("Training iceflow rheology inversion...")
    temps = gdirs_climate[3]
    cb_plots(θ, l, UA_f) = callback_plots(θ, l, UA_f, temps, A_noise)
    # Setup optimization of the problem
    optf = OptimizationFunction((θ,_)->loss(θ), Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, θ)
    iceflow_trained = solve(optprob, optimizer, callback = cb_plots, maxiters = epochs)

    return iceflow_trained

end

"""
    loss_iceflow(θ, context, UA, PDE_refs::Dict{String, Any}, temp_series) 

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow_inversion(θ, UD, gdirs_climate, context_batches)
    
    V_pred = perform_V_inversion(θ, UD, gdirs_climate, context_batches)

    # Compute loss function for the full batch
    l_V = 0.0f0
    for i in 1:length(H_V_preds)

        # Get ice velocities from ITS_LIVE or Millan et al. (2022)
        V_ref = context_batches[i][10]

        if scale_loss
            normV = V_ref[V_ref .!= 0.0] .+ ϵ
            l_V += Flux.Losses.mse(V_pred[V_ref .!= 0.0] ./normVx, V_ref[V_ref.!= 0.0] ./normV; agg=mean)
        else
            l_V += Flux.Losses.mse(V_pred[V_ref .!= 0.0], V_ref[V_ref.!= 0.0]; agg=mean)
        end
    end 

    return l_V
end

"""
    perform_iceflow_inversion(θ, UA, gdirs_climate, context_batches)

Performs an inversion of the iceflow law with a UDE in different batches
"""
function perform_V_inversion(θ, UD, gdirs_climate, context_batches)
    T_batches = gdirs_climate[3]
    V = pmap((context, T) -> SIA(T, context, θ, UD), context_batches, T_batches)
    return V
end
