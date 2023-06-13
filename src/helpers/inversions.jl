export train_iceflow_inversion

"""
invert_iceflow(gdirs, train_settings, θ_trained=[], loss_history=[])

Performs an inversion on the SIA to train a NN on the ice flow law
"""
function train_iceflow_inversion(rgi_ids, tspan, train_settings; gdirs_climate=nothing, gdirs_climate_batches=nothing, gdir_refs=nothing, gtd_file=nothing, θ_trained=[], target="D")
    println("Training ice rheology inversion...")
    if isnothing(gdirs_climate_batches) || isnothing(gdirs_climate)
        # Initialize gdirs with ice thickness data
        gdirs = init_gdirs(rgi_ids)
        # Process climate data for glaciers
        gdirs_climate, gdirs_climate_batches = get_gdirs_with_climate(gdirs, tspan, overwrite=false, massbalance=false, plot=false)
    end
    if !isnothing(gtd_file)
        # Produce Glathida dataset
        gtd_grids = get_glathida!(gtd_file, gdirs; force=false)
    else
        gtd_grids=nothing
    end
    # Perform inversion with the given gdirs and climate data
    rheology_trained = invert_iceflow(gdirs_climate, gdirs_climate_batches, gtd_grids, gdir_refs, tspan, train_settings, θ_trained, target)

    return rheology_trained
end

"""
invert_iceflow(gdirs, train_settings, θ_trained=[], loss_history=[])

Performs an inversion on the SIA to train a NN on the ice flow law
"""
function invert_iceflow(gdirs_climate, gdirs_climate_batches, gtd_grids, gdir_refs, tspan, train_settings, θ_trained, target)
    # Configure the current epoch and training history for the run
    config_training_state(θ_trained)
    
    optimizer = train_settings[1]
    epochs = train_settings[2]
    batch_size = train_settings[3]
    UD, θ = get_NN_inversion(θ_trained, target)
    gdirs = gdirs_climate[2]

    # Build context for all the batches before training
    println("Building context...")
    context_batches = try 
         map((gdir, gdir_ref) -> build_UDE_context_inv(gdir, gdir_ref, tspan), gdirs, gdir_refs)
    catch error
        @error "$error: Missing data for some glaciers. The list of missing_glaciers has been updated. Try again."
    end
    
    cb_plots_inv(θ, l, UD_f) = callback_plots_inv(θ, l, UD_f, gdirs_climate, target)

    # Create batches for inversion training 
    train_loader = generate_batches(batch_size, UD, target, gdirs_climate_batches, gdir_refs, context_batches; gtd_grids=gtd_grids)
    
    # Setup optimization of the problem
    optf = OptimizationFunction((θ, _, UD_batch, gdirs_climate_batch, gdir_refs_batch, context_batch, gtd_grids_batch, target_batch)->loss_iceflow_inversion(θ, UD_batch, gdirs_climate_batch, gdir_refs_batch, context_batch, gtd_grids_batch, target_batch), Optimization.AutoZygote())
    
    # optf = OptimizationFunction((θ,_)->loss(θ), Optimization.AutoZygote())
    
    optprob = OptimizationProblem(optf, θ)
    println("Training iceflow rheology inversion...")

    # rheology_trained = solve(optprob, optimizer, maxiters=epochs, allow_f_increases=true, callback=cb_plots_inv, progress=true)

    rheology_trained = solve(optprob, optimizer, ncycle(train_loader, epochs), allow_f_increases=true, callback=cb_plots_inv, progress=true)

    return rheology_trained
end

"""
    loss_iceflow_inversion(θ, UD, gdirs_climate, gdir_refs, context_batches, gtd_grids, target)

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow_inversion(θ, UD, gdirs_climate, gdir_refs, context_batches, gtd_grids, target)
    # (Vx, Vy, V)
    V_preds = perform_V_inversion(θ, UD, gdirs_climate, gtd_grids, context_batches, target)

    # Compute loss function for the full batch
    l_Vx, l_Vy = 0.0f0, 0.0f0
    for i in 1:length(V_preds)
        if isnothing(gtd_grids[1])
            # Get reference dataset
            H_ref = gdir_refs[i]["H"]
            Vx_ref = gdir_refs[i]["Vx"]
            Vy_ref = gdir_refs[i]["Vy"]
            V_ref = sqrt.(Vx_ref.^2 .+ Vy_ref.^2)
        else
            # Get ice velocities from Millan et al. (2022)
            V_ref = avg(context_batches[i][6])
        end
        Vx_pred = V_preds[i][1]
        Vy_pred = V_preds[i][2]

        # H = context_batches[i][7]

        # l_Vx += mean((abs.(Vx_pred[Vx_ref .!= 0.0] .- Vx_ref[Vx_ref.!= 0.0]).^7))^(1/7)
        # l_Vy += mean((abs.(Vy_pred[Vy_ref .!= 0.0] .- Vy_ref[Vy_ref.!= 0.0]).^7))^(1/7)

        # Squared-Mean-Root-Error :D
        # l_Vx += mean((abs.(Vx_pred[Vx_ref .!= 0.0] .- Vx_ref[Vx_ref.!= 0.0]).^(1/4)))^4
        # l_Vy += mean((abs.(Vy_pred[Vy_ref .!= 0.0] .- Vy_ref[Vy_ref.!= 0.0]).^(1/4)))^4
        # Squared-Mean-Root-Error
        # normVx = mean(abs.(Vx_ref[Vx_ref .!= 0.0f0]).^1/2)^2 #.+ ϵ
        # normVy = mean(abs.(Vy_ref[Vx_ref .!= 0.0f0]).^1/2)^2  #.+ ϵ
        # normVx = Vx_ref[Vx_ref .!= 0.0f0] .+ ϵ
        # normVy = Vy_ref[Vy_ref .!= 0.0f0] .+ ϵ
        # @show normVx
        # l_Vx += mean((abs.(Vx_pred[Vx_ref .!= 0.0f0]./normVx .- Vx_ref[Vx_ref .!= 0.0f0]./normVx).^(1/2)))^2
        # l_Vy += mean((abs.(Vy_pred[Vy_ref .!= 0.0f0]./normVy .- Vy_ref[Vy_ref .!= 0.0f0]./normVy).^(1/2)))^2
        # l_Vx += (1/normVx) * mean((abs.(Vx_pred[Vx_ref .!= 0.0f0] .- Vx_ref[Vx_ref .!= 0.0f0]).^(1/2)))^2
        # l_Vy += (1/normVy) * mean((abs.(Vy_pred[Vy_ref .!= 0.0f0] .- Vy_ref[Vy_ref .!= 0.0f0]).^(1/2)))^2

        # normV = (mean(Vx_ref.^1/2 .+ Vy_ref.^1/2))^2
        # l_V = Flux.Losses.mae(Vx_pred.^1/2, Vx_ref.^1/2; agg=mean) + Flux.Losses.mae(Vy_pred.^1/2, Vy_ref.^1/2; agg=mean)
        # l_V += normV^1.0f0 * log(l_V_local)

        # MAE
        # l_Vx += Flux.Losses.mae(Vx_pred[Vx_ref .!= 0.0]./normVx, Vx_ref[Vx_ref.!= 0.0]./normVx; agg=mean)
        # l_Vy += Flux.Losses.mae(Vy_pred[Vy_ref .!= 0.0]./normVy, Vy_ref[Vy_ref.!= 0.0]./normVy; agg=mean)

        # RMSE
        # l_Vx += Flux.Losses.mse(Vx_pred[Vx_ref .!= 0.0], Vx_ref[Vx_ref.!= 0.0]; agg=mean)^0.5
        # l_Vy += Flux.Losses.mse(Vy_pred[Vy_ref .!= 0.0], Vy_ref[Vy_ref.!= 0.0]; agg=mean)^0.5

        # Classic loss function with the full matrix
        # normV = mean(V_ref[V_ref .!= 0.0f0].^2)^0.5f0 #.+ ϵ
        normVx = mean(Vx_ref[Vx_ref .!= 0.0f0].^2)^0.5f0 #.+ ϵ
        normVy = mean(Vy_ref[Vy_ref .!= 0.0f0].^2)^0.5f0  #.+ ϵ
        # normVx = mean(Vx_ref[Vx_ref .!= 0.0f0]) #.+ ϵ
        # normVy = mean(Vy_ref[Vy_ref .!= 0.0f0]) #.+ ϵ
        # l_Vx += normV^(2) * Flux.Losses.mse(Vx_pred, Vx_ref; agg=mean)
        # l_Vy += normV^(2) * Flux.Losses.mse(Vy_pred, Vy_ref; agg=mean)

        ice_threshold = 10.0
        normV = (mean(Vx_ref.^2) + mean(Vy_ref.^2))^0.5f0
        # normV = maximum(Vx_ref[inn1(H_ref) .> 100.0].^2 .+ Vy_ref[inn1(H_ref) .> 100.0].^2)^0.5f0
        l_V_local = (normV)^(-2) * Flux.Losses.mse(Vx_pred[inn1(H_ref) .> ice_threshold], Vx_ref[inn1(H_ref) .> ice_threshold]; agg=mean) + Flux.Losses.mse(Vy_pred[inn1(H_ref) .> ice_threshold], Vy_ref[inn1(H_ref) .> ice_threshold]; agg=mean)
        l_V += l_V_local
        # l_V += normV^1.0f0 * log(l_V_local)        
    end 

    # We use the average loss between x and y V
    # l_V = (l_Vx + l_Vy)/2

    # Plot V diffs to understand training
    @ignore_derivatives begin
        plot_V_diffs(gdirs_climate, gdir_refs, V_preds)
    end

    return l_V, UD
end

"""
    perform_V_inversion(θ, UD, gdirs_climate, gtd_grids, context_batches)

Performs an inversion of the iceflow law with a UDE in different batches
"""
function perform_V_inversion(θ, UD, gdirs_climate_batch, gtd_grids_batch, context_batch, target)
    if all(isnothing.(gtd_grids_batch))
        years = gdirs_climate_batch[1][1]
        gtd_grids_batch = zeros(size(years))
    end
    V_preds = pmap((H, context, gdirs_climate) -> SIA(H, gdirs_climate, context, θ, UD[1], target[1]), gtd_grids_batch, context_batch, gdirs_climate_batch)

    return V_preds # (Vx, Vy, V)
end

callback_plots_inv = function (θ, l, UD_f, gdirs_climate, target) # callback function to observe training
    # Choose between the two callbacks to display the training progress
    training_path = joinpath(root_plots,"inversions")
    if target == "D"
        callback_plots_inv_D(θ, l, UD_f, training_path)
    elseif target == "A"
        gdirs = gdirs_climate[2]
        temps = gdirs_climate[3]
        A_noise = randn(rng_seed(), length(gdirs)).* noise_A_magnitude
        callback_plots_A(θ, l, UD_f, temps, A_noise, training_path)
    end
    false
end

callback_plots_inv_D = function (θ, l, UD_f, training_path) 
    println("Epoch #$(current_epoch[]) - Loss $(loss_type[]): ", l)
    # Let's explore the NN's output
    H = collect(20.0:100.0:1000.0)
    T = collect(-40.0:5.0:5.0)
    ∇S = collect(0.0:0.06:0.6) # 0 to 35 º (in rad)
    D_preds = []
    for (h,t,∇s) in zip(H,T,∇S)
        X = build_D_features(h, t, ∇s)
        push!(D_preds, predict_diffusivity(UD_f, θ, X)[1])
    end

    # TODO: make 3 plots with all combinations
    # Let's plot this in 3D   
    hs = collect(LinRange(minimum(H), maximum(H), length(H)))
    ts = collect(LinRange(minimum(T), maximum(T), length(H)))
    ss = collect(LinRange(minimum(∇S), maximum(∇S), length(H)))

    pHT = Plots.plot(hs, ts, D_preds, zcolor = reverse(D_preds), cbar = true, w = 3, 
                            xlabel="H", ylabel="T", zlabel="D")

    pH∇S = Plots.plot(hs, ss, D_preds, zcolor = reverse(D_preds), cbar = true, w = 3, 
                        xlabel="H", ylabel="∇S", zlabel="D")

    training_path = joinpath(root_plots,"inversions")
    generate_plot_folders(training_path)

    save_plot(pHT, training_path, "HT") 
    save_plot(pH∇S, training_path, "H∇S") 

    global current_epoch += 1
    push!(loss_history, l)

    plot_loss = Plots.plot(loss_history, label="", xlabel="Epoch", yaxis=:log10,
                ylabel="Loss (V)", lw = 3, c=:darkslategray3)

    save_plot(plot_loss, training_path, "loss") 

    false
end
