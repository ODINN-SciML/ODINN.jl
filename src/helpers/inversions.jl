export train_iceflow_inversion

"""
invert_iceflow(gdirs, train_settings, θ_trained=[], loss_history=[])

Performs an inversion on the SIA to train a NN on the ice flow law
"""
function train_iceflow_inversion(rgi_ids, tspan, train_settings; gdir_refs=nothing, gtd_file=nothing, θ_trained=[], target="D")
    println("Training ice rheology inversion...")
    # filter_missing_glaciers!(rgi_ids) # already done in init_gdirs()
    # Initialize gdirs with ice thickness data
    gdirs = init_gdirs(rgi_ids)
    # Process climate data for glaciers
    gdirs_climate = get_gdirs_with_climate(gdirs, tspan, overwrite=false, massbalance=false, plot=false)
    if !isnothing(gtd_file)
        # Produce Glathida dataset
        gtd_grids = get_glathida!(gtd_file, gdirs; force=false)
    else
        gtd_grids=nothing
    end
    # Perform inversion with the given gdirs and climate data
    rheology_trained = invert_iceflow(gdirs_climate, gtd_grids, gdir_refs, tspan, train_settings, θ_trained, target)

    return rheology_trained
end

"""
invert_iceflow(gdirs, train_settings, θ_trained=[], loss_history=[])

Performs an inversion on the SIA to train a NN on the ice flow law
"""
function invert_iceflow(gdirs_climate, gtd_grids, gdir_refs, tspan, train_settings, θ_trained, target)
    if length(θ_trained) == 0
        global current_epoch[] = 1 # reset epoch count
    end
    optimizer = train_settings[1]
    epochs = train_settings[2]
    UD, θ = get_NN_inversion(θ_trained, target)
    gdirs = gdirs_climate[2]

    # Build context for all the batches before training
    println("Building context...")
    context_batches = try 
         map(gdir -> build_UDE_context_inv(gdir, tspan), gdirs)
    catch error
        @error "$error: Missing data for some glaciers. The list of missing_glaciers has been updated. Try again."
    end
    loss(θ) = loss_iceflow_inversion(θ, UD, gdirs_climate, gtd_grids, gdir_refs, context_batches, target) # closure
    
    cb_plots_inv(θ, l, UD) = callback_plots_inv(θ, l, UD)

    # Setup optimization of the problem
    optf = OptimizationFunction((θ,_)->loss(θ), Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, θ)
    println("Training iceflow rheology inversion...")
    rheology_trained = solve(optprob, optimizer, maxiters=epochs, allow_f_increases=true, callback=cb_plots_inv, progress=true)

    return rheology_trained
end

"""
    loss_iceflow_inversion(θ, UD, gdirs_climate, gtd_grids, context_batches)

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow_inversion(θ, UD, gdirs_climate, gtd_grids, gdir_refs, context_batches, target)

    V_preds = perform_V_inversion(θ, UD, gdirs_climate, gtd_grids, context_batches, target)

    # Compute loss function for the full batch
    l_V = 0.0f0
    for i in 1:length(V_preds)
        # TODO: choose between Millan22 or simulated reference V_preds
        if isnothing(gtd_grids)
            Vx_ref = gdir_refs[i]["Vx"]
            Vy_ref = gdir_refs[i]["Vy"]
            V_ref = sqrt.(Vx_ref.^2 .+ Vy_ref.^2)
            @ignore @infiltrate
            H = context_batches[i].x[21]
        else
            # Get ice velocities from Millan et al. (2022)
            V_ref = avg(context_batches[i][6])
        end

        if scale_loss[]
            # We only evaluate the loss where there are ice thickness obs
            normV = V_ref[inn1(gtd_grids[i]) .!= 0.0] .+ ϵ
            l_V += Flux.Losses.mse(V_preds[i] ./normV, V_ref[inn1(gtd_grids[i]) .!= 0.0] ./normV; agg=mean)
        else
            l_V += Flux.Losses.mse(V_preds[i], V_ref[inn1(gtd_grids[i]) .!= 0.0]; agg=mean)
        end
    end 

    return l_V, UD
end

"""
    perform_V_inversion(θ, UD, gdirs_climate, gtd_grids, context_batches)

Performs an inversion of the iceflow law with a UDE in different batches
"""
function perform_V_inversion(θ, UD, gdirs_climate, gtd_grids, context_batches, target)
    T_batches = gdirs_climate[3]
    years = gdirs_climate[1][1]
    if isnothing(gtd_grids)
        gtd_grids = zeros(size(T_batches))
    end
    V = map((H, context, T) -> SIA(H, T, context, years, θ, UD, target), gtd_grids, context_batches, T_batches)
    return V
end

callback_plots_inv = function (θ, l, UD_f) # callback function to observe training
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

    global current_epoch[] += 1
    push!(loss_history, l)

    plot_loss = Plots.plot(loss_history, label="", xlabel="Epoch", yaxis=:log10,
                ylabel="Loss (V)", lw = 3, c=:darkslategray3)

    save_plot(plot_loss, training_path, "loss") 

    false
end