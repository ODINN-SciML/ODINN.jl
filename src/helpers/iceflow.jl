
export generate_ref_dataset, train_iceflow_UDE, spinup
export predict_A̅, A_fake

function spinup(gdirs_climate, tspan; solver = RDPK3Sp35(), random_MB=nothing)
    println("Spin up simulation for $(Int(tspan[2]) - Int(tspan[1])) years...\n")
    # Run batches in parallel
    climate_years_list = gdirs_climate[1]
    gdirs = gdirs_climate[2]
    longterm_temps = gdirs_climate[3]
    A_noises = randn(rng_seed(), length(gdirs)) .* noise_A_magnitude
    if isnothing(random_MB)
        refs = @showprogress pmap((gdir, longterm_temp, climate_years, A_noise) -> batch_iceflow_PDE(gdir, longterm_temp, climate_years, A_noise, tspan, solver;run_spinup=true), gdirs, longterm_temps, climate_years_list, A_noises)
    else
        refs = @showprogress pmap((gdir, longterm_temp, climate_years, A_noise, MB) -> batch_iceflow_PDE(gdir, longterm_temp, climate_years, A_noise, tspan, solver; run_spinup=true,random_MB=MB), gdirs, longterm_temps, climate_years_list, A_noises, random_MB)
    end

    # Gather information per gdir
    gdir_refs = get_gdir_refs(refs, gdirs)

    spinup_path = joinpath(ODINN.root_dir, "data/spinup")
    if !isdir(spinup_path)
        mkdir(spinup_path)
    end
    jldsave(joinpath(ODINN.root_dir, "data/spinup/gdir_refs.jld2"); gdir_refs)

    GC.gc() # run garbage collector
end

"""
    generate_ref_dataset(temp_series, H₀)

Generate reference dataset based on the iceflow PDE
"""
function generate_ref_dataset(gdirs_climate, tspan; solver = RDPK3Sp35(), random_MB=nothing)
  
    # Perform reference simulation with forward model 
    println("Running forward PDE ice flow model...\n")
    # Run batches in parallel
    gdirs = gdirs_climate[2]
    longterm_temps = gdirs_climate[3]
    climate_years_list = gdirs_climate[1]
    A_noises = randn(rng_seed(), length(gdirs)) .* noise_A_magnitude
    if isnothing(random_MB)
        # refs = @showprogress pmap((gdir, longterm_temp, climate_years, A_noise) -> batch_iceflow_PDE(gdir, longterm_temp, climate_years, A_noise, tspan, solver;run_spinup=false), gdirs, longterm_temps, climate_years_list, A_noises)
        refs = @showprogress pmap((gdir, longterm_temp, climate_years, A_noise) -> batch_iceflow_PDE(gdir, longterm_temp, climate_years, A_noise, tspan, solver;run_spinup=false), gdirs, longterm_temps, climate_years_list, A_noises)
    else
        # refs = @showprogress pmap((gdir, longterm_temp, climate_years, A_noise, MB) -> batch_iceflow_PDE(gdir, longterm_temp, climate_years, A_noise, tspan, solver; run_spinup=false,random_MB=MB), gdirs, longterm_temps, climate_years_list, A_noises, random_MB)
        refs = @showprogress pmap((gdir, longterm_temp, climate_years, A_noise, MB) -> batch_iceflow_PDE(gdir, longterm_temp, climate_years, A_noise, tspan, solver; run_spinup=false,random_MB=MB), gdirs, longterm_temps, climate_years_list, A_noises, random_MB)
    end

    # Gather information per gdir
    gdir_refs = get_gdir_refs(refs, gdirs)

    GC.gc() # run garbage collector 

    return gdir_refs
end

"""
    batch_iceflow_PDE(climate, gdir, context) 

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series batch
"""
function batch_iceflow_PDE(gdir, longterm_temp, years, A_noise, tspan, solver; run_spinup=false, random_MB=nothing) 
    println("Processing glacier: ", gdir.rgi_id)

    context, H = build_PDE_context(gdir, longterm_temp, years, A_noise, tspan; run_spinup=run_spinup, random_MB=random_MB)
    # Callback  
    if use_MB[] && !isnothing(random_MB)
        # Define stop times every one month
        tstops, step = define_callback_steps(tspan)
        stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, tstops) #closure
        function action!(integrator)
            year = floor(Int, integrator.t[end]) 
            compute_MB_matrix!(context, integrator.u, year) 
            MB = context[25]
            # Since this represents the anual mass balance, we need to distribute in 12 months
            integrator.u .+= MB .* step
        end
        cb_MB = DiscreteCallback(stop_condition, action!)
    elseif !use_MB[]
        tstops = []
        cb_MB = DiscreteCallback((u,t,integrator)->false, nothing)
    else
        throw(ArgumentError())
    end

    refs = simulate_iceflow_PDE(H, context, solver, tstops, cb_MB)

    return refs
end

"""
    simulate_iceflow_PDE(H, context, solver) 

Make forward simulation of the SIA PDE.
"""
function simulate_iceflow_PDE(H, context, solver, tstops, cb_MB)
    tspan = context[22]

    iceflow_prob = ODEProblem{true,SciMLBase.FullSpecialize}(iceflow!, H, tspan, tstops=tstops, context)
    iceflow_sol = solve(iceflow_prob, 
                        solver, 
                        callback=cb_MB, 
                        tstops=tstops, 
                        reltol=1e-7, 
                        save_everystep=false, 
                        progress=true, 
                        progress_steps=10)
    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    H_ref = iceflow_sol.u[end]
    H_ref[H_ref.<0.0] .= H_ref[H_ref.<0.0] .* 0.0 # remove remaining negative values
    temps = context[7]
    B = context[2]
    V̄x_ref, V̄y_ref = avg_surface_V(context, H_ref, mean(temps), "PDE") # Average velocity with average temperature
    S = B .+ H_ref # Surface topography
    refs = Dict("Vx"=>V̄x_ref, "Vy"=>V̄y_ref, "H"=>H_ref, "S"=>S, "B"=>B)
    return refs
end

"""
    train_iceflow_UDE(gdirs_climate, gdirs_climate_batches, gdir_refs, tspan, 
                        train_settings=(Adam(), 1, length(gdirs_climate[1])), 
                        θ_trained=[], UDE_settings=nothing, loss_history=[]; random_MB=nothing)

Train the Shallow Ice Approximation iceflow UDE. UDE_settings is optional, and requires a Dict specifiying the `reltol`, 
`sensealg` and `solver` for the UDE.  
"""
function train_iceflow_UDE(gdirs_climate, gdirs_climate_batches, gdir_refs, tspan, 
                            train_settings=nothing, θ_trained=[], UDE_settings=nothing, loss_history=[]; 
                            random_MB=nothing)

    # Fill default training settings if not available 
    UDE_settings, train_settings = get_default_training_settings!(gdirs_climate, 
                                                                UDE_settings, train_settings, 
                                                                θ_trained, random_MB)

    optimizer = train_settings[1]
    epochs = train_settings[2]
    batch_size = train_settings[3]
    n_gdirs = length(gdirs_climate[2])
    UA_f, θ = get_NN(θ_trained)
    gdirs = gdirs_climate[2]

    # Build context for all the batches before training
    println("Building context...")
    context_batches = get_UDE_context(gdirs_climate, tspan, random_MB)

    println("Training iceflow UDE...")
    temps = gdirs_climate[3]
    A_noise = randn(rng_seed(), length(gdirs)).* noise_A_magnitude
    training_path = joinpath(root_plots,"training")
    cb_plots(θ, l, UA_f) = callback_plots_A(θ, l, UA_f, temps, A_noise, training_path, batch_size, n_gdirs)
    # Create batches for inversion training 
    train_loader = generate_batches(batch_size, UA_f, gdirs_climate_batches, context_batches, gdir_refs, UDE_settings)

    # Setup optimization of the problem
    if optimization_method == "AD+AD"
        println("Optimization based on pure AD")
        optf = OptimizationFunction((θ, _, UA_batch, gdirs_climate_batch, context_batch, gdir_refs_batch, UDE_settings_batch)->loss_iceflow(θ, UA_batch, gdirs_climate_batch, context_batch, gdir_refs_batch, UDE_settings_batch), Optimization.AutoZygote())
        optprob = OptimizationProblem(optf, θ)
        iceflow_trained = solve(optprob, 
                                optimizer, ncycle(train_loader, epochs), allow_f_increases=true,
                                callback=cb_plots, progress=true)

    elseif optimization_method == "AD+Diff"
        println("Optimization based on AD for NN and finite differences for ODE solver")
        η = 0.01
        p = [temps]
        function loc_loss(θ, p)
            loss_iceflow(θ, UA_f, gdirs_climate, context_batches, gdir_refs, UDE_settings)[1] 
        end
        function customized_grad!(g::Vector, θ, p)
            # compute output of NN using temps 
            temp = mean(p[1][1])  # Is this the best way of getting the temperature? 
            ∇θ_A = gradient(θ -> UA_f(θ)([temp])[1], θ)[1]
            loss₀ = loc_loss(θ, p)
            θ₁ = Array{Float32}(θ .+ η .* ∇θ_A)
            loss₁ = loc_loss(θ₁, p)
            scalar_factor = (loss₁ - loss₀) ./ (η * norm(∇θ_A)^2)
            g .= scalar_factor .* ∇θ_A
        end   
        optf = OptimizationFunction((θ,_)->loss(θ), 
                                    grad=customized_grad!)
        optprob = OptimizationProblem(optf, θ, p)
        # cb_plots(θ, l) = callback_plots(θ, l, UA_f, temps, A_noise)
        iceflow_trained = solve(optprob, optimizer, callback=cb_plots, maxiters=epochs)
    end

    return iceflow_trained, UA_f, loss_history
end


"""
    loss_iceflow(θ, UA_f, gdirs_climate, context_batches, gdir_refs::Dict{String, Any}, UDE_settings)  

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow(θ, UA_f, gdirs_climate, context_batches, gdir_refs, UDE_settings) 
    # UA_f and UDE_settings need to be passed as scalars since they were transformed to Vectors for the batches
    H_V_preds = predict_iceflow(θ, UA_f[1], gdirs_climate, context_batches, UDE_settings[1])
    # Compute loss function for the full batch
    l_V, l_Vx, l_Vy, l_H = 0.0, 0.0, 0.0, 0.0
    for i in 1:length(H_V_preds)
        # Get ice thickness from the reference dataset
        H_ref = gdir_refs[i]["H"]
        # Get ice velocities for the reference dataset
        Vx_ref = gdir_refs[i]["Vx"]
        Vy_ref = gdir_refs[i]["Vy"]
        # Get ice thickness from the UDE predictions
        H = H_V_preds[i][1]
        # Get ice velocities prediction from the UDE
        V̄x_pred = H_V_preds[i][2]
        V̄y_pred = H_V_preds[i][3]

        if scale_loss[]
            normHref = mean(H_ref.^2.0)^0.5
            normVref = mean(Vx_ref.^2.0 .+ Vy_ref.^2.0)^0.5
            # normV = mean(abs.(Vx_ref .+ Vy_ref))
            l_H_loc = Flux.Losses.mse(H, H_ref; agg=mean) 
            l_V_loc = Flux.Losses.mse(V̄x_pred, Vx_ref; agg=mean) + Flux.Losses.mse(V̄y_pred, Vy_ref; agg=mean)
            # l_V_loc = mean(abs.(V̄x_pred .- Vx_ref)) + mean(abs.(V̄y_pred .- Vy_ref))
            l_H += normHref^(-2.0) * l_H_loc
            l_V += normVref^(2.0) * log(l_V_loc)
            # l_V += normVref * log(l_V_loc)
            # l_V += normVref * l_V_loc^(1/3)

            # @ignore begin
            # normVref_series = (Vx_ref.^2.0 .+ Vy_ref.^2.0).^0.5
            # normVpred_series = (V̄x_pred.^2.0 .+ V̄y_pred.^2.0).^0.5
            # display(Plots.heatmap(normVref_series, title="Reference average surface velocities", c=:speed))
            # display(Plots.heatmap(normVpred_series, title="Predicted average surface velocities", c=:speed))
            # end

        else
            l_H += Flux.Losses.mse(H[H_ref .!= 0.0], H_ref[H_ref.!= 0.0]; agg=mean) 
            l_Vx += Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0], Vx_ref[Vx_ref.!= 0.0]; agg=mean)
            l_Vy += Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0], Vy_ref[Vy_ref.!= 0.0]; agg=mean)
            l_V += l_Vx + l_Vy
        end
    end

    @assert (loss_type[] == "H" || loss_type[] == "V" || loss_type[] == "HV") "Invalid loss_type. Needs to be 'H', 'V' or 'HV'"
    if loss_type[] == "H"
        l_tot = l_H/length(gdir_refs)
    elseif loss_type[] == "V"
        l_tot = l_V/length(gdir_refs) 
    elseif loss_type[] == "HV"
        l_tot = (l_V + l_H)/length(gdir_refs)
    end
    return l_tot, UA_f[1]
end

"""
    predict_iceflow(θ, UA_f, gdirs_climate, context_batches, UDE_settings)

Makes a prediction of glacier evolution with the UDE for a given temperature series in different batches
"""
function predict_iceflow(θ, UA_f, gdirs_climate_batches, context_batches, UDE_settings; testmode=false)
    # Train UDE in parallel
    H_V_pred = pmap((context, gdirs_climate_batch) -> batch_iceflow_UDE(θ, UA_f, context, gdirs_climate_batch, UDE_settings; testmode=testmode), context_batches, gdirs_climate_batches)
    return H_V_pred
end


"""
    batch_iceflow_UDE(θ, UA_f, context, longterm_temps_batch, UDE_settings, testmode) 

Solve the Shallow Ice Approximation iceflow UDE for a given temperature series batch
"""
function batch_iceflow_UDE(θ, UA_f, context, gdirs_climate_batch, UDE_settings; testmode) 
    # context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)
    H = context[2]
    tspan = context[6]
    longterm_temps_batch = gdirs_climate_batch[3]

    # Callback  
    if use_MB[] 
        # Define stop times every one month
        tstops, step = define_callback_steps(tspan)
        stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, tstops) #closure
        function action!(integrator)
            year = floor(Int, integrator.t[end])  
            MB = compute_MB_matrix(context, integrator.u, year)
            # Since the mass balance is anual, we need to divide per month
            integrator.u .+= MB .* step
        end
        cb_MB = DiscreteCallback(stop_condition, action!)
    else
        tstops = []
        cb_MB = DiscreteCallback((u,t,integrator)->false, nothing)
    end
    
    iceflow_UDE_batch(H, θ, t) = iceflow_NN(H, θ, t, UA_f, context, longterm_temps_batch, testmode) # closure
    iceflow_prob = ODEProblem(iceflow_UDE_batch, H, tspan, tstops=tstops, θ)
    iceflow_sol = solve(iceflow_prob, 
                        UDE_settings["solver"], 
                        callback=cb_MB,
                        tstops=tstops,
                        u0=H, 
                        p=θ,
                        sensealg=UDE_settings["sensealg"],
                        reltol=UDE_settings["reltol"], 
                        save_everystep=false,  
                        progress=false, 
                        progress_steps=50)
    # @show iceflow_sol.destats
    # Get ice velocities from the UDE predictions
    H_end = iceflow_sol.u[end]
    H_pred = ifelse.(H_end .< 0.0, 0.0, H_end)
    V̄x_pred, V̄y_pred = avg_surface_V(context, H_pred, mean(longterm_temps_batch), "UDE", θ, UA_f; testmode) # Average velocity with average temperature
    rgi_id = @ignore gdirs_climate_batch[2].rgi_id
    H_V_pred = (H_pred, V̄x_pred, V̄y_pred, rgi_id)
    # H_V_pred = (H_pred, V̄x_pred, V̄y_pred)
    return H_V_pred
end

"""
    iceflow!(dH, H, context,t)

Runs a single time step of the iceflow PDE model in-place
"""
function iceflow!(dH, H, context,t)
    # First, enforce values to be positive
    H[H.<0.0] .= H[H.<0.0] .* 0.0
    # Unpack parameters
    #A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year 
    current_year = context[18]
    A = context[1]
    t₁ = context[22][end]
    A_noise = context[23]
    climate_years = context[30]

    # Get current year for MB and ELA
    year = floor(Int, t) 
    if year != current_year && year <= t₁ 
        temp = context[7][year .== climate_years]
        A[] = A_fake(temp, A_noise, noise)[1]
        current_year .= year
    end

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, context)
end    

"""
    iceflow_NN(H, θ, t, UA_f, context, temps)

Runs a single time step of the iceflow UDE model 
"""
function iceflow_NN(H, θ, t, UA_f, context, temps, testmode)
    H_buf = Buffer(H)
    @views H_buf .= ifelse.(H.<0.0, 0.0, H) # prevent values from going negative
    H = copy(H_buf)
    climate_years = context[11]

    year = floor(Int, t) 
    temp = temps[year .== climate_years]
    if testmode
        A = A_fake(temp)[1]
    else
        A = predict_A̅(UA_f, θ, temp)[1]
    end
    # Compute the Shallow Ice Approximation in a staggered grid
    # if !use_MB[] && typeof(A) <: AbstractFloat
    #     dH = SIA(H, A, context) 
    # elseif use_MB[]
    #     # When including MB, Zygote returns 
    #     dH = SIA(H, A.value, context) 
    # else
    #     throw(DomainError(A, "Error with type of A"))
    # end
    dH = SIA(H, A, context) 

    return dH
end  

"""
    SIA!(dH, H, context)

Compute an in-place step of the Shallow Ice Approximation PDE in a forward model
"""
function SIA!(dH, H, context)
    # Retrieve parameters
    #[A], B, S, dSdx, dSdy, D, copy(temp_series[1]), dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, [current_year], nxy, Δxy
    A = context[1]
    B = context[2]
    S = context[3]
    dSdx = context[4]
    dSdy = context[5]
    D = context[6]
    dSdx_edges = context[8]
    dSdy_edges = context[9]
    ∇S = context[10]
    Fx = context[11]
    Fy = context[12]
    Δx = context[20][1]
    Δy = context[20][2]
    Γ = context[27]

    # Update glacier surface altimetry
    S .= B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    diff_x!(dSdx, S, Δx)  
    diff_y!(dSdy, S, Δy)  
    ∇S .= (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 

    Γ[] = 2.0 * A[] * (ρ[] * g[])^n[] / (n[]+2) # 1 / m^3 s 
    D .= Γ .* avg(H).^(n[] + 2) .* ∇S

    # Compute flux components
    @views diff_x!(dSdx_edges, S[:,2:end - 1], Δx)
    @views diff_y!(dSdy_edges, S[2:end - 1,:], Δy)
    Fx .= .-avg_y(D) .* dSdx_edges
    Fy .= .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    inn(dH) .= .-(diff_x(Fx) ./ Δx .+ diff_y(Fy) ./ Δy) 
end

"""
    SIA(H, A, context)

Compute a step of the Shallow Ice Approximation UDE in a forward model. Allocates memory.
"""
function SIA(H, A, context)
    # Retrieve parameters
    # context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)
    B = context[1]
    Δx = context[5][1]
    Δy = context[5][2]

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) ./ Δx
    dSdy = diff_y(S) ./ Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 

    Γ = 2.0 * A * (ρ[] * g[])^n[] / (n[]+2) # 1 / m^3 s 
    D = Γ .* avg(H).^(n[] + 2) .* ∇S

    # Compute flux components
    @views dSdx_edges = diff_x(S[:,2:end - 1]) ./ Δx
    @views dSdy_edges = diff_y(S[2:end - 1,:]) ./ Δy
    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    @tullio dH[i,j] := -(diff_x(Fx)[pad(i-1,1,1),pad(j-1,1,1)] / Δx + diff_y(Fy)[pad(i-1,1,1),pad(j-1,1,1)] / Δy) # MB to be added here 

    return dH
end

"""
    SIA(H, T, context, years, θ, UD_f, target)

Compute a step of the Shallow Ice Approximation, returning ice surface velocities. Allocates memory.
"""
function SIA(H, T, context, years, θ, UD_f, target)
    @assert (target == "A" || target == "D") "Functional inversion target needs to be either A or D!"
    # Retrieve parameters
    # context = (nxy, Δxy, tspan, rgi_id, S, V)
    Δx = context[2][1]
    Δy = context[2][2]
    S = context[5] # TODO: update this to Millan et al.(2022) DEM

    if H==0.0 # Retrieve H from context if Glathida thickness is not present 
        H = context[7]
    end

    # Get long term temperature for the Millan et al.(2022) dataset
    temp = T[years .== 2017]

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) ./ Δx
    dSdy = diff_y(S) ./ Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 
    
    if target == "D"
        X = build_D_features(H, temp, ∇S)
        D = predict_diffusivity(UD_f, θ, X)
    elseif target == "A"
        A = predict_A̅(UD_f, θ, temp)
        Γ = 2.0 * A * (ρ[] * g[])^n[] / (n[]+2.0) # 1 / m^3 s 
        D = Γ .* inn1(H).^(n[] + 2.0) .* ∇S
    end
    V = D .* abs.(∇S[inn1(H) .!= 0.0])
    return V
end

"""
    avg_surface_V(context, H, temp, sim, θ=[], UA_f=[])

Computes the average ice surface velocity for a given glacier evolution period
based on the initial and final ice thickness states. 
"""
function avg_surface_V(context, H, temp, sim, θ=[], UA_f=[]; testmode=false)
    # context = (B, H₀, H, nxy, Δxy)
    B, H₀, Δx, Δy, A_noise = retrieve_context(context, sim)
    # We compute the initial and final surface velocity and average them
    # TODO: Add more datapoints to better interpolate this
    Vx₀, Vy₀ = surface_V(H₀, B, Δx, Δy, temp, sim, A_noise, θ, UA_f; testmode=testmode)
    Vx, Vy = surface_V(H, B, Δx, Δy, temp, sim, A_noise, θ, UA_f; testmode=testmode)
    V̄x = (Vx₀ .+ Vx)./2.0
    V̄y = (Vy₀ .+ Vy)./2.0

    return V̄x, V̄y
        
end

"""
    surface_V(H, B, Δx, Δy, temp, sim, A_noise, θ=[], UA_f=[])

Computes the ice surface velocity for a given glacier state
"""
function surface_V(H, B, Δx, Δy, temp, sim, A_noise, θ=[], UA_f=[]; testmode=false)
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 
    
    @assert (sim == "UDE" || sim == "PDE") "Wrong type of simulation. Needs to be 'UDE' or 'PDE'."
    if sim == "UDE" && !testmode
        A = predict_A̅(UA_f, θ, [temp]) 
    elseif sim == "PDE" || testmode
        A = A_fake(temp, A_noise, noise)
    end
    Γꜛ = 2.0 * A * (ρ[] * g[])^n[] / (n[]+1) # surface stress (not average)  # 1 / m^3 s 
    D = Γꜛ .* avg(H).^(n[] + 1) .* ∇S
    
    # Compute averaged surface velocities
    Vx = - D .* avg_y(dSdx)
    Vy = - D .* avg_x(dSdy)

    return Vx, Vy    
end

"""
    define_callback_steps(tspan; step=1.0/12.0)

Defines the times to stop for the DiscreteCallback given a step
"""
function define_callback_steps(tspan; step=1.0/12.0)
    tmin_int = Int(tspan[1])
    tmax_int = Int(tspan[2])+1
    tstops = range(tmin_int+step, tmax_int, step=step) |> collect
    tstops = filter(x->( (Int(tspan[1])<x) & (x<=Int(tspan[2])) ), tstops)
    return tstops, step
end

"""
    stop_condition_tstops(u,t,integrator, tstops)  

Function that iterates through the tstops, with a closure including `tstops`
"""
function stop_condition_tstops(u,t,integrator, tstops) 
    t in tstops
end

callback_plots_A = function (θ, l, UA_f, temps, A_noise, training_path, batch_size, n_gdirs) # callback function to observe training
    # Update training status
    update_training_state(l, batch_size, n_gdirs)

    if current_minibatches == 0.0
        avg_temps = [mean(temps[i]) for i in 1:length(temps)]
        p = sortperm(avg_temps)
        avg_temps = avg_temps[p]
        pred_A = predict_A̅(UA_f, θ, collect(-23.0:1.0:0.0)')
        pred_A = [pred_A...] # flatten
        true_A = A_fake(avg_temps, A_noise[p], noise)

        yticks = collect(0.0:2e-17:8e-17)

        Plots.scatter(avg_temps, true_A, label="True A", c=:lightsteelblue2)
        plot_epoch = Plots.plot!(-23:1:0, pred_A, label="Predicted A", 
                            xlabel="Long-term air temperature (°C)", yticks=yticks,
                            ylabel="A", ylims=(0.0,maxA[]), lw = 3, c=:dodgerblue4,
                            legend=:topleft)
        if !isdir(joinpath(training_path,"png")) || !isdir(joinpath(training_path,"pdf"))
            mkpath(joinpath(training_path,"png"))
            mkpath(joinpath(training_path,"pdf"))
        end
        # Plots.savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.svg"))
        Plots.savefig(plot_epoch,joinpath(training_path,"png","epoch$(current_epoch).png"))
        Plots.savefig(plot_epoch,joinpath(training_path,"pdf","epoch$(current_epoch).pdf"))

        plot_loss = Plots.plot(loss_history, label="", xlabel="Epoch", yaxis=:log10,
                    ylabel="Loss (V)", lw = 3, c=:darkslategray3)
        Plots.savefig(plot_loss,joinpath(training_path,"png","loss$(current_epoch).png"))
        Plots.savefig(plot_loss,joinpath(training_path,"pdf","loss$(current_epoch).pdf"))
    end

    GC.gc()

    return false
end
