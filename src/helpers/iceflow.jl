
export generate_ref_dataset, train_iceflow_UDE, spinup

function spinup(gdirs, tspan; solver = RDPK3Sp35())
    println("Spin up simulation for $(Int(tspan[2]) - Int(tspan[1])) years...\n")
    # Run batches in parallel
    A_noises = randn(rng_seed(), length(gdirs)) .* noise_A_magnitude
    refs = @showprogress pmap((gdir, A_noise) -> batch_iceflow_PDE(gdir, A_noise, tspan, solver; run_spinup=true), gdirs, A_noises)

    # Gather information per gdir
    gdir_refs = get_gdir_refs(refs, gdirs)

    spinup_path = joinpath(ODINN.root_dir, "data/spinup")
    if !isdir(spinup_path)
        mkdir(spinup_path)
    end
    jldsave(joinpath(ODINN.root_dir, "data/spinup/gdir_refs.jld2"); gdir_refs)

    @everywhere GC.gc() # run garbage collector
end

"""
    generate_ref_dataset(gdirs_climate, tspan; solver = RDPK3Sp35(), random_MB=nothing)

Generate reference dataset based on the iceflow PDE
"""
function generate_ref_dataset(gdirs, tspan; solver = RDPK3Sp35())
    # Generate climate data if necessary
    @timeit to "generate raw climate files" begin
    pmap((gdir) -> generate_raw_climate_files(gdir, tspan), gdirs)
    end
    # Perform reference simulation with forward model 
    println("Running forward PDE ice flow model...\n")
    # Run batches in parallel
    A_noises = randn(rng_seed(), length(gdirs)) .* noise_A_magnitude
    refs = @showprogress pmap((gdir, A_noise) -> batch_iceflow_PDE(gdir, A_noise, tspan, solver; run_spinup=false), gdirs, A_noises)

    # Gather information per gdir
    gdir_refs = get_gdir_refs(refs, gdirs)

    @everywhere GC.gc() # run garbage collector 

    return gdir_refs
end

"""
    batch_iceflow_PDE(gdir, longterm_temp, years, A_noise, tspan, solver; run_spinup=false)

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series batch
"""
function batch_iceflow_PDE(gdir, A_noise, tspan, solver; run_spinup=false) 
    println("Processing glacier: ", gdir.rgi_id)
    _, step = define_callback_steps(tspan)
    context, H = build_PDE_context(gdir ,A_noise, tspan; run_spinup=run_spinup)
    S = context[3]
    S_coords = context[32]
    # This needs to be passed in a cleaner way
    mb_model = TI_model_1(DDF=5.0/1000.0, acc_factor=1.2/1000.0) # in m.w.e.
    # Initialize climate dataset
    climate = init_climate(gdir, tspan, step, S, S_coords)
    
    # Callback  
    # Define stop times every one month
    tstops, step = define_callback_steps(tspan)
    stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, tstops) #closure
    function action!(integrator)
        if use_MB 
            # Compute mass balance
            @timeit to "MB PDE" begin
            MB = context[25]
            S = context[3]
            S_coords = context[32]
            MB_timestep!(MB, mb_model, climate, S, S_coords, integrator.t[end], step)
            integrator.u .+= MB
            end
        end
        # Recompute A value
        @timeit to "Updating A PDE" begin
        A = context[1]
        A_noise = context[23]
        A[] = A_fake(mean(climate.longterm_temps), A_noise, noise)[1]
        end
    end
    cb_MB = DiscreteCallback(stop_condition, action!)

    refs = @timeit to "simulate_iceflow_PDE" simulate_iceflow_PDE(H, context, climate, solver, tstops, cb_MB)

    return refs
end

"""
    function simulate_iceflow_PDE(H, context, solver, tstops, cb_MB, θ=nothing, UA_f=nothing; du=iceflow!, sim="PDE")

Make forward simulation of the SIA PDE.
"""
function simulate_iceflow_PDE(H, context, climate, solver, tstops, cb_MB, θ=Vector{Float32}[], UA_f=nothing; du=iceflow!, sim="PDE")
    tspan = context[22]
    if sim == "UDE_inplace"
        progress=false
    elseif sim == "PDE"
        progress=true
    end
    iceflow_prob = ODEProblem{true,SciMLBase.FullSpecialize}(du, H, tspan, tstops=tstops, context)
    iceflow_sol = solve(iceflow_prob, 
                        solver, 
                        callback=cb_MB, 
                        tstops=tstops, 
                        reltol=1e-7, 
                        save_everystep=false, 
                        progress=progress, 
                        progress_steps=10)
    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    H_ref::Matrix{Float64} = iceflow_sol.u[end]
    H_ref[H_ref.<0.0] .= H_ref[H_ref.<0.0] .* 0.0 # remove remaining negative values
    B::Matrix{Float64} = context[2]
    S::Matrix{Float64} = context[3]
    V̄x_ref, V̄y_ref = avg_surface_V(context, H_ref, mean(climate.longterm_temps), sim, θ, UA_f) # Average velocity with average temperature
    S .= B .+ H_ref # Surface topography
    refs = Dict("Vx"=>V̄x_ref, "Vy"=>V̄y_ref, "H"=>H_ref, "S"=>S, "B"=>B)
    return refs
end

"""
function train_iceflow_UDE(gdirs_climate, gdirs_climate_batches, gdir_refs, tspan, 
                            train_settings=nothing, θ_trained=[], loss_history=[]; 
                            UDE_settings=nothing)

Train the Shallow Ice Approximation iceflow UDE. UDE_settings is optional, and requires a Dict specifiying the `reltol`, 
`sensealg` and `solver` for the UDE.  
"""
function train_iceflow_UDE(gdirs, gdir_refs, tspan, 
                            train_settings=nothing, θ_trained=[], loss_history=[]; 
                            UDE_settings=nothing)

    # Fill default training settings if not available 
    UDE_settings, train_settings = get_default_training_settings!(gdirs, 
                                                                UDE_settings, train_settings, 
                                                                θ_trained)

    # Generate climate data if necessary
    @timeit to "generate raw climate files" begin
    pmap((gdir) -> generate_raw_climate_files(gdir, tspan), gdirs)
    end

    optimizer = train_settings[1]
    epochs = train_settings[2]
    batch_size::Int = train_settings[3]
    n_gdirs::Int = length(gdirs)
    UA_f, θ = get_NN(θ_trained)

    A_noise = randn(rng_seed(), length(gdirs)).* noise_A_magnitude
    training_path = joinpath(root_plots,"training")
    @timeit to "get longterm temps" begin
    longterm_temps = pmap((gdir) -> get_longterm_temps(gdir), gdirs)
    end
    cb_plots(θ, l, UA_f) = callback_plots_A(θ, l, UA_f, longterm_temps, A_noise, training_path, batch_size, n_gdirs)

    # Setup optimization of the problem
    if optimization_method == "AD+AD"
        println("Optimization based on pure AD")
        # Build context for all the batches before training
        println("Building context...")
        context_batches = get_UDE_context(gdirs, tspan)
        # Create batches for inversion training 
        train_loader = generate_batches(batch_size, UA_f, gdirs, context_batches, gdir_refs, UDE_settings)
        optf = OptimizationFunction((θ, _, UA_batch, gdir, context_batch, gdir_refs_batch, UDE_settings_batch)->loss_iceflow(θ, UA_batch, gdir, context_batch, gdir_refs_batch, UDE_settings_batch), Optimization.AutoZygote())
        optprob = OptimizationProblem(optf, θ)
        println("Training iceflow UDE...")
        iceflow_trained = solve(optprob, 
                                optimizer, ncycle(train_loader, epochs), allow_f_increases=true,
                                callback=cb_plots, progress=true)

    elseif optimization_method == "AD+Diff"
        println("Optimization based on AD for NN and finite differences for ODE solver")

        function update_gradient_glacier!(g::Vector, θ, UA_f, gdir, gdir_ref_batch, temps, tspan)
            # AD for the NN gradient
            ∇θ_A = gradient(θ -> UA_f(θ)([mean(temps)])[1], θ)[1]
            # Compute scaling factor for gradient
            grad_norm = Float64(norm(∇θ_A))
            A₀ = UA_f(θ)([mean(temps)])[1]
            δ = 2.0^10 * eps(Float64)^0.5
            η =  δ * A₀ / grad_norm^2

            # Do for loop to see if there are wiggles. 
            θ₋ = Array{Float64}(θ .- η .* ∇θ_A)
            θ₊ = Array{Float64}(θ .+ η .* ∇θ_A)
            loss₋, _ = loss_iceflow_finite_differences(θ₋, [UA_f], [gdir], [gdir_ref_batch], [tspan]) 
            loss₊, _ = loss_iceflow_finite_differences(θ₊, [UA_f], [gdir], [gdir_ref_batch], [tspan])
            scalar_factor = (loss₊ - loss₋) ./ (2 * δ * A₀)
            g .+= scalar_factor .* ∇θ_A # update gradient
        end

        # We compute the combined gradient for all batches in parallel
        function customized_grad!(g::Vector, θ, _, UA_fs, gdirs, gdir_refs_batches, tspan)
            map((gdir, gdir_ref_batch, temps) -> update_gradient_glacier!(g::Vector, θ, UA_fs[1], gdir, gdir_ref_batch, temps, tspan[1]), 
                gdirs, gdir_refs_batches, longterm_temps)
        end   

        train_loader = generate_batches(batch_size, UA_f, gdirs, gdir_refs, tspan)
        optf = OptimizationFunction((θ, _, UA_f, gdirs, gdir_refs, tspan) -> loss_iceflow_finite_differences(θ, UA_f, gdirs, gdir_refs, tspan), 
                                    Optimization.AutoZygote(), # only necessary because of a bug. To be removed soon.                            
                                    grad=customized_grad!)

        optprob = OptimizationProblem(optf, θ)
        println("Training iceflow UDE...")
        iceflow_trained = solve(optprob, 
                                optimizer, 
                                ncycle(train_loader, epochs), allow_f_increases=true,
                                callback=cb_plots, maxiters=epochs,
                                progress=true)
    end

    @everywhere GC.gc() # run garbage collector 

    return iceflow_trained, UA_f, loss_history
end


"""
    loss_iceflow(θ, UA_f, gdirs_climate, context_batches, gdir_refs::Dict{String, Any}, UDE_settings)  

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow(θ, UA_f, gdirs, context_batches, gdir_refs, UDE_settings) 
    # UA_f and UDE_settings need to be passed as scalars since they were transformed to Vectors for the batches
    H_V_preds = predict_iceflow(θ, UA_f[1], gdirs, context_batches, UDE_settings[1])

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
            normHref = mean(H_ref.^2)^0.5
            normVref = mean(Vx_ref.^2 .+ Vy_ref.^2)^0.5
            l_H_loc = Flux.Losses.mse(H, H_ref; agg=mean) 
            l_V_loc = Flux.Losses.mse(V̄x_pred, Vx_ref; agg=mean) + Flux.Losses.mse(V̄y_pred, Vy_ref; agg=mean)
            l_H += normHref^(-1) * l_H_loc
            l_V += normVref^(-1) * l_V_loc

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

# TODO: reduce redundancy with loss function above
function loss_iceflow_finite_differences(θ, UA_f, gdirs, gdir_refs, tspan) 
    
    gdir_preds = predict_iceflow_inplace(θ, UA_f, gdirs, tspan)

    # Compute loss function for a single glacier
    l_tot = 0.0
    for i in 1:length(gdir_preds)
        # Get ice thickness from the reference dataset
        H_ref::Matrix{Float64} = gdir_refs[i]["H"]
        # Get ice velocities for the reference dataset
        Vx_ref::Matrix{Float64} = gdir_refs[i]["Vx"]
        Vy_ref::Matrix{Float64} = gdir_refs[i]["Vy"]
        # Get ice thickness from the UDE predictions
        H::Matrix{Float64} = gdir_preds[i]["H"]
        # Get ice velocities prediction from the UDE
        V̄x_pred::Matrix{Float64} = gdir_preds[i]["Vx"]
        V̄y_pred::Matrix{Float64} = gdir_preds[i]["Vy"]

        # TODO: make loss function part standalone (to be re-used)

        normHref = mean(H_ref.^2)^0.5
        normVref = mean(Vx_ref.^2 .+ Vy_ref.^2)^0.5
        # normV = mean(abs.(Vx_ref .+ Vy_ref))
        l_H_loc = Flux.Losses.mse(H, H_ref; agg=mean) 
        l_V_loc = Flux.Losses.mse(V̄x_pred, Vx_ref; agg=mean) + Flux.Losses.mse(V̄y_pred, Vy_ref; agg=mean)
        # l_H = normHref^(-2.0) * l_H_loc
        # l_V = normVref^(-2.0) * l_V_loc
        l_H = normHref^(-1) * l_H_loc
        l_V = normVref^(-1) * l_V_loc

        @assert (loss_type[] == "H" || loss_type[] == "V" || loss_type[] == "HV") "Invalid loss_type. Needs to be 'H', 'V' or 'HV'"
        if loss_type[] == "H"
            l_tot += l_H/length(gdir_refs)
        elseif loss_type[] == "V"
            l_tot += l_V/length(gdir_refs) 
        elseif loss_type[] == "HV"
            l_tot += (l_V + l_H)/length(gdir_refs)
        end
    end

    return l_tot, UA_f[1]
end

"""
    predict_iceflow(θ, UA_f, gdirs_climate, context_batches, UDE_settings)

Makes a prediction of glacier evolution with the UDE for a given temperature series in different batches
"""
function predict_iceflow(θ, UA_f, gdirs_batches, context_batches, UDE_settings)
    # Train UDE in parallel
    H_V_pred = pmap((context, gdir) -> batch_iceflow_UDE(θ, UA_f, context, gdir, UDE_settings), context_batches, gdirs_batches)
    return H_V_pred
end

"""
    predict_iceflow_inplace(θs, UA_f, gdirs_climate_batches, tspan)

Makes a prediction of glacier evolution with the UDE in-place for a given temperature series in different batches
"""
function predict_iceflow_inplace(θ, UA_fs, gdirs_batches, tspans)
    # This is already inside a pmap (leave as map)
    preds = pmap((UA_f, gdir, tspan) -> batch_iceflow_UDE_inplace(θ, UA_f, gdir, tspan), UA_fs, gdirs_batches, tspans)
    # Gather information per gdir
    gdir_preds = get_gdir_refs(preds, gdirs_batches)
    return gdir_preds
end

"""
    batch_iceflow_UDE(θ, UA_f, context, longterm_temps_batch, UDE_settings, testmode) 

Solve the Shallow Ice Approximation iceflow UDE for a given temperature series batch
"""
function batch_iceflow_UDE(θ, UA_f, context, gdir, UDE_settings) 
    # context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)
    H::Matrix{Float64} = context[2]
    tspan::Tuple{Float64, Float64} = context[6]
    # This needs to be passed in a cleaner way
    mb_model = TI_model_1(DDF=5.0/1000.0, acc_factor=1.2/1000.0)

    # Initialize climate dataset
    _, step = @ignore define_callback_steps(tspan)
    S_coords = context[13]
    climate = @ignore init_climate(gdir, tspan, step, context[9], S_coords)
    # Callback  
    # Define stop times every one month
    tstops, step = define_callback_steps(tspan)
    stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, tstops) #closure
    function action!(integrator)
        if use_MB
            S::Matrix{Float64} = context[9]
            S_coords = context[13]
            MB = context[15]
            @ignore MB_timestep!(MB, mb_model, climate, S, S_coords, integrator.t[end], step)
            integrator.u .+= MB 
        end
        # Recompute A value
        A = context[14]
        testmode = context[16]
        testmode ? A[] = A_fake(mean(climate.longterm_temps))[1] : A[] = predict_A̅(UA_f, θ, [mean(climate.longterm_temps)])[1]
    end
    cb_MB = DiscreteCallback(stop_condition, action!)

    iceflow_UDE_batch(H, θ, t) = iceflow_NN(H, θ, t, UA_f, context) # closure
    iceflow_prob = ODEProblem(iceflow_UDE_batch, H, tspan, tstops=tstops, θ)
    iceflow_sol = solve(iceflow_prob, 
                        UDE_settings["solver"], 
                        callback=cb_MB,
                        tstops=tstops,
                        u0=H, 
                        p=θ,
                        sensealg=UDE_settings["sensealg"],
                        reltol=UDE_settings["reltol"], 
                        # saveat = get_interpolating_step(UDE_settings["interpolating_step"], tspan),
                        save_everystep=false,  
                        progress=false)
    # @show iceflow_sol.destats
    # Get ice velocities from the UDE predictions
    H_end::Matrix{Float64} = iceflow_sol.u[end]
    H_pred::Matrix{Float64} = ifelse.(H_end .< 0.0, 0.0, H_end)
    testmode = context[16]
    V̄x_pred::Matrix{Float64}, V̄y_pred::Matrix{Float64} = avg_surface_V(context, H_pred, mean(climate.longterm_temps), "UDE", θ, UA_f; 
                                                                        testmode=testmode) # Average velocity with average temperature
    rgi_id::String = @ignore gdir.rgi_id
    H_V_pred = (H_pred, V̄x_pred, V̄y_pred, rgi_id)

    @ignore GC.gc() # Run the garbage collector to tame the RAM

    return H_V_pred
end

function batch_iceflow_UDE_inplace(θ, UA_f, gdir, tspan; solver = RDPK3Sp35()) 
    context, H = build_PDE_context(gdir, nothing, tspan)
    # Callback  
    mb_model = TI_model_1(DDF=5.0/1000.0, acc_factor=1.2/1000.0)

    # Initialize climate dataset
    _, step = @ignore define_callback_steps(tspan)
    S = context[3]
    S_coords = context[32]
    climate = @ignore init_climate(gdir, tspan, step, S, S_coords)
    # Define stop times every one month
    tstops, step = define_callback_steps(tspan)
    stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, tstops) #closure
    function action!(integrator)
        if use_MB 
            # Compute mass balance
            MB = context[25]
            S = context[3]
            S_coords = context[32]
            MB_timestep!(MB, mb_model, climate, S, S_coords, integrator.t[end], step)
            integrator.u .+= MB
        end
        # Recompute A value
        A = context[1]
        A[] = predict_A̅(UA_f, θ, [mean(climate.longterm_temps)])[1]
    end
    cb_MB = DiscreteCallback(stop_condition, action!)

    iceflow_UDE_batch!(dH, H, context, t) = iceflow_NN!(dH, H, context, t, θ, UA_f) # closure
    preds = simulate_iceflow_PDE(H, context, climate, solver, tstops, cb_MB, θ, UA_f; du=iceflow_UDE_batch!, sim="UDE_inplace") # run the PDE with a NN input

    return preds
end

"""
    iceflow!(dH, H, context,t)

Runs a single time step of the iceflow PDE model in-place
"""
function iceflow!(dH, H, context, t)
    @timeit to "iceflow! PDE" begin
    # First, enforce values to be positive
    H[H.<0.0] .= H[H.<0.0] .* 0.0

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, context)
    end
end    

"""
    iceflow_NN!(dH, H, context, t, θ, UA_f)

Runs a single time step of the iceflow UDE model in-place using a NN
"""
function iceflow_NN!(dH, H, context, t, θ, UA_f)
    # First, enforce values to be positive
    H[H.<0.0] .= H[H.<0.0] .* 0.0
    # Unpack parameters
    #A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year 

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, context)
end    

"""
    iceflow_NN(H, θ, t, UA_f, context, temps)

Runs a single time step of the iceflow UDE model 
"""
function iceflow_NN(H, θ, t, UA_f, context)
    H_buf = Buffer(H)
    @views H_buf .= ifelse.(H.<0.0, 0.0, H) # prevent values from going negative
    H = copy(H_buf)

    dH = SIA(H, context) 
    return dH
end  

"""
    SIA!(dH, H, context)

Compute an in-place step of the Shallow Ice Approximation PDE in a forward model
"""
function SIA!(dH, H, context)
    # Retrieve parameters
    #[A], B, S, dSdx, dSdy, D, copy(temp_series[1]), dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, [current_year], nxy, Δxy
    @timeit to "Variable initialization PDE" begin
    A::Base.RefValue{Float64} = context[1]
    B::Matrix{Float64} = context[2]
    S::Matrix{Float64} = context[3]
    dSdx::Matrix{Float64} = context[4]
    dSdy::Matrix{Float64} = context[5]
    D::Matrix{Float64} = context[6]
    dSdx_edges::Matrix{Float64} = context[8]
    dSdy_edges::Matrix{Float64} = context[9]
    ∇S::Matrix{Float64} = context[10]
    Fx::Matrix{Float64} = context[11]
    Fy::Matrix{Float64} = context[12]
    Δx::Float64 = context[20][1]
    Δy::Float64 = context[20][2]
    Γ::Base.RefValue{Float64} = context[27]
    end

    # Update glacier surface altimetry
    S .= B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    @timeit to "S gradients PDE" begin
    diff_x!(dSdx, S, Δx)  
    diff_y!(dSdy, S, Δy) 
    end 
    @timeit to "∇S PDE" begin
    ∇S .= (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 
    end

    @timeit to "Diffusivity PDE" begin
    Γ[] = 2.0 * A[] * (ρ[] * g[])^n[] / (n[]+2) # 1 / m^3 s 
    D .= Γ .* avg(H).^(n[] + 2) .* ∇S
    end

    @timeit to "Flux PDE" begin
    # Compute flux components
    @views diff_x!(dSdx_edges, S[:,2:end - 1], Δx)
    @views diff_y!(dSdy_edges, S[2:end - 1,:], Δy)
    Fx .= .-avg_y(D) .* dSdx_edges
    Fy .= .-avg_x(D) .* dSdy_edges 
    end

    #  Flux divergence
    @timeit to "dH PDE" begin
    inn(dH) .= .-(diff_x(Fx) ./ Δx .+ diff_y(Fy) ./ Δy) 
    end
end

"""
    SIA(H, A, context)

Compute a step of the Shallow Ice Approximation UDE in a forward model. Allocates memory.
"""
function SIA(H, context)
    # Retrieve parameters
    # context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)
    B = context[1]
    Δx = context[5][1]
    Δy = context[5][2]
    A = context[14]

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) ./ Δx
    dSdy= diff_y(S) ./ Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 

    Γ = 2.0 * A[] * (ρ[] * g[])^n[] / (n[]+2) # 1 / m^3 s 
    D = Γ .* avg(H).^(n[] + 2) .* ∇S

    # Compute flux components
    @views dSdx_edges = diff_x(S[:,2:end - 1]) ./ Δx
    @views dSdy_edges = diff_y(S[2:end - 1,:]) ./ Δy
    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    @tullio dH[i,j] := -(diff_x(Fx)[pad(i-1,1,1),pad(j-1,1,1)] / Δx + diff_y(Fy)[pad(i-1,1,1),pad(j-1,1,1)] / Δy) 

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
    temp= T[years .== 2017]

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
function avg_surface_V(context, H, temp, sim, θ=nothing, UA_f=nothing; testmode=false)
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
function surface_V(H, B, Δx, Δy, temp, sim, A_noise, θ=nothing, UA_f=nothing; testmode=false)
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 
    
    @assert (sim == "UDE" || sim == "PDE" || sim == "UDE_inplace") "Wrong type of simulation. Needs to be 'UDE' , 'UDE_inplace' or 'PDE'."
    if (sim == "UDE" && !testmode) || (sim == "UDE_inplace" && !testmode)
        A = predict_A̅(UA_f, θ, [temp])[1] 
    elseif sim == "PDE" || testmode
        A = A_fake(temp, A_noise, noise)[1]
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

callback_plots_A = function (θ, l, UA_f, longterm_temps, A_noise, training_path, batch_size, n_gdirs) # callback function to observe training
    # Update training status
    update_training_state(l, batch_size, n_gdirs)

    if current_minibatches == 0.0
        avg_temps = [mean(longterm_temps[i]) for i in 1:length(longterm_temps)]
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

    return false
end
