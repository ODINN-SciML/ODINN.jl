
export generate_ref_dataset, train_iceflow_UDE, spinup
export predict_A̅, A_fake

function spinup(gdirs_climate, tspan; solver = Ralston(), random_MB=nothing)
    println("Spin up simulation for $(tspan[2]) years...\n")
    # Run batches in parallel
    gdirs = gdirs_climate[2]
    longterm_temps = gdirs_climate[3]
    A_noises = randn(rng_seed(), length(gdirs)) .* noise_A_magnitude
    if isnothing(random_MB)
        refs = @showprogress pmap((gdir, longterm_temp, A_noise) -> batch_iceflow_PDE(gdir, longterm_temp, A_noise, tspan, solver;run_spinup=true), gdirs, longterm_temps, A_noises)
    else
        refs = @showprogress pmap((gdir, longterm_temp, A_noise, MB) -> batch_iceflow_PDE(gdir, longterm_temp, A_noise, tspan, solver; run_spinup=true,random_MB=MB), gdirs, longterm_temps, A_noises, random_MB)
    end

    # Gather information per gdir
    gdir_refs = get_gdir_refs(refs, gdirs)

    spinup_path = joinpath(ODINN.root_dir, "data/spinup")
    if !isdir(spinup_path)
        mkdir(spinup_path)
    end
    jldsave(joinpath(ODINN.root_dir, "data/spinup/gdir_refs.jld2"); gdir_refs)
end

"""
    generate_ref_dataset(temp_series, H₀)

Generate reference dataset based on the iceflow PDE
"""
function generate_ref_dataset(gdirs_climate, tspan; solver = Ralston(), random_MB=nothing)
  
    # Perform reference simulation with forward model 
    println("Running forward PDE ice flow model...\n")
    # Run batches in parallel
    gdirs = gdirs_climate[2]
    longterm_temps = gdirs_climate[3]
    A_noises = randn(rng_seed(), length(gdirs)) .* noise_A_magnitude
    if isnothing(random_MB)
        refs = @showprogress pmap((gdir, longterm_temp, A_noise) -> batch_iceflow_PDE(gdir, longterm_temp, A_noise, tspan, solver;run_spinup=false), gdirs, longterm_temps, A_noises)
    else
        refs = @showprogress pmap((gdir, longterm_temp, A_noise, MB) -> batch_iceflow_PDE(gdir, longterm_temp, A_noise, tspan, solver; run_spinup=false,random_MB=MB), gdirs, longterm_temps, A_noises, random_MB)
    end

    # Gather information per gdir
    gdir_refs = get_gdir_refs(refs, gdirs)

    return gdir_refs
end

"""
    batch_iceflow_PDE(climate, gdir, context) 

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series batch
"""
function batch_iceflow_PDE(gdir, longterm_temp, A_noise, tspan, solver; run_spinup=false, random_MB=nothing) 
    println("Processing glacier: ", gdir.rgi_id)

    context, H = build_PDE_context(gdir, longterm_temp, A_noise, tspan; run_spinup=run_spinup, random_MB=random_MB)
    refs = simulate_iceflow_PDE(H, context, solver)

    return refs
end

"""
    simulate_iceflow_PDE(H, context, solver) 

Make forward simulation of the SIA PDE.
"""
function simulate_iceflow_PDE(H, context, solver)
    tspan = context.x[22]
    iceflow_prob = ODEProblem(iceflow!,H,tspan,context)
    iceflow_sol = solve(iceflow_prob, solver,
                    reltol=1e-6, save_everystep=false, 
                    progress=true, progress_steps = 10)
    # Compute average ice surface velocities for the simulated period
    H_ref = iceflow_sol.u[end]
    temps = context.x[7]
    B = context.x[2]
    V̄x_ref, V̄y_ref = avg_surface_V(context, H_ref, mean(temps), "PDE") # Average velocity with average temperature
    S = B .+ H_ref # Surface topography
    refs = Dict("Vx"=>V̄x_ref, "Vy"=>V̄y_ref, "H"=>H_ref, "S"=>S, "B"=>B)
    return refs
end

"""
     train_iceflow_UDE(gdirs_climate, tspan, train_settings, gdir_refs, θ_trained=[], UDE_settings=nothing, loss_history=[])

Train the Shallow Ice Approximation iceflow UDE. UDE_settings is optional, and requires a Dict specifiying the `reltol`, 
`sensealg` and `solver` for the UDE.  
"""
function train_iceflow_UDE(gdirs_climate, tspan, train_settings, gdir_refs, 
                           θ_trained=[], UDE_settings=nothing, loss_history=[]; random_MB=nothing)
    # Setup default parameters
    if length(θ_trained) == 0
        reset_epochs()
        global loss_history = []
    end
    # Fill default UDE_settings if not available 
    if isnothing(UDE_settings)
        UDE_settings = Dict("reltol"=>10f-6,
                        "solver"=>ROCK4(),
                        "sensealg"=>InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    end

    if random_MB == nothing
        ODINN.set_use_MB(false) 
    end

    optimizer = train_settings[1]
    epochs = train_settings[2]
    UA_f, θ = get_NN(θ_trained)
    gdirs = gdirs_climate[2]
    # Build context for all the batches before training
    println("Building context...")
    context_batches = map((gdir) -> build_UDE_context(gdir, tspan; run_spinup=false, random_MB=random_MB), gdirs)
    loss(θ) = loss_iceflow(θ, UA_f, gdirs_climate, context_batches, gdir_refs, UDE_settings) # closure

    println("Training iceflow UDE...")
    temps = gdirs_climate[3]
    A_noise = randn(rng_seed(), length(gdirs)).* noise_A_magnitude
    cb_plots(θ, l, UA_f) = callback_plots(θ, l, UA_f, temps, A_noise)
    # Setup optimization of the problem
    optf = OptimizationFunction((θ,_)->loss(θ), Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, θ)
    iceflow_trained = solve(optprob, optimizer, callback = cb_plots, maxiters = epochs)

    return iceflow_trained, UA_f, loss_history
end

callback_plots = function (θ, l, UA_f, temps, A_noise) # callback function to observe training
    println("Epoch #$current_epoch - Loss $(loss_type[]): ", l)

    avg_temps = [mean(temps[i]) for i in 1:length(temps)]
    p = sortperm(avg_temps)
    avg_temps = avg_temps[p]
    pred_A = predict_A̅(UA_f, θ, collect(-23.0f0:1.0f0:0.0f0)')
    pred_A = [pred_A...] # flatten
    true_A = A_fake(avg_temps, A_noise[p], noise)

    yticks = collect(0.0:2f-17:8f-17)

    Plots.scatter(avg_temps, true_A, label="True A", c=:lightsteelblue2)
    plot_epoch = Plots.plot!(-23f0:1f0:0f0, pred_A, label="Predicted A", 
                        xlabel="Long-term air temperature (°C)", yticks=yticks,
                        ylabel="A", ylims=(0.0f0,maxA[]), lw = 3, c=:dodgerblue4,
                        legend=:topleft)
    training_path = joinpath(root_plots,"training")
    if !isdir(joinpath(training_path,"png")) || !isdir(joinpath(training_path,"pdf"))
        mkpath(joinpath(training_path,"png"))
        mkpath(joinpath(training_path,"pdf"))
    end
    # Plots.savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.svg"))
    Plots.savefig(plot_epoch,joinpath(training_path,"png","epoch$current_epoch.png"))
    Plots.savefig(plot_epoch,joinpath(training_path,"pdf","epoch$current_epoch.pdf"))
    global current_epoch += 1
    push!(loss_history, l)

    plot_loss = Plots.plot(loss_history, label="", xlabel="Epoch", yaxis=:log10,
                ylabel="Loss (V)", lw = 3, c=:darkslategray3)
    Plots.savefig(plot_loss,joinpath(training_path,"png","loss$current_epoch.png"))
    Plots.savefig(plot_loss,joinpath(training_path,"pdf","loss$current_epoch.pdf"))
    
    false
end

"""
    loss_iceflow(θ, UA_f, gdirs_climate, context_batches, gdir_refs::Dict{String, Any}, UDE_settings)  

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow(θ, UA_f, gdirs_climate, context_batches, gdir_refs, UDE_settings) 
    H_V_preds = predict_iceflow(θ, UA_f, gdirs_climate, context_batches, UDE_settings)
    # Compute loss function for the full batch
    l_Vx, l_Vy, l_H = 0.0f0, 0.0f0, 0.0f0
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

        # Classic loss function with the full matrix
        if scale_loss[]
            normH  = mean(H_ref[H_ref .!= 0.0f0].^2.0f0)^0.5f0 #.+ ϵ
            normVx = mean(Vx_ref[Vx_ref .!= 0.0f0].^2.0f0)^0.5f0 #.+ ϵ
            normVy = mean(Vy_ref[Vy_ref .!= 0.0f0].^2.0f0)^0.5f0  #.+ ϵ
            l_H  += normH^(-2.0f0)  * Flux.Losses.mse(H[H_ref .!= 0.0f0], H_ref[H_ref.!= 0.0f0]; agg=mean) 
            l_Vx += normVx^(-2.0f0) * Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0f0], Vx_ref[Vx_ref.!= 0.0f0]; agg=mean)
            l_Vy += normVy^(-2.0f0) * Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0f0], Vy_ref[Vy_ref.!= 0.0f0]; agg=mean)
        else
            l_H += Flux.Losses.mse(H[H_ref .!= 0.0f0], H_ref[H_ref.!= 0.0f0]; agg=mean) 
            l_Vx += Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0f0], Vx_ref[Vx_ref.!= 0.0f0]; agg=mean)
            l_Vy += Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0f0], Vy_ref[Vy_ref.!= 0.0f0]; agg=mean)
        end
    end

    @assert (loss_type[] == "H" || loss_type[] == "V" || loss_type[] == "HV") "Invalid loss_type. Needs to be 'H', 'V' or 'HV'"
    if loss_type[] == "H"
        l_avg = l_H/length(gdir_refs)
    elseif loss_type[] == "V"
        l_avg = (l_Vx/length(gdir_refs) + l_Vy/length(gdir_refs))/2.0f0
    elseif loss_type[] == "HV"
        l_avg = (l_Vx/length(gdir_refs) + l_Vy/length(gdir_refs) + l_H/length(gdir_refs))/3.0f0
    end
    return l_avg, UA_f
end

"""
    predict_iceflow(θ, UA_f, gdirs_climate, context_batches, UDE_settings)

Makes a prediction of glacier evolution with the UDE for a given temperature series
"""
function predict_iceflow(θ, UA_f, gdirs_climate, context_batches, UDE_settings)

    # Train UDE in parallel
    # gdirs_climate = (dates, gdirs, longterm_temps, annual_temps)
    longterm_temps = gdirs_climate[3]
    H_V_pred = pmap((context, longterm_temps_batch) -> batch_iceflow_UDE(θ, UA_f, context, longterm_temps_batch, UDE_settings), context_batches, longterm_temps)
    return H_V_pred
end

"""
    batch_iceflow_UDE(θ, UA_f, context, longterm_temps_batch, UDE_settings) 

Solve the Shallow Ice Approximation iceflow UDE for a given temperature series batch
"""
function batch_iceflow_UDE(θ, UA_f, context, longterm_temps_batch, UDE_settings) 
    # Retrieve long-term temperature series
    H = context[3]
    tspan = context[6]
    longterm_temps_batch = Float32.(longterm_temps_batch)
    iceflow_UDE_batch(H, θ, t) = iceflow_NN(H, θ, t, UA_f, context, longterm_temps_batch) # closure
    iceflow_prob = ODEProblem(iceflow_UDE_batch,H,tspan,θ)
    iceflow_sol = solve(iceflow_prob, UDE_settings["solver"], u0=H, p=θ,
                    sensealg=UDE_settings["sensealg"],
                    reltol=UDE_settings["reltol"], save_everystep=false, 
                    progress=true, progress_steps = 100)
    # Get ice velocities from the UDE predictions
    H_pred = iceflow_sol.u[end]
    V̄x_pred, V̄y_pred = avg_surface_V(context, H_pred, mean(longterm_temps_batch), "UDE", θ, UA_f) # Average velocity with average temperature
    H_V_pred = (H_pred, V̄x_pred, V̄y_pred)
    return H_V_pred
end

"""
    iceflow!(dH, H, context,t)

Runs a single time step of the iceflow PDE model in-place
"""
function iceflow!(dH, H, context,t)
    # First, enforce values to be positive
    H[H.<0.0f0] .= H[H.<0.0f0] .* 0.0f0
    # Then, clip values if they get too high due to solver instabilities
    H₀ = context.x[21]
    H[H.>(1.5f0 * maximum(H₀))] .= 1.5f0 * maximum(H₀)
    # Unpack parameters
    #A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year 
    current_year = context.x[18]
    A = context.x[1]
    t₁ = context.x[22][end]
    B = context.x[2]
    H_y = context.x[21]
    A_noise = context.x[23]

    # Get current year for MB and ELA
    year = floor(Int, t) + 1
    if year != current_year && year <= t₁ 
        temp = context.x[7][year]
        A .= A_fake(temp, A_noise, noise)
        current_year .= year
        # println("current_year: ", current_year[])

        if use_MB[]
            H_y .= H
            compute_MB_matrix!(context, B, H_y, year)
        end
    end

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, context)
    # @show maximum(H)
    # @show minimum(H)
end    

"""
    iceflow_NN(H, θ, t, UA_f, context, temps)

Runs a single time step of the iceflow UDE model 
"""
function iceflow_NN(H, θ, t, UA_f, context, temps)
    t₁ = context[6][end]
    H₀ = context[2]
    H_buf = Buffer(H)
    @views H_buf .= ifelse.(H.<0.0f0, 0.0f0, H) # prevent values from going negative
    @views H_buf .= ifelse.(H.>(1.5f0 * maximum(H₀)), 1.5f0 * maximum(H₀), H) # prevent values from becoming too large
    H = copy(H_buf)
    B = context[1]
    S = B .+ H
    year = floor(Int, t) + 1

    # Define the mass balance as line between minimum and maximum surface
    if use_MB[]
        MB = compute_MB_matrix(context, S, H, year)
    else
        MB = 0.0f0
    end

    if year <= t₁
        temp = temps[year]
    else
        temp = temps[year-1]
    end
    A = predict_A̅(UA_f, θ, [temp])
    
    # Compute the Shallow Ice Approximation in a staggered grid
    dH = SIA(H, A, context) .+ MB

    # years = collect(1:t₁)
    # if any(isapprox.(t,years;atol=1e-4))
    #     rgi_id = context[8]
    #     println("$rgi_id - t: ", t, " - dH: ", maximum(dH))
    # end
    
    return dH
end  

"""
    SIA!(dH, H, context)

Compute an in-place step of the Shallow Ice Approximation PDE in a forward model
"""
function SIA!(dH, H, context)
    # Retrieve parameters
    #[A], B, S, dSdx, dSdy, D, copy(temp_series[1]), dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, [current_year], nxy, Δxy
    A = context.x[1]
    B = context.x[2]
    S = context.x[3]
    dSdx = context.x[4]
    dSdy = context.x[5]
    D = context.x[6]
    dSdx_edges = context.x[8]
    dSdy_edges = context.x[9]
    ∇S = context.x[10]
    Fx = context.x[11]
    Fy = context.x[12]
    Δx = context.x[20][1]
    Δy = context.x[20][2]
    MB = context.x[25]
    Γ = context.x[27]

    # Update glacier surface altimetry
    S .= B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx .= diff_x(S) ./ Δx
    dSdy .= diff_y(S) ./ Δy
    ∇S .= (avg_y(dSdx).^2.0f0 .+ avg_x(dSdy).^2.0f0).^((n[] - 1.0f0)/2.0f0) 

    Γ .= 2.0f0 * A * (ρ[] * g[])^n[] / (n[]+2.0f0) # 1 / m^3 s 
    D .= Γ .* avg(H).^(n[] + 2.0f0) .* ∇S

    # Compute flux components
    @views dSdx_edges .= diff_x(S[:,2:end - 1]) ./ Δx
    @views dSdy_edges .= diff_y(S[2:end - 1,:]) ./ Δy
    Fx .= .-avg_y(D) .* dSdx_edges
    Fy .= .-avg_x(D) .* dSdy_edges 

    # println("MB: ", minimum(MB[]), " - ", maximum(MB[]))
    # println("dH: ", minimum(.-(diff_x(Fx) ./ Δx .+ diff_y(Fy) ./ Δy)), " - ", maximum(.-(diff_x(Fx) ./ Δx .+ diff_y(Fy) ./ Δy)))

    #  Flux divergence
    inn(dH) .= .-(diff_x(Fx) ./ Δx .+ diff_y(Fy) ./ Δy) .+ MB 
end

"""
    SIA(H, A, context)

Compute a step of the Shallow Ice Approximation UDE in a forward model. Allocates memory.
"""
function SIA(H, A, context)
    # Retrieve parameters
    # context = (B, H₀, H, nxy, Δxy)
    B = context[1]
    Δx = context[5][1]
    Δy = context[5][2]

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) ./ Δx
    dSdy = diff_y(S) ./ Δy
    ∇S = (avg_y(dSdx).^2.0f0 .+ avg_x(dSdy).^2.0f0).^((n[] - 1.0f0)/2.0f0) 


    Γ = 2.0f0 * A * (ρ[] * g[])^n[] / (n[]+2.0f0) # 1 / m^3 s 
    D = Γ .* avg(H).^(n[] + 2.0f0) .* ∇S

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
    avg_surface_V(H, B, temp)

Computes the average ice velocity for a given input temperature
"""
function avg_surface_V(context, H, temp, sim, θ=[], UA_f=[])
    # context = (B, H₀, H, nxy, Δxy)
    B, H₀, Δx, Δy, A_noise = retrieve_context(context)

    # Update glacier surface altimetry
    S = B .+ (H₀ .+ H)./2.0f0 # Use average ice thickness for the simulated period

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2.0f0 .+ avg_x(dSdy).^2.0f0).^((n[] - 1.0f0)/2.0f0) 
    
    @assert (sim == "UDE" || sim == "PDE") "Wrong type of simulation. Needs to be 'UDE' or 'PDE'."
    if sim == "UDE"
        A = predict_A̅(UA_f, θ, [temp]) 
    elseif sim == "PDE"
        A = A_fake(temp, A_noise, noise)
    end
    Γꜛ = 2.0f0 * A * (ρ[] * g[])^n[] / (n[]+1.0f0) # surface stress (not average)  # 1 / m^3 s 
    D = Γꜛ .* avg(H).^(n[] + 1.0f0) .* ∇S
    
    # Compute averaged surface velocities
    Vx = - D .* avg_y(dSdx)
    Vy = - D .* avg_x(dSdy)

    return Vx, Vy
        
end

# Polynomial fit for Cuffey and Paterson data 
A_f = fit(A_values[1,:], A_values[2,:]) # degree = length(xs) - 1

"""
    A_fake(temp, noise=false)

Fake law establishing a theoretical relationship between ice viscosity (A) and long-term air temperature.
"""
function A_fake(temp, A_noise=nothing, noise=false)
    # A = @. minA + (maxA - minA) * ((temp-minT)/(maxT-minT) )^2
    A = A_f.(temp) # polynomial fit
    if noise[]
        A = abs.(A .+ A_noise)
    end
    return A
end

"""
    predict_A̅(UA_f, θ, temp)

Predicts the value of A with a neural network based on the long-term air temperature.
"""
function predict_A̅(UA_f, θ, temp)
    UA = UA_f(θ)
    return UA(temp) .* 1f-17
end

"""
    get_initial_geometry(glacier_gd)

Retrieves the initial glacier geometry (bedrock + ice thickness) for a glacier.
"""
function get_initial_geometry(gdir, run_spinup, smoothing=true)
    # Load glacier gridded data
    glacier_gd = xr.open_dataset(gdir.get_filepath("gridded_data"))
    if run_spinup || !use_spinup[]
        # Retrieve initial conditions from OGGM
        H₀ = Float32.(glacier_gd.consensus_ice_thickness.data) # initial ice thickness conditions for forward model
        fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
        if smoothing 
            smooth!(H₀)  # Smooth initial ice thickness to help the solver
        end

        # Create path for spinup simulation results
        gdir_path =  dirname(gdir.get_filepath("dem"))
        if !isdir(gdir_path)
            mkdir(gdir_path)
        end
    else
        # Retrieve initial state from previous spinup simulation
        gdir_refs = load(joinpath(ODINN.root_dir, "data/gdir_refs.jld2"))["gdir_refs"]
        H₀ = similar(gdir_refs[1]["H"])
        found = false
        for i in 1:length(gdir_refs)
            if gdir_refs[i]["RGI_ID"] == gdir.rgi_id
                H₀ = gdir_refs[i]["H"]
                found = true
                break
            end
        end
        @assert found == true "Spin up glacier simulation not found for $(gdir.rgi_id)."

    end

    H = deepcopy(H₀)
    B = glacier_gd.topo.data .- H₀ # bedrock
    nx = glacier_gd.y.size # glacier extent
    ny = glacier_gd.x.size # really weird, but this is inversed 
    Δx = Float32(abs(gdir.grid.dx))
    Δy = Float32(abs(gdir.grid.dy))

    return H₀, H, B, (nx,ny), (Δx,Δy)

end

function build_PDE_context(gdir, longterm_temp, A_noise, tspan; run_spinup=false, random_MB=nothing)
    # Determine initial geometry conditions
    H₀, H, B, nxy, Δxy = get_initial_geometry(gdir, run_spinup)
    rgi_id = gdir.rgi_id
    # Initialize all matrices for the solver
    nx, ny = nxy
    S, dSdx, dSdy = zeros(Float32,nx,ny),zeros(Float32,nx-1,ny),zeros(Float32,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1),zeros(Float32,nx-1,ny-1)
    D, Fx, Fy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1)
    V, Vx, Vy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1)
    MB = zeros(Float32,nx-2,ny-2)
    A = [2f-16]
    α = [0.0f0]                      # Weertman-type basal sliding (Weertman, 1964, 1972). 1 -> sliding / 0 -> no sliding
    C = [15f-14]    
    Γ = [0.0f0]
    maxS, minS = [0.0f0], [0.0f0]     
    
    # Gather simulation parameters
    current_year = [0.0f0] 
    if isnothing(random_MB)
        random_MB = zeros(Float32,Int(tspan[2]))
    end
    context = ArrayPartition(A, B, S, dSdx, dSdy, D, longterm_temp, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, 
                            current_year, nxy, Δxy, H₀, tspan, A_noise, random_MB, MB, rgi_id, Γ, maxS, minS)
    return context, H
end

function build_UDE_context(gdir, tspan; run_spinup=false, random_MB=nothing)
    H₀, H, B, nxy, Δxy = get_initial_geometry(gdir, run_spinup)
    rgi_id = gdir.rgi_id

    # Tuple with all the temp series
    context = (B, H₀, H, nxy, Δxy, tspan, random_MB, rgi_id)

    return context
end

"""
    retrieve_context(context::Tuple)

Retrieves context variables for computing the surface velocities of the UDE.
"""
function retrieve_context(context::Tuple)
    B = context[1]
    H₀ = context[2]
    Δx = context[5][1]
    Δy = context[5][2]
    return B, H₀, Δx, Δy, nothing
end

"""
    retrieve_context(context::ArrayPartition)

Retrieves context variables for computing the surface velocities of the PDE.
"""
function retrieve_context(context::ArrayPartition)
    B = context.x[2]
    H₀ = context.x[21]
    Δx = context.x[20][1]
    Δy = context.x[20][2]
    A_noise = context.x[23]
    return B, H₀, Δx, Δy, A_noise
end

"""
    get_NN()

Generates a neural network.
"""
function get_NN(θ_trained)
    UA = Chain(
        Dense(1,3, x->softplus.(x)),
        Dense(3,10, x->softplus.(x)),
        Dense(10,3, x->softplus.(x)),
        Dense(3,1, sigmoid_A)
    )
    # See if parameters need to be retrained or not
    θ, UA_f = Flux.destructure(UA)
    if !isempty(θ_trained)
        θ = θ_trained
    end
    return UA_f, θ
end

function sigmoid_A(x) 
    minA_out = 8.0f-3 # /!\     # these depend on predict_A̅, so careful when changing them!
    maxA_out = 8.0f0
    return minA_out + (maxA_out - minA_out) / ( 1.0f0 + exp(-x) )
end