
export generate_ref_dataset, train_iceflow_UDE
export predict_A̅, A_fake

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
        refs = @showprogress pmap((gdir, longterm_temp, A_noise) -> batch_iceflow_PDE(gdir, longterm_temp, A_noise, tspan, solver), gdirs, longterm_temps, A_noises)
    else
        refs = @showprogress pmap((gdir, longterm_temp, A_noise, MB) -> batch_iceflow_PDE(gdir, longterm_temp, A_noise, tspan, solver; random_MB=MB), gdirs, longterm_temps, A_noises, random_MB)
    end

    # Split into different vectors
    H_refs, V̄x_refs, V̄y_refs, S_refs, B_refs = [],[],[],[],[]
    for ref in refs
        push!(H_refs, ref["H"])
        push!(V̄x_refs, ref["Vx"])
        push!(V̄y_refs, ref["Vy"])
        push!(S_refs, ref["S"])
        push!(B_refs, ref["B"])
    end

    return H_refs, V̄x_refs, V̄y_refs, S_refs, B_refs
end

"""
    batch_iceflow_PDE(climate, gdir, context) 

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series batch
"""
function batch_iceflow_PDE(gdir, longterm_temp, A_noise, tspan, solver; random_MB=nothing) 
    println("Processing glacier: ", gdir.rgi_id)

    context, H = build_PDE_context(gdir, longterm_temp, A_noise, tspan; random_MB=random_MB)
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
     train_iceflow_UDE(gdirs_climate, tspan, train_settings, PDE_refs, θ_trained=[], UDE_settings=nothing, loss_history=[])

Train the Shallow Ice Approximation iceflow UDE. UDE_settings is optional, and requires a Dict specifiying the `reltol`, 
`sensealg` and `solver` for the UDE.  
"""
function train_iceflow_UDE(gdirs_climate, tspan, train_settings, PDE_refs, 
                           θ_trained=[], UDE_settings=nothing, loss_history=[]; random_MB=nothing)
    # Setup default parameters
    if length(θ_trained) == 0
        global current_epoch = 1 # reset epoch count
        global loss_history = []
    end

    if isnothing(UDE_settings)
        UDE_settings = Dict("reltol"=>10e-6,
                        "solver"=>ROCK4(),
                        "sensealg"=>InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    end

    optimizer = train_settings[1]
    epochs = train_settings[2]
    UA_f, θ = get_NN(θ_trained)
    gdirs = gdirs_climate[2]
    H_refs = PDE_refs["H_refs"]
    Vx_refs = PDE_refs["V̄x_refs"]
    Vy_refs = PDE_refs["V̄y_refs"]
    # Build context for all the batches before training
    println("Building context...")
    context_batches = pmap((gdir, H_ref, Vx_ref, Vy_ref) -> build_UDE_context(gdir, H_ref, Vx_ref, Vy_ref, tspan; random_MB=random_MB), gdirs, H_refs, Vx_refs, Vy_refs)
    loss(θ) = loss_iceflow(θ, UA_f, gdirs_climate, context_batches, PDE_refs, UDE_settings) # closure

    println("Training iceflow UDE...")
    temps = gdirs_climate[3]
    A_noise = randn(rng_seed(), length(gdirs)).* noise_A_magnitude
    cb(θ, l, UA_f) = callback(θ, l, UA_f, temps, A_noise)

    optf = OptimizationFunction((θ,_)->loss(θ),Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, θ)
    iceflow_trained = solve(optprob, optimizer, callback = cb, maxiters = epochs)
    #iceflow_trained = DiffEqFlux.sciml_train(loss, θ, optimizer, cb=cb, maxiters = epochs)
    # iceflow_trained = DiffEqFlux.sciml_train(loss, θ, optimizer, cb=callback, maxiters = epochs)

    return iceflow_trained, UA_f
end

callback = function (θ, l, UA_f, temps, A_noise) # callback function to observe training
    println("Epoch #$current_epoch - Loss $loss_type: ", l)

    avg_temps = [mean(temps[i]) for i in 1:length(temps)]
    p = sortperm(avg_temps)
    avg_temps = avg_temps[p]
    pred_A = predict_A̅(UA_f, θ, collect(-23:1:0)')
    pred_A = [pred_A...] # flatten
    true_A = A_fake(avg_temps, A_noise[p], noise)

    Plots.scatter(avg_temps, true_A, label="True A", c=:lightsteelblue2)
    plot_epoch = Plots.plot!(-23:1:0, pred_A, label="Predicted A", 
                        xlabel="Long-term air temperature (°C)",
                        ylabel="A", ylims=(0.0,maxA), lw = 3, c=:dodgerblue4,
                        legend=:topleft)
    # Plots.savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.svg"))
    Plots.savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.png"))
    global current_epoch += 1
    push!(loss_history, l)

    plot_loss = Plots.plot(loss_history, xlabel="Epoch",
                ylabel="Loss (V)", lw = 3, c=:darkslategray3)
    Plots.savefig(plot_loss,joinpath(root_plots,"training","loss$current_epoch.png"))
    
    false
end

"""
    loss_iceflow(θ, UA_f, gdirs_climate, context_batches, PDE_refs::Dict{String, Any}, UDE_settings)  

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow(θ, UA_f, gdirs_climate, context_batches, PDE_refs::Dict{String, Any}, UDE_settings) 
    H_V_preds = predict_iceflow(θ, UA_f, gdirs_climate, context_batches, UDE_settings)
    # println("iceflow predicted")
    # Compute loss function for the full batch
    l_Vx, l_Vy, l_H = 0.0, 0.0, 0.0
    for i in 1:length(H_V_preds)

        # Get ice thickness from the reference dataset
        H_ref = PDE_refs["H_refs"][i]
        # Get ice velocities for the reference dataset
        Vx_ref = PDE_refs["V̄x_refs"][i]
        Vy_ref = PDE_refs["V̄y_refs"][i]
        # Get ice thickness from the UDE predictions
        H = H_V_preds[i][1]
        # Get ice velocities prediction from the UDE
        V̄x_pred = H_V_preds[i][2]
        V̄y_pred = H_V_preds[i][3]

        if random_sampling_loss
            # sample random indices for which V_ref is non-zero
            n_sample, n_counts = 50, 0
            nxy = length(H_ref[H_ref .!= 0.0])

            while n_counts < n_sample
                j = rand(1:nxy-2)
                H_ref_f = H_ref[H_ref .!= 0.0]
                Vx_ref_f = Vx_ref[Vx_ref .!= 0.0]
                Vy_ref_f = Vy_ref[Vy_ref .!= 0.0]
                H_f = H[H_ref .!= 0.0]
                V̄x_pred_f = V̄x_pred[Vx_ref .!= 0.0]
                V̄y_pred_f = V̄y_pred[Vy_ref .!= 0.0]

                if norm_loss
                    normH = H_ref_f[j]  .+ ϵ
                    normVx = Vx_ref_f[j] .+ ϵ
                    normVy = Vy_ref_f[j] .+ ϵ
                else
                    normH, normH, normVy = 1, 1, 1
                end
                l_H += Flux.Losses.mse(H_f[j]  ./normH, H_ref_f[j]  ./normH; agg=mean) 
                l_Vx += Flux.Losses.mse(V̄x_pred_f[j]  ./normVx, Vx_ref_f[j]  ./normVx; agg=mean)
                l_Vy += Flux.Losses.mse(V̄y_pred_f[j]  ./normVy, Vy_ref_f[j]  ./normVy; agg=mean)

                n_counts += 1
            end
            l_H = l_H / n_sample
            l_Vx = l_Vx / n_sample
            l_Vy = l_Vy / n_sample
        else
            # Classic loss function with the full matrix
            if scale_loss
                normH  = mean(H_ref[H_ref .!= 0.0].^2.0)^0.5 #.+ ϵ
                normVx = mean(Vx_ref[Vx_ref .!= 0.0].^2.0)^0.5 #.+ ϵ
                normVy = mean(Vy_ref[Vy_ref .!= 0.0].^2.0)^0.5  #.+ ϵ
                l_H  += normH^(-2.0)  * Flux.Losses.mse(H[H_ref .!= 0.0], H_ref[H_ref.!= 0.0]; agg=mean) 
                l_Vx += normVx^(-2.0) * Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0], Vx_ref[Vx_ref.!= 0.0]; agg=mean)
                l_Vy += normVy^(-2.0) * Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0], Vy_ref[Vy_ref.!= 0.0]; agg=mean)
            else
                l_H += Flux.Losses.mse(H[H_ref .!= 0.0], H_ref[H_ref.!= 0.0]; agg=mean) 
                l_Vx += Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0], Vx_ref[Vx_ref.!= 0.0]; agg=mean)
                l_Vy += Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0], Vy_ref[Vy_ref.!= 0.0]; agg=mean)
            end
        end
    end

    @assert (loss_type == "H" || loss_type == "V" || loss_type == "HV") "Invalid loss_type. Needs to be 'H', 'V' or 'HV'"
    if loss_type == "H"
        l_avg = l_H/length(PDE_refs["H_refs"])
    elseif loss_type == "V"
        l_avg = (l_Vx/length(PDE_refs["V̄x_refs"]) + l_Vy/length(PDE_refs["V̄y_refs"]))/2
    elseif loss_type == "HV"
        l_avg = (l_Vx/length(PDE_refs["V̄x_refs"]) + l_Vy/length(PDE_refs["V̄y_refs"]) + l_H/length(PDE_refs["H_refs"]))/3
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
    #H_V_pred = map((context, longterm_temps_batch) -> batch_iceflow_UDE(θ, UA_f, context, longterm_temps_batch, UDE_settings), context_batches, longterm_temps)
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
    # Unpack parameters
    #A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year 
    current_year = Ref(context.x[18])
    A = Ref(context.x[1])
    t₁ = context.x[22][end]
    B₀ = Ref(context.x[2])
    H₀ = Ref(context.x[21])
    A_noise = Ref(context.x[23])
    MB_series = Ref(context.x[24])
    MB = Ref(context.x[25])
    if !isnothing(MB_series[])
        # MB array has tuples with (RGI_ID, MB_max, MB_min)
        max_MB = Ref(MB_series[][2])
        min_MB = Ref(MB_series[][3]) 
    end
    
    # Get current year for MB and ELA
    year = floor(Int, t) + 1
    if year != current_year[] && year <= t₁ 
        temp = Ref{Float64}(context.x[7][year])
        A[] .= A_fake(temp[], A_noise[], noise)
        current_year[] .= year

        if !isnothing(MB_series)
            # Add mass balance based on gradient
            max_S = maximum(B₀[][H₀[] .!= 0.0] .+ H₀[][H₀[] .!= 0.0])
            min_S = minimum(B₀[][H₀[] .!= 0.0] .+ H₀[][H₀[] .!= 0.0])
            # Define the mass balance as line between minimum and maximum surface
            MB[] .= inn((min_MB[][year] .+ (B₀[] .+ H₀[] .- min_S) .* 
                        ((max_MB[][year] - min_MB[][year]) / (max_S - min_S))) .* Float64.(Matrix(H₀[].>0.0)))
        end
    end

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, context) .+ MB[]
end    

"""
    iceflow_NN(H, θ, t, UA_f, context, temps)

Runs a single time step of the iceflow UDE model 
"""
function iceflow_NN(H, θ, t, UA_f, context, temps)

    year = floor(Int, t) + 1
    t₁ = context[6][end]
    B₀ = context[1]
    H₀ = context[2]
    S₀ = B₀ .+ H₀
    #println("Maximum surface: ", maximum(S₀))

    #println("Mean slope, max, min: ", minimum(B₀ .+ H₀), mean(B₀ .+ H₀), maximum(B₀ .+ H₀))
    
    if year <= t₁
        temp = temps[year]
    else
        temp = temps[year-1]
    end
    A = predict_A̅(UA_f, θ, [temp]) 

    # Define the mass balance as line between minimum and maximum surface
    if !isnothing(context[7])
        max_MB = context[7][2][year]
        min_MB = context[7][3][year]
        max_S = maximum(S₀[H₀ .!= 0.0])
        min_S = minimum(S₀[H₀ .!= 0.0])
        MB = (min_MB .+ (S₀ .- min_S) .* ((max_MB - min_MB) / (max_S - min_S))) .* Float64.(Matrix(H₀.>0.0))
        # println("MB max: ", maximum(MB))
        # println("MB min: ", minimum(MB))
        # println("MB med: ", median(MB))
    else
        MB = 0.0
    end

    # Compute the Shallow Ice Approximation in a staggered grid
    dH = SIA(H, A, context) .+ MB
    
    # years = collect(1:t₁)
    # if any(isapprox.(t,years;atol=5e-2))
    #     println("t: ", t)
    #     # display(Plots.heatmap(MB))
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

    # Update glacier surface altimetry
    S .= B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx .= diff_x(S) ./ Δx
    dSdy .= diff_y(S) ./ Δy
    ∇S .= (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 

    Γ = 2.0 * A * (ρ * g)^n / (n+2.0) # 1 / m^3 s 
    D .= Γ .* avg(H).^(n + 2.0) .* ∇S

    # Compute flux components
    dSdx_edges .= diff_x(S[:,2:end - 1]) ./ Δx
    dSdy_edges .= diff_y(S[2:end - 1,:]) ./ Δy
    Fx .= .-avg_y(D) .* dSdx_edges
    Fy .= .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    inn(dH) .= .-(diff_x(Fx) ./ Δx .+ diff_y(Fy) ./ Δy) # MB to be added here 
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
    ∇S = (avg_y(dSdx).^2.0 .+ avg_x(dSdy).^2.0).^((n - 1.0)/2.0) 

    Γ = 2.0 * A * (ρ * g)^n / (n+2.0) # 1 / m^3 s 
    D = Γ .* avg(H).^(n + 2.0) .* ∇S

    # Compute flux components
    dSdx_edges = diff_x(S[:,2:end - 1]) ./ Δx
    dSdy_edges = diff_y(S[2:end - 1,:]) ./ Δy
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
    S = B .+ (H₀ .+ H)./2.0 # Use average ice thickness for the simulated period

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2.0 .+ avg_x(dSdy).^2.0).^((n - 1.0)/2.0) 
    
    @assert (sim == "UDE" || sim == "PDE") "Wrong type of simulation. Needs to be 'UDE' or 'PDE'."
    if sim == "UDE"
        A = predict_A̅(UA_f, θ, [temp]) 
    elseif sim == "PDE"
        A = A_fake(temp, A_noise, noise)
    end
    Γꜛ = 2.0 * A * (ρ * g)^n / (n+1.0) # surface stress (not average)  # 1 / m^3 s 
    D = Γꜛ .* avg(H).^(n + 1.0) .* ∇S
    
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
    if noise
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
    return UA(temp) .* 1e-17
end

# function fake_temp_series(t, means=Array{Float64}([0.0,-2.0,-3.0,-5.0,-10.0,-12.0,-14.0,-15.0,-20.0]))
#     temps, norm_temps, norm_temps_flat = [],[],[]
#     for mean in means
#         push!(temps, mean .+ rand(t).*1e-1) # static
#         append!(norm_temps_flat, mean .+ rand(t).*1e-1) # static
#     end

#     # Normalise temperature series
#     norm_temps_flat = Flux.normalise([norm_temps_flat...]) # requires splatting

#     # Re-create array of arrays 
#     for i in 1:t₁:length(norm_temps_flat)
#         push!(norm_temps, norm_temps_flat[i:i+(t₁-1)])
#     end

#     return temps, norm_temps
# end

"""
    get_initial_geometry(glacier_gd)

Retrieves the initial glacier geometry (bedrock + ice thickness) for a glacier.
"""
function get_initial_geometry(gdir, smoothing=true)
    # Load glacier gridded data
    glacier_gd = xr.open_dataset(gdir.get_filepath("gridded_data"))
    H₀ = glacier_gd.consensus_ice_thickness.data # initial ice thickness conditions for forward model
    fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
    if smoothing 
        smooth!(H₀)  # Smooth initial ice thickness to help the solver
    end
    H = deepcopy(H₀)
    B = glacier_gd.topo.data .- H₀ # bedrock

    nx = glacier_gd.y.size # glacier extent
    ny = glacier_gd.x.size # really weird, but this is inversed 
    Δx = abs(gdir.grid.dx)
    Δy = abs(gdir.grid.dy)

    return H₀, H, B, (nx,ny), (Δx,Δy)
end

function build_PDE_context(gdir, longterm_temp, A_noise, tspan; random_MB=nothing)
    # Determine initial geometry conditions
    H₀, H, B, nxy, Δxy = get_initial_geometry(gdir)
    # Initialize all matrices for the solver
    nx, ny = nxy
    S, dSdx, dSdy = zeros(Float64,nx,ny),zeros(Float64,nx-1,ny),zeros(Float64,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1),zeros(Float64,nx-1,ny-1)
    D, Fx, Fy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1)
    V, Vx, Vy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1)
    MB = zeros(Float64,nx-2,ny-2)
    A = 2e-16
    α = 0                       # Weertman-type basal sliding (Weertman, 1964, 1972). 1 -> sliding / 0 -> no sliding
    C = 15e-14                  # Sliding factor, between (0 - 25) [m⁸ N⁻³ a⁻¹]
    
    # Gather simulation parameters
    current_year = 0 
    context = ArrayPartition([A], B, S, dSdx, dSdy, D, longterm_temp, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, [current_year], nxy, Δxy, H₀, tspan, A_noise, random_MB, MB)
    return context, H
end

function build_UDE_context(gdir, H_ref, Vx_ref, Vy_ref, tspan; random_MB=nothing)
    H₀, H, B, nxy, Δxy = get_initial_geometry(gdir)

    # Tuple with all the temp series and H_refs
    context = (B, H₀, H, nxy, Δxy, tspan, random_MB)

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
    minA_out = 8.0e-3 # /!\     # these depend on predict_A̅, so careful when changing them!
    maxA_out = 8.0
    return minA_out + (maxA_out - minA_out) / ( 1.0 + exp(-x) )
end