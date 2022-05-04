
export generate_ref_dataset, train_iceflow_UDE
export predict_A̅, A_fake

"""
    generate_ref_dataset(temp_series, H₀)

Generate reference dataset based on the iceflow PDE
"""
function generate_ref_dataset(gdirs_climate, tspan, solver = Ralston())
  
    # Perform reference simulation with forward model 
    println("Running forward PDE ice flow model...\n")
    # Run batches in parallel
    gdirs = gdirs_climate[2]
    longterm_temps = gdirs_climate[3]
    refs = @showprogress pmap((gdir, longterm_temp) -> batch_iceflow_PDE(gdir, longterm_temp, tspan, solver), gdirs, longterm_temps)

    # Split into different vectors
    H_refs, V̄x_refs, V̄y_refs = [],[],[]
    for ref in refs
        push!(H_refs, ref["H"])
        push!(V̄x_refs, ref["Vx"])
        push!(V̄y_refs, ref["Vy"])
    end

    return H_refs, V̄x_refs, V̄y_refs
end

"""
    batch_iceflow_PDE(climate, gdir, context) 

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series batch
"""
function batch_iceflow_PDE(gdir, longterm_temp, tspan, solver) 
    println("Processing glacier: ", gdir.rgi_id)

    context, H = build_PDE_context(gdir, longterm_temp, tspan)
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
    V̄x_ref, V̄y_ref = avg_surface_V(context, H_ref, mean(temps), "PDE") # Average velocity with average temperature
    refs = Dict("Vx"=>V̄x_ref, "Vy"=>V̄y_ref, "H"=>H_ref)
    return refs
end

"""
    train_iceflow_UDE(H₀, UA, θ, train_settings, PDE_refs, temp_series)

Train the Shallow Ice Approximation iceflow UDE
"""
function train_iceflow_UDE(gdirs_climate, tspan, train_settings, PDE_refs, θ_trained=[], solver = ROCK4(), loss_history=[])
    if length(θ_trained) == 0
        global current_epoch = 1 # reset epoch count
    end
    optimizer = train_settings[1]
    epochs = train_settings[2]
    UA, θ = get_NN(θ_trained)
    gdirs = gdirs_climate[2]
    H_refs = PDE_refs["H_refs"]
    Vx_refs = PDE_refs["V̄x_refs"]
    Vy_refs = PDE_refs["V̄y_refs"]
    # Build context for all the batches before training
    println("Building context...")
    context_batches = pmap(gdir -> build_UDE_context(gdir, tspan), gdirs)
    loss(θ) = loss_iceflow(θ, UA, gdirs_climate, context_batches, PDE_refs, solver) # closure

    println("Training iceflow UDE...")
    iceflow_trained = DiffEqFlux.sciml_train(loss, θ, optimizer, cb=callback, maxiters = epochs)

    return iceflow_trained, UA
end

"""
    invert_iceflow(gdirs, train_settings, θ_trained=[], loss_history=[])

Performs an inversion on the SIA to train a NN on the ice flow law
"""
function train_iceflow_inversion(train_settings, θ_trained=[], loss_history=[])
    # Download Glathida dataset
    gtd_file = utils.file_downloader("https://cluster.klima.uni-bremen.de/~oggm/glathida/glathida-v3.1.0/data/TTT_per_rgi_id.h5")
    # Process file with Dataframes.jl instead of pandas

    # Initialize gdirs with ice thickness data
    gdirs = init_gdirs(rgi_ids, force=false)
    # Process climate data for glaciers
    gdirs_climate = get_gdirs_with_climate(gdirs, tspan, overwrite=false, plot=false)
    
    # Perform inversion with the given gdirs and climate data
    invert_iceflow(gdirs_climate, train_settings, θ_trained, loss_history)

end

"""
    invert_iceflow(gdirs, train_settings, θ_trained=[], loss_history=[])

Performs an inversion on the SIA to train a NN on the ice flow law
"""
function invert_iceflow(gdirs_climate, train_settings, θ_trained, loss_history)
    if length(θ_trained) == 0
        global current_epoch = 1 # reset epoch count
    end
    optimizer = train_settings[1]
    epochs = train_settings[2]
    UA, θ = get_NN(θ_trained)
    gdirs = gdirs_climate[2]

    # Build context for all the batches before training
    println("Building context...")
    context_batches = pmap(gdir -> build_UDE_context(gdir, tspan), gdirs)
    loss(θ) = loss_iceflow_inversion(θ, UA, gdirs_climate, context_batches) # closure
    # Do Flux.train here

end


"""
    loss_iceflow(θ, context, UA, PDE_refs::Dict{String, Any}, temp_series) 

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow_inversion(θ, UA, gdirs_climate, context_batches)
    
    V̄x_preds, V̄y_preds = perform_iceflow_inversion(θ, UA, gdirs_climate, context_batches)

    # Compute loss function for the full batch
    l_Vx, l_Vy = 0.0, 0.0
    for i in 1:length(H_V_preds)

        # Get ice velocities from ITS_LIVE or Millan et al. (2022)
        V̄x_ref = context_batches[i][4][1]
        V̄y_ref = context_batches[i][4][2]
        # Get ice velocities prediction from the UDE
        V̄x_pred = V̄x_preds[i]
        V̄y_pred = V̄y_preds[i]

        if scale_loss
            normVx = Vx_ref[Vx_ref .!= 0.0] .+ ϵ
            normVy = Vy_ref[Vy_ref .!= 0.0] .+ ϵ
            l_Vx += Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0] ./normVx, Vx_ref[Vx_ref.!= 0.0] ./normVx; agg=mean)
            l_Vy += Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0] ./normVy, Vy_ref[Vy_ref.!= 0.0] ./normVy; agg=mean)
        else
            l_Vx += Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0], Vx_ref[Vx_ref.!= 0.0]; agg=mean)
            l_Vy += Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0], Vy_ref[Vy_ref.!= 0.0]; agg=mean)
        end
    end
    

end

callback = function (θ, l, UA) # callback function to observe training
    println("Epoch #$current_epoch - Loss $loss_type: ", l)

    pred_A = predict_A̅(UA, θ, collect(-20.0:0.0)')
    pred_A = [pred_A...] # flatten
    true_A = A_fake(-20.0:0.0, noise)

    Plots.scatter(-20.0:0.0, true_A, label="True A")
    plot_epoch = Plots.plot!(-20.0:0.0, pred_A, label="Predicted A", 
                        xlabel="Long-term air temperature (°C)",
                        ylabel="A", ylims=(minA,maxA),
                        legend=:topleft)
    Plots.savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.png"))
    global current_epoch += 1
    push!(loss_history, l)

    false
end

"""
    loss_iceflow(θ, context, UA, PDE_refs::Dict{String, Any}, temp_series) 

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow(θ, UA, gdirs_climate, context_batches, PDE_refs::Dict{String, Any}, solver) 
    H_V_preds = predict_iceflow(θ, UA, gdirs_climate, context_batches, solver)

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

        if scale_loss
            normH = H_ref[H_ref .!= 0.0] .+ ϵ
            normVx = Vx_ref[Vx_ref .!= 0.0] .+ ϵ
            normVy = Vy_ref[Vy_ref .!= 0.0] .+ ϵ
            l_H += Flux.Losses.mse(H[H_ref .!= 0.0] ./normH, H_ref[H_ref.!= 0.0] ./normH; agg=mean) 
            l_Vx += Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0] ./normVx, Vx_ref[Vx_ref.!= 0.0] ./normVx; agg=mean)
            l_Vy += Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0] ./normVy, Vy_ref[Vy_ref.!= 0.0] ./normVy; agg=mean)
        else
            l_H += Flux.Losses.mse(H[H_ref .!= 0.0], H_ref[H_ref.!= 0.0]; agg=mean) 
            l_Vx += Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0], Vx_ref[Vx_ref.!= 0.0]; agg=mean)
            l_Vy += Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0], Vy_ref[Vy_ref.!= 0.0]; agg=mean)
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
    return l_avg, UA
end

"""
    predict_iceflow(θ, UA, context, temp_series) 

Makes a prediction of glacier evolution with the UDE for a given temperature series in different batches
"""
function predict_iceflow(θ, UA, gdirs_climate, context_batches, solver)

    # Train UDE in parallel
    # gdirs_climate = (dates, gdirs, longterm_temps, annual_temps)
    longterm_temps = gdirs_climate[3]
    H_V_pred = pmap((context, longterm_temps_batch) -> batch_iceflow_UDE(θ, UA, context, longterm_temps_batch, solver), context_batches, longterm_temps)
    return H_V_pred
end


"""
    perform_iceflow_inversion(θ, UA, gdirs_climate, context_batches)

Performs an inversion of the iceflow law with a UDE in different batches
"""
function perform_iceflow_inversion(θ, UA, gdirs_climate, context_batches)
    longterm_temps = gdirs_climate[3]
    V̄x_pred, V̄y_pred = pmap((context, longterm_temps_batch) -> avg_surface_V(context, H_pred, mean(longterm_temps_batch), "UDE", θ, UA), context_batches, longterm_temps)
    return V̄x_pred, V̄y_pred
end

"""
    batch_iceflow_UDE(θ, H, climate, context) 

Solve the Shallow Ice Approximation iceflow UDE for a given temperature series batch
"""
function batch_iceflow_UDE(θ, UA, context, longterm_temps_batch, solver) 
    # Retrieve long-term temperature series
    # context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)
    H = context[3]
    tspan = context[7]
    iceflow_UDE_batch(H, θ, t) = iceflow_NN(H, θ, t, UA, context, longterm_temps_batch) # closure
    iceflow_prob = ODEProblem(iceflow_UDE_batch,H,tspan,θ)
    iceflow_sol = solve(iceflow_prob, solver, u0=H, p=θ,
                    reltol=1e-6, save_everystep=false, 
                    progress=true, progress_steps = 10)
    # Get ice velocities from the UDE predictions
    H_pred = iceflow_sol.u[end]
    V̄x_pred, V̄y_pred = avg_surface_V(context, H_pred, mean(longterm_temps_batch), "UDE", θ, UA) # Average velocity with average temperature
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
    
    # Get current year for MB and ELA
    year = floor(Int, t) + 1
    if year != current_year[] && year <= t₁
        temp = Ref{Float64}(context.x[7][year])
        A[] .= A_fake(temp[], noise)
        current_year[] .= year
    end

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, context)
end    

"""
    iceflow_NN(H, θ, t, context, temps, UA)

Runs a single time step of the iceflow UDE model 
"""
function iceflow_NN(H, θ, t, UA, context, temps)

    year = floor(Int, t) + 1
    # context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)
    t₁ = context[7][end]
    if year <= t₁
        temp = temps[year]
    else
        temp = temps[year-1]
    end
    A = predict_A̅(UA, θ, [temp]) # FastChain prediction requires explicit parameters

    # Compute the Shallow Ice Approximation in a staggered grid
    return SIA(H, A, context)
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
    # context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)
    B = context[1]
    Δx = context[6][1]
    Δy = context[6][2]

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

Computes the average ice surface velocity for a given input temperature
"""
function avg_surface_V(context, H, temp, sim, θ=[], UA=[])
    # context = (B, H₀, H, nxy, Δxy)
    B, H₀, Δx, Δy = retrieve_context(context)

    # We compute the initial and final surface velocity and average them
    # TODO: Add more H datapoints to better interpolate this
    Vx₀, Vy₀ = surface_V(H₀, B, Δx, Δy, temp, sim, θ, UA)
    Vxₜ, Vyₜ = surface_V(H, B, Δx, Δy, temp, sim, θ, UA)
    Vx = (Vx₀ .+ Vxₜ)./2.0
    Vy = (Vy₀ .+ Vyₜ)./2.0

    return Vx, Vy
        
end

"""
    avg_surface_V(H, B, temp)

Computes the ice surface velocity for a given input temperature
"""
function surface_V(H, B, Δx, Δy, temp, sim, θ=[], UA=[])
    # Update glacier surface altimetry
    S = B .+ (H)./2.0 # Use average ice thickness for the simulated period

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2.0 .+ avg_x(dSdy).^2.0).^((n - 1.0)/2.0) 
    
    @assert (sim == "UDE" || sim == "PDE" || sim == "inversion") "Wrong type of simulation. Needs to be 'UDE', 'PDE' or 'inversion'."
    if sim == "UDE" || sim == "inversion"
        A = predict_A̅(UA, θ, [temp]) # FastChain prediction requires explicit parameters
    elseif sim == "PDE"
        A = A_fake(temp, noise)
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
function A_fake(temp, noise=false)
    # A = @. minA + (maxA - minA) * ((temp-minT)/(maxT-minT) )^2
    A = A_f.(temp) # polynomial fit
    if noise
        A = A .+ randn(rng_seed(), length(temp)).*6e-18
    end
    return A
end

"""
    predict_A̅(UA, θ, temp)

Predicts the value of A with a neural network based on the long-term air temperature.
"""
function predict_A̅(UA, θ, temp)
    return UA(temp, θ) .* 1e-17
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
    get_glacier_data(glacier_gd)

Retrieves the initial glacier geometry (bedrock + ice thickness) for a glacier with other necessary data (e.g. grid size and ice surface velocities).
"""
function get_glacier_data(gdir, smoothing=true)
    # Load glacier gridded data
    glacier_gd = xr.open_dataset(gdir.get_filepath("gridded_data"))
    H₀ = glacier_gd.consensus_ice_thickness.data # initial ice thickness conditions for forward model
    fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
    if smoothing 
        smooth!(H₀)  # Smooth initial ice thickness to help the solver
    end
    H = deepcopy(H₀)
    B = glacier_gd.topo.data .- H₀ # bedrock
    Vx_obs = glacier_gd.obs_icevel_x.data
    Vy_obs = glacier_gd.obs_icevel_x.data

    nx = glacier_gd.y.size # glacier extent
    ny = glacier_gd.x.size # really weird, but this is inversed 
    Δx = abs(gdir.grid.dx)
    Δy = abs(gdir.grid.dy)

    return H₀, H, B, (Vx_obs,Vy_obs), (nx,ny), (Δx,Δy)
end

function build_PDE_context(gdir, longterm_temp, tspan)
    # Determine initial geometry conditions
    H₀, H, B, Vxy_obs, nxy, Δxy = get_glacier_data(gdir)
    # Initialize all matrices for the solver
    nx, ny = nxy
    S, dSdx, dSdy = zeros(Float64,nx,ny),zeros(Float64,nx-1,ny),zeros(Float64,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1),zeros(Float64,nx-1,ny-1)
    D, Fx, Fy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1)
    V, Vx, Vy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1)
    A = 2e-16
    α = 0                       # Weertman-type basal sliding (Weertman, 1964, 1972). 1 -> sliding / 0 -> no sliding
    C = 15e-14                  # Sliding factor, between (0 - 25) [m⁸ N⁻³ a⁻¹]
    
    # Gather simulation parameters
    current_year = 0 
    context = ArrayPartition([A], B, S, dSdx, dSdy, D, longterm_temp, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, [current_year], nxy, Δxy, H₀, tspan)
    return context, H
end

function build_UDE_context(gdir, tspan)
    H₀, H, B, Vxy_obs, nxy, Δxy = get_glacier_data(gdir)

    # Tuple with all the temp series and H_refs
    context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)

    return context
end

"""
    retrieve_context(context::Tuple)

Retrieves context variables for computing the surface velocities of the UDE.
"""
function retrieve_context(context::Tuple)
    # context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)
    B = context[1]
    H₀ = context[2]
    Δx = context[6][1]
    Δy = context[6][2]
    return B, H₀, Δx, Δy
end

"""
    retrieve_context(context::ArrayPartition)

Retrieves context variables for computing the surface velocities of the PDE.
"""
function retrieve_context(context::ArrayPartition)
    # context = ArrayPartition([A], B, S, dSdx, dSdy, D, longterm_temp, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, [current_year], nxy, Δxy, H₀, tspan)
    B = context.x[2]
    H₀ = context.x[21]
    Δx = context.x[20][1]
    Δy = context.x[20][2]
    return B, H₀, Δx, Δy, nothing
end

"""
    get_NN()

Generates a neural network.
"""
function get_NN(θ_trained)
    UA = FastChain(
        FastDense(1,3, x->softplus.(x)),
        FastDense(3,10, x->softplus.(x)),
        FastDense(10,3, x->softplus.(x)),
        FastDense(3,1, sigmoid_A)
    )
    # See if parameters need to be retrained or not
    if isempty(θ_trained)
        θ = initial_params(UA)
    else
        θ = θ_trained
    end
    return UA, θ
end

function sigmoid_A(x) 
    minA_out = 8.5e-3 # /!\  # these depend on predict_A̅, so careful when changing them!
    maxA_out = 8.0
    return minA_out + (maxA_out - minA_out) / ( 1.0 + exp(-x) )
end