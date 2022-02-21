
@everywhere include("utils.jl")

"""
    generate_ref_dataset(temp_series, H₀)

Generate reference dataset based on the iceflow PDE
"""
function generate_ref_dataset(temp_series, H₀)
    # Compute reference dataset in parallel
    H = deepcopy(H₀)
    
    # Initialize all matrices for the solver
    S, dSdx, dSdy = zeros(Float64,nx,ny),zeros(Float64,nx-1,ny),zeros(Float64,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1),zeros(Float64,nx-1,ny-1)
    D, dH, Fx, Fy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-2,ny-2),zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1)
    V, Vx, Vy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1)
    A = 2e-16
    α = 0                       # Weertman-type basal sliding (Weertman, 1964, 1972). 1 -> sliding / 0 -> no sliding
    C = 15e-14                  # Sliding factor, between (0 - 25) [m⁸ N⁻³ a⁻¹]
    
    # Gather simulation parameters
    current_year = 0
    context = ArrayPartition([A], B, S, dSdx, dSdy, D, copy(temp_series[5]), dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, [current_year])

    # Perform reference simulation with forward model 
    println("Running forward PDE ice flow model...\n")
    # Train batches in parallel
    # Fetch reference simulated ice thickness
    H_refs  = @showprogress pmap(temps -> prob_iceflow_PDE(H, temps, context), temp_series)

    # Compute average ice surface velocities for the simulated period
    V̄x_refs, V̄y_refs = [],[]
    for (H_ref, temps) in zip(H_refs, temp_series) 
        V̄x_ref, V̄y_ref = avg_surface_V(H_ref, B, mean(temps), "PDE") # Average velocity with average temperature
        push!(V̄x_refs, V̄x_ref)
        push!(V̄y_refs, V̄y_ref)
    end

    return H_refs, V̄x_refs, V̄y_refs
end

@everywhere begin
"""
    prob_iceflow_PDE(H, temps, context)

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series
"""
function prob_iceflow_PDE(H, temps, context) 
        
    println("Processing temp series ≈ ", mean(temps))
    context.x[7] .= temps # We set the temp_series for the ith trajectory

    iceflow_prob = ODEProblem(iceflow!,H,(0.0,t₁),context)
    iceflow_sol = solve(iceflow_prob, solver,
                    reltol=1e-6, save_everystep=false, 
                    progress=true, progress_steps = 10)

    return iceflow_sol.u[end]
end
end # @everywhere

"""
    train_iceflow_UDE(H₀, UA, θ, train_settings, PDE_refs, temp_series)

Train the Shallow Ice Approximation iceflow UDE
"""
function train_iceflow_UDE(H₀, UA, θ, train_settings, PDE_refs, temp_series)
    H = deepcopy(H₀)
    optimizer = train_settings[1]
    epochs = train_settings[2]
    # Tuple with all the temp series and H_refs
    context = (B, H)
    loss(θ) = loss_iceflow(θ, context, UA, PDE_refs, temp_series) # closure

    println("Training iceflow UDE...")
    # println("Using solver: ", solver)
    iceflow_trained = DiffEqFlux.sciml_train(loss, θ, optimizer, cb=callback, maxiters = 1)

    return iceflow_trained
end

@everywhere begin 

callback = function (θ,l) # callback function to observe training
    println("Epoch #$current_epoch - Loss $loss_type: ", l)

    # pred_A = predict_A̅(UA, θ, collect(-20.0:0.0)')
    # pred_A = [pred_A...] # flatten
    # true_A = A_fake(-20.0:0.0, noise)

    # Plots.scatter(-20.0:0.0, true_A, label="True A")
    # plot_epoch = Plots.plot!(-20.0:0.0, pred_A, label="Predicted A", 
    #                     xlabel="Long-term air temperature (°C)",
    #                     ylabel="A", ylims=(2e-17,8e-16),
    #                     legend=:topleft)
    # Plots.savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.png"))
    global current_epoch += 1

    false
end

"""
    loss_iceflow(θ, context, UA, PDE_refs::Dict{String, Any}, temp_series) 

Loss function based on glacier ice velocities and/or ice thickness
"""
function loss_iceflow(θ, context, UA, PDE_refs::Dict{String, Any}, temp_series) 
    if multibatch
        H_preds = predict_iceflow(θ, UA, context, temp_series)
    else
        H_preds = predict_iceflow_onebatch(θ, UA, context, temp_series)
    end
    
    # Compute loss function for the full batch
    l_Vx, l_Vy, l_H = 0.0, 0.0, 0.0

    if multibatch
        for i in 1:length(H_preds)

            # Get ice thickness from the reference dataset
            H_ref = PDE_refs["H_refs"][i]
            # Get ice thickness from the UDE predictions
            H = H_preds[i]
            # Get ice velocities for the reference dataset
            Vx_ref = PDE_refs["V̄x_refs"][i]
            Vy_ref = PDE_refs["V̄y_refs"][i]
            # Get ice velocities from the UDE predictions
            V̄x_pred, V̄y_pred = avg_surface_V(H_preds[i], B, mean(temp_series[i]), "UDE") # Average velocity with average temperature

            if random_sampling_loss
                # sample random indices for which V_ref is non-zero
                n_sample = 10
                n_counts = 0 
                while n_counts < n_sample
                    i, j = rand(1:nx), rand(1:ny)
                    if Vx_ref[i,j] != 0.0
                        #println("New non-zero count for loss function: ", i, j)
                        n_counts += 1
                        l_Vx += (V̄x_pred[i,j] - Vx_ref[i,j] )^2
                        l_Vy += (V̄y_pred[i,j] - Vy_ref[i,j] )^2
                        # what about trying a percentual error?
                        # l_H += ( (H[i,j] - H_ref[i,j]) / H_ref[i,j] )^2
                    end
                end
                l_Vx = l_Vx / n_sample
                l_Vy = l_Vy / n_sample
            else
                # Classic loss function with the full matrix
                l_H += Flux.Losses.mse(H[H_ref .!= 0.0], H_ref[H_ref.!= 0.0]; agg=mean)
                l_Vx += Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0], Vx_ref[Vx_ref.!= 0.0]; agg=mean)
                l_Vy += Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0], Vy_ref[Vy_ref.!= 0.0]; agg=mean)
            end
        end
    else 
        i = 1
        # Get ice thickness from the reference dataset
        H_ref = PDE_refs["H_refs"][i]
        # Get ice thickness from the UDE predictions
        H = H_preds
        # Get ice velocities for the reference dataset
        Vx_ref = PDE_refs["V̄x_refs"][i]
        Vy_ref = PDE_refs["V̄y_refs"][i]
        # Get ice velocities from the UDE predictions
        V̄x_pred, V̄y_pred = avg_surface_V(H_preds, B, mean(temp_series[i]), "UDE") # Average velocity with average temperature

        if random_sampling_loss
            # sample random indices for which V_ref is non-zero
            n_sample, n_counts = 50, 0
            nxy = length(H_ref[H_ref .!= 0.0])
            # Zygote.ignore() do 
            #     @infiltrate
            # end
            while n_counts < n_sample
                j = rand(1:nxy-2)
                H_ref_f = H_ref[H_ref .!= 0.0]
                Vx_ref_f = Vx_ref[Vx_ref .!= 0.0]
                Vy_ref_f = Vy_ref[Vy_ref .!= 0.0]
                H_f = H[H_ref .!= 0.0]
                V̄x_pred_f = V̄x_pred[Vx_ref .!= 0.0]
                V̄y_pred_f = V̄y_pred[Vy_ref .!= 0.0]

                normH = H_ref_f[j]  .+ ϵ
                normVx = Vx_ref_f[j] .+ ϵ
                normVy = Vy_ref_f[j] .+ ϵ
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
            normH = H_ref[H_ref .!= 0.0] .+ ϵ
            normVx = Vx_ref[Vx_ref .!= 0.0] .+ ϵ
            normVy = Vy_ref[Vy_ref .!= 0.0] .+ ϵ
            l_H += Flux.Losses.mse(H[H_ref .!= 0.0] ./normH, H_ref[H_ref.!= 0.0] ./normH; agg=mean) 
            l_Vx += Flux.Losses.mse(V̄x_pred[Vx_ref .!= 0.0] ./normVx, Vx_ref[Vx_ref.!= 0.0] ./normVx; agg=mean)
            l_Vy += Flux.Losses.mse(V̄y_pred[Vy_ref .!= 0.0] ./normVy, Vy_ref[Vy_ref.!= 0.0] ./normVy; agg=mean)
        end
    end

    @assert (loss_type == "H" || loss_type == "V" || loss_type == "HV") "Invalid loss_type. Needs to be 'H', 'V' or 'HV'"
    if loss_type == "H"
        l_avg = l_H/length(H_preds)
    elseif loss_type == "V"
        l_avg = (l_Vx/length(PDE_refs["V̄x_refs"]) + l_Vy/length(PDE_refs["V̄y_refs"]))/2
    elseif loss_type == "HV"
        l_avg = (l_Vx/length(PDE_refs["V̄x_refs"]) + l_Vy/length(PDE_refs["V̄y_refs"]) + l_H/length(H_preds))/3
    end
    return l_avg
end

"""
    predict_iceflow(θ, UA, context, temp_series) 

Makes a prediction of glacier evolution with the UDE for a given temperature series
"""
function predict_iceflow(θ, UA, context, temp_series)
    # (B, H)
    H = context[2]

    # Train UDE in parallel
    H_preds = pmap(temps -> prob_iceflow_UDE(θ, H, temps, context, UA), temp_series)

    return H_preds
end

function predict_iceflow_onebatch(θ, UA, context, temp_series)
    # (B, H)
    H = context[2]

    # Train UDE in parallel
    H_preds = prob_iceflow_UDE(θ, H, temp_series[1], context, UA)

    return H_preds
end

function prob_iceflow_UDE(θ, H, temps, context, UA) 
        
    # println("Processing temp series ≈ ", mean(temps))
    iceflow_UDE_batch(H, θ, t) = iceflow_NN(H, θ, t, context, temps, UA) # closure
    iceflow_prob = ODEProblem(iceflow_UDE_batch,H,(0.0,t₁),θ)
    iceflow_sol = solve(iceflow_prob, solver, u0=H, p=θ,
                    reltol=1e-6, save_everystep=false, 
                    progress=true, progress_steps = 10)

    return iceflow_sol.u[end]
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
function iceflow_NN(H, θ, t, context, temps, UA)

    year = floor(Int, t) + 1
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
    #A, B, S, dSdx, dSdy, D, norm_temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year, H_ref, H, UA, θ
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
    
    # Update glacier surface altimetry
    S .= B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx .= diff_x(S) / Δx
    dSdy .= diff_y(S) / Δy
    ∇S .= (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 

    Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s 
    D .= Γ .* avg(H).^(n + 2) .* ∇S

    # Compute flux components
    dSdx_edges .= diff_x(S[:,2:end - 1]) / Δx
    dSdy_edges .= diff_y(S[2:end - 1,:]) / Δy
    Fx .= .-avg_y(D) .* dSdx_edges
    Fy .= .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    inn(dH) .= .-(diff_x(Fx) / Δx .+ diff_y(Fy) / Δy) # MB to be added here 
end

"""
    SIA(H, A, context)

Compute a step of the Shallow Ice Approximation UDE in a forward model. Allocates memory.
"""
function SIA(H, A, context)
    # Retrieve parameters
    B = context[1]

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 

    Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s 
    D = Γ .* avg(H).^(n + 2) .* ∇S

    # Compute flux components
    dSdx_edges = diff_x(S[:,2:end - 1]) / Δx
    dSdy_edges = diff_y(S[2:end - 1,:]) / Δy
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
function avg_surface_V(H, B, temp, sim)

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 
    
    @assert (sim == "UDE" || sim == "PDE") "Wrong type of simulation. Needs to be 'UDE' or 'PDE'."
    if sim == "UDE"
        A = predict_A̅(UA, θ, [temp]) # FastChain prediction requires explicit parameters
    elseif sim == "PDE"
        A = A_fake(temp, noise)
    end
    Γ₂ = 2 * A * (ρ * g)^n / (n+1)     # 1 / m^3 s 
    # Zygote.ignore() do 
    #     @infiltrate
        
    # end
    D = Γ₂ .* avg(H).^(n + 1) .* ∇S
    
    # Compute averaged surface velocities
    Vx = - D .* avg_y(dSdx)
    Vy = - D .* avg_x(dSdy)

    return Vx, Vy
        
end

"""
    A_fake(temp, noise=false)

Fake law establishing a theoretical relationship between ice viscosity (A) and long-term air temperature.
"""
function A_fake(temp, noise=false)
    A = @. minA + (maxA - minA) * ((temp-minT)/(maxT-minT) )^2
    if noise
        A = A .+ randn(rng_seed(), length(temp)).*4e-17
    end

    return A
end

"""
    predict_A̅(UA, θ, temp)

Predicts the value of A with a neural network based on the long-term air temperature.
"""
predict_A̅(UA, θ, temp) = UA(temp, θ) .* 1e-16

function fake_temp_series(t, means=Array{Float64}([0,-2.0,-3.0,-5.0,-10.0,-12.0,-14.0,-15.0,-20.0]))
    temps, norm_temps, norm_temps_flat = [],[],[]
    for mean in means
        push!(temps, mean .+ rand(t).*1e-1) # static
        append!(norm_temps_flat, mean .+ rand(t).*1e-1) # static
    end

    # Normalise temperature series
    norm_temps_flat = Flux.normalise([norm_temps_flat...]) # requires splatting

    # Re-create array of arrays 
    for i in 1:t₁:length(norm_temps_flat)
        push!(norm_temps, norm_temps_flat[i:i+(t₁-1)])
    end

    return temps, norm_temps
end



sigmoid_A(x) = minA_out + (maxA_out - minA_out) / ( 1 + exp(-x) )

end # @everywhere 
    