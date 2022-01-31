###################################################################
###   Functions for PDE solving in staggered grids with UDEs   ####
###################################################################

include("utils.jl")

"""
    generate_ref_dataset(temp_series, gref, H₀, t)

Generate reference dataset with multiple glaciers forced with 
different temperature series.
"""
function generate_ref_dataset(temp_series, H₀, ensemble=ensemble)
    # Compute reference dataset in parallel
    H = deepcopy(H₀)
    
    # Initialize all matrices for the solver
    S, dSdx, dSdy = zeros(Float64,nx,ny),zeros(Float64,nx-1,ny),zeros(Float64,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1),zeros(Float64,nx-1,ny-1)
    D, dH, Fx, Fy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-2,ny-2),zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1)
    V, Vx, Vy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1)
    
    # Gather simulation parameters
    current_year = 0
    context = ArrayPartition([A], B, S, dSdx, dSdy, D, copy(temp_series[5]), dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, [current_year])

    function prob_iceflow_func(prob, i, repeat, context, temp_series) # closure
        
        batch_temp = round(mean(temp_series[i]), digits=3)
        println("Processing temp series #$i ≈ ", batch_temp)
        context.x[7] .= temp_series[i] # We set the temp_series for the ith trajectory

        return remake(prob, p=context, progress_name="#$i Temp series ≈ $batch_temp")
    end

    prob_func(prob, i, repeat) = prob_iceflow_func(prob, i, repeat, context, temp_series) # closure

    # Perform reference simulation with forward model 
    println("Running forward PDE ice flow model...\n")
    iceflow_prob = ODEProblem(iceflow!,H,(0.0,t₁),context)
    ensemble_prob = EnsembleProblem(iceflow_prob, prob_func = prob_func)
    iceflow_sol = solve(ensemble_prob, BS3(), ensemble, trajectories = length(temp_series), 
                        pmap_batch_size=length(temp_series), reltol=1e-6, 
                        progress=true, save_everystep=false, progress_steps = 50)

    # Save only matrices
    H_refs = [] 
    for result in iceflow_sol
        push!(H_refs, result.u[end])
    end

    return H_refs  
end
    
"""
    train_iceflow_UDE(H₀, UA, θ, H_refs, temp_series, hyparams)

Training of multiple UDEs of glacier ice flow in order to learn a given parameter
"""
function train_iceflow_UDE(H₀, UA, θ, H_refs, temp_series)
    H = deepcopy(H₀)
    current_year = 0.0
    # Tuple with all the temp series and H_refs
    context = (B, H, current_year, temp_series)
    loss(θ) = loss_iceflow(θ, context, UA, H_refs) # closure

    # Debugging
    # println("Gradients: ", gradient(loss, θ))
    # @infiltrate

    println("Training iceflow UDE...")
    iceflow_trained = DiffEqFlux.sciml_train(loss, θ, RMSProp(η), cb=callback, maxiters = epochs)

    return iceflow_trained
end

@everywhere begin

callback = function (θ,l) # callback function to observe training
    println("Epoch #$current_epoch - Loss H: ", l)

    pred_A = predict_A̅(UA, θ, collect(-20.0:0.0)')
    pred_A = [pred_A...] # flatten
    true_A = A_fake(-20.0:0.0)

    plot(true_A, label="True A")
    plot_epoch = plot!(pred_A, label="Predicted A")
    savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.png"))
    global current_epoch += 1

    false
end

"""
    loss(H, glacier_ref, UA, p, t, t₁)

Computes the loss function for a specific batch
"""
function loss_iceflow(θ, context, UA, H_refs) 
    H_preds = predict_iceflow(θ, UA, context)

    println("after predict")

    # Compute loss function for the full batch
    l_H = 0.0
    for (H_pred, H_ref) in zip(H_preds, H_refs)
        H = H_pred.u[end]
        l_H += Flux.Losses.mse(H[H .!= 0.0], H_ref[H.!= 0.0]; agg=mean)
    end

    l_H_avg = l_H/length(H_preds)

    println("sending loss")
    
    return l_H_avg
end

"""
    predict_iceflow(θ, UA, context, ensemble=ensemble)

Predict one batch of glacier ice flow UDEs.
"""
function predict_iceflow(θ, UA, context, ensemble=ensemble)

    function prob_iceflow_func(prob, i, repeat, context, UA) # closure

        # B, H, current_year, temp_series)  
        temp_series = context[4]
        batch_temp = string(mean(temp_series[i]))[1:4] # /!\ round() crashes here

        # We add the ith temperature series 
        iceflow_UDE_batch!(dH, H, θ, t) = iceflow_NN!(dH, H, θ, t, context, temp_series[i], UA) # closure
        
        return remake(prob, f=iceflow_UDE_batch!, progress_name="#$i Temp series ≈ $batch_temp")
    end

    prob_func(prob, i, repeat) = prob_iceflow_func(prob, i, repeat, context, UA)

    # (B, H, current_year, temp_series)
    H = context[2]
    tspan = (0.0,t₁)

    iceflow_UDE!(dH, H, θ, t) = iceflow_NN!(dH, H, θ, t, context, temp_series[5], UA) # closure
    iceflow_prob = ODEProblem(iceflow_UDE!,H,tspan,θ)
    ensemble_prob = EnsembleProblem(iceflow_prob, prob_func = prob_func)

    H_pred = solve(ensemble_prob, BS3(), ensemble, trajectories = length(temp_series), 
                    pmap_batch_size=length(temp_series), u0=H, p=θ, reltol=1e-6, 
                    sensealg = InterpolatingAdjoint(), save_everystep=false, 
                    progress=true, progress_steps = 10)

    return H_pred
end


"""
    iceflow!(H,p,t,t₁)

Forward ice flow model solving the Shallow Ice Approximation PDE 
"""
function iceflow!(dH, H, context,t)
    # Unpack parameters
    #A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year 
    current_year = Ref(context.x[18])
    A = Ref(context.x[1])
    
    # Get current year for MB and ELA
    year = floor(Int, t) + 1
    if year != current_year[] && year <= t₁
        temp = Ref{Float32}(context.x[7][year])
        A[] .= A_fake(temp[])
        current_year[] .= year
    end

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, context)
end  

"""
    iceflow_NN!(dH, H, θ, t, context, UA)

Hybrid forward ice flow model combining the SIA PDE and neural networks with neural networks into an UDE
"""
function iceflow_NN!(dH, H, θ, t, context, temps, UA)
   
    year = floor(Int, t) + 1
    if year <= t₁
        temp = temps[year]
    else
        temp = temps[year-1]
    end

    A = predict_A̅(UA, θ, [temp]) # FastChain prediction requires explicit parameters

    # Compute the Shallow Ice Approximation in a staggered grid
    dH .= SIA(dH, H, A, context)
end   

"""
    SIA!(dH, H, context)

Compute a step of the Shallow Ice Approximation PDE 
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
    SIA!(dH, H, context)

Compute a step of the Shallow Ice Approximation UDE 
"""
function SIA(dH, H, A, context)
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
    C_fake(MB, ∇S)

Fake law to determine the sliding rate factor 
"""
# TODO: to be updated in order to make it depend on the surrounding
# ice surface velocity pattern
function C_fake(MB, ∇S)
    MB[MB .> 0] .= 0
    MB[MB .< -20] .= 0
    #println("∇S max: ", maximum(∇S))
    #println("((MB).^2)/4) max: ", maximum(((MB).^2)/4))

    return ((avg(MB).^2)./6) .* 3e-13
    #return ((avg(MB).^2)./6) 

end

"""
    predict_A̅(UA, θ, temp)

Make a prediction of A with a neural network
"""
predict_A̅(UA, θ, temp) = UA(temp, θ) .* 1e-16

"""
    A_fake(ELA)

Fake law to determine A in the SIA
"""
function A_fake(temp)
    # Matching point MB values to A values

    #temp_range = -25:0.01:1

    #A_step = (maxA-minA)/length(temp_range)
    #A_range = sigmoid.(Flux.normalise(minA:A_step:maxA).*1.5e14).*1.5e-18 # add nonlinear relationship

    #A = A_range[closest_index(temp_range, temp)]

    return @. minA + (maxA - minA) * ((temp-minT)/(maxT-minT) )^2
    #return A
end

"""
    create_NNs()

Generates the hyperaparameters and the neural networks needed for the training of UDEs
"""
function create_NNs()
    ######### Define the network  ############
    # We determine the hyperameters for the training
    # hyparams = Hyperparameters()

    # Constraints A within physically plausible values
    minA = 0.3
    maxA = 8
    # rangeA = minA:1e-3:maxA
    # stdA = std(rangeA)*2
    # relu_A(x) = min(max(minA, x), maxA)
    #relu_A(x) = min(max(minA, 0.00001 * x), maxA)
    sigmoid_A(x) = minA + (maxA - minA) / ( 1 + exp(-x) )

    # A_init(custom_std, dims...) = randn(Float32, dims...) .* custom_std
    # A_init(custom_std) = (dims...) -> A_init(custom_std, dims...)

    UA = FastChain(
        FastDense(1,3, x->tanh.(x)),
        FastDense(3,10, x->tanh.(x)),
        FastDense(10,3, x->tanh.(x)),
        FastDense(3,1, sigmoid_A)
    )

    return UA
end

end # @everywhere 

