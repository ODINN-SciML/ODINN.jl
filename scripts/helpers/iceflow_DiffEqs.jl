###################################################################
###   Functions for PDE solving in staggered grids with UDEs   ####
###################################################################

using Zygote
using Flux
using Flux: @epochs
using Tullio
include("utils.jl")

# Patch suggested by Michael Abbott needed in order to correctly retrieve gradients
Flux.Optimise.update!(opt, x::AbstractMatrix, Δ::AbstractVector) = Flux.Optimise.update!(opt, x, reshape(Δ, size(x)))

"""
    generate_ref_dataset(temp_series, gref, H₀, t)

Training of reference dataset with multiple glaciers forced with 
different temperature series.
"""
function generate_ref_dataset(temp_series, H₀, t)
    
    # Compute reference dataset in parallel
    iceflow_prob, hm = map(temps -> ref_glacier(temps, H₀, t), temp_series)
    
    return iceflow_prob, hm
    
end


"""
    ref_glacier(temps, gref, H₀, t)

Training reference dataset of a single glacier
"""
function ref_glacier(temps, H₀, t)
      
    tempn = mean(temps)
    println("Reference simulation with temp ≈ ", tempn)
    H = deepcopy(H₀)
    
    # Initialize all matrices for the solver
    S, dSdx, dSdy = zeros(Float32,nx,ny),zeros(Float32,nx-1,ny),zeros(Float32,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1),zeros(Float32,nx-1,ny-1)
    D, dH, Fx, Fy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-2,ny-2),zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1)
    V, Vx, Vy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1)
    
    # Gather simulation parameters
    current_year = 0
    p = [A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year]

    # Perform reference simulation with forward model 
    println("Running forward PDE ice flow model...\n")
    iceflow_prob = ODEProblem(iceflow!,H,(0.0,t₁),p)
    #@time solve(iceflow_prob, alg_hints=[:stiff], reltol=1e-14,abstol=1e-14, progress=true, progress_steps = 1)
    @time iceflow_sol = solve(iceflow_prob, Vern7(), dt=1e-14, progress=true, progress_steps = 1)
    

    ### Glacier ice thickness evolution  ### Not that useful
    # hm11 = heatmap(H₀, c = :ice, title="Ice thickness (t=0)")
    # hm12 = heatmap(H, c = :ice, title="Ice thickness (t=$t₁)")
    # hm1 = Plots.plot(hm11,hm12, layout=2, aspect_ratio=:equal, size=(800,350),
    #     colorbar_title="Ice thickness (m)",
    #     clims=(0,maximum(H₀)), link=:all)
    # display(hm1)

    ###  Glacier ice thickness difference  ###
    lim = maximum( abs.(H .- H₀) )
    hm = heatmap(H .- H₀, c = cgrad(:balance,rev=true), aspect_ratio=:equal,
        clim = (-lim, lim),
        title="Variation in ice thickness")
    
    #if x11 
    #    display(hm2) 
    #end

    return iceflow_sol, hm
    
end

"""
    train_batch_iceflow_UDE!(H₀, UA, glacier_refs, temp_series, hyparams, idx, p)

Training of a batch for the iceflow UDE based on the SIA.
"""
function train_batch_iceflow_UDE(H₀, UA, glacier_refs, temp_series, hyparams, idxs)
    
    # Train UDE batch in parallel
    loss_UAs, back_UAs = map(idx -> train_iceflow_UDE(H₀, UA, glacier_refs, temp_series, hyparams, idx), idxs) 
    
    return loss_UA, back_UA
    
end


function train_iceflow_UDE(H₀, UA, glacier_refs, temp_series, hyparams, idx)
    temps = temp_series[idx]
    norm_temps = norm_temp_series[idx]
    glacier_ref = glacier_refs[idx]
    println("\nTemperature in training: ", temps[1])

    # Gather simulation parameters
    p = (Δx, Δy, Γ, A, B, norm_temps, C, α) 
    loss_UA, back_UA = iceflow_UDE(H₀,glacier_ref,UA,hyparams,p,t,t₁)   

    predicted_A = predict_A̅(UA, [mean(norm_temps)]')[1]
    fake_A = A_fake(mean(temps)) 
    A_error = predicted_A - fake_A
    println("Predicted A: ", predicted_A)
    println("Fake A: ", fake_A)
    println("A error: ", A_error)

    return loss_UA, back_UA
    
end


"""
    update_UDE_batch!(UA, back_UAs)

Update neural network weights after having trained on a whole batch of glaciers. 
"""
function update_UDE_batch!(UA, loss_UAs, back_UAs)
    
    println("Backpropagation...")
    # We update the weights with the gradients of all tha glaciers in the batch
    # This is equivalent to taking the gradient with respect of the full loss function
    θ = Flux.params(UA)
    
    for back_UA in back_UAs
        ∇_UA = back_UA(1)
        println("#$i Updating NN weights")
        Flux.Optimise.update!(opt, θ, ∇_UA) # with UA
    end

    #∇_UA = back_UA(one(mean(trackers["losses_batch"]))) # with UA
    #println("Updating NN weights")
    #Flux.Optimise.update!(opt, θ, ∇_UA) # with UA

    # Keep track of the loss function per batch
    println("Loss batch: ", mean(loss_UAs))  

end


"""
    plot_training!(UA, old_trained, temp_values, norm_temp_values)

Plot evolution of the functions learnt by the UDE. 
"""
function plot_training!(old_trained, UA, loss_UAs, temp_values, norm_temp_values)
    
    # Plot progress of the loss function 
    # temp_values = LinRange(-25, 0, 20)'
    # plot(temp_values', A_fake.(temp_values)', label="Fake A")
    # pfunc = scatter!(temp_values', predict_A̅(UA, temp_values)', yaxis="A", xaxis="Air temperature (°C)", label="Trained NN", color="red")
    # ploss = plot(trackers["losses"], title="Loss", xlabel="Epoch", aspect=:equal)
    # display(plot(pfunc, ploss, layout=(2,1)))
    
    # Plot the evolution
    plot(temp_values', A_fake.(temp_values)', label="Fake A")
    scatter!(temp_values', predict_A̅(UA, norm_temp_values)', yaxis="A", xaxis="Air temperature (°C)", label="Trained NN", color="red")#, ylims=(3e-17,8e-16)))
    pfunc = scatter!(temp_values', old_trained, label="Previous NN", color="grey", aspect=:equal, legend=:outertopright)#, ylims=(3e-17,8e-16)))
    ploss = plot(loss_UAs, xlabel="Epoch", ylabel="Loss", aspect=:equal, legend=:outertopright, label="")
    ptrain = plot(pfunc, ploss, layout=(2,1))

    savefig(ptrain,joinpath(root_dir,"plots/training","epoch$i.png"))
    #if x11 
    #    display(ptrain) 
    #end

    old_trained = predict_A̅(UA, norm_temp_values)'
    
end


"""
    iceflow_UDE!(H₀, glacier_ref, UA, hyparams, p, t, t₁)

Hybrid ice flow model solving and optimizing the Shallow Ice Approximation (SIA) PDE using 
Universal Differential Equations (UDEs)
"""

function iceflow_UDE(H₀, glacier_ref, UA, hyparams, p, t, t₁)
    
    # We define an optimizer
    opt = RMSProp(hyparams.η)
    # opt = ADAM(hyparams.η)
    #opt = BFGS(hyparams.η)
    
    # Retrieve model parameters
    θ = Flux.params(UA)
    # println("Resetting initial H state")
    H = deepcopy(H₀) # Make sure we go back to the original initial state for each epoch

    println("Forward pass")
    loss_UA, back_UA = Zygote.pullback(() -> loss(H, glacier_ref, UA, p, t, t₁), θ) # with UA

    # loss_UA, back_UA = Zygote.pullback(A -> loss(H, A, p, t, t₁), A) # inverse problem 

    # ∇_UA = back_UA(one(loss_UA))[1] # inverse problem

    # for ps in θ
    #    println("Gradients ∇_UA[ps]: ", ∇_UA[ps])
    # end
    # println("Gradients ∇_UA: ", ∇_UA)

    # println("Predicted A: ", predict_A̅(UA, [mean(p[6])]'))

    # Flux.Optimise.update!(opt, A, ∇_UA) # inverse problem
    # Δx, Δy, Γ, A, B, v, MB, MB_avg, C, α, var_format = p # unpack
    # p = (Δx, Δy, Γ, ŶA, B, v, MB, MB_avg, C, α, var_format) # repack
    
    return loss_UA, back_UA

end



"""
    loss(H, glacier_ref, UA, p, t, t₁)

Computes the loss function for a specific batch
"""
# We determine the loss function
function loss(H, glacier_ref, UA, p, t, t₁)
  
    H, V̂ = iceflow!(H, UA, p,t,t₁)

    l_H = sqrt(Flux.Losses.mse(H[H .!= 0.0], glacier_ref["H"][end][H.!= 0.0]; agg=sum))
    
    # l_V = sqrt(Flux.Losses.mse(V̂[V̂ .!= 0.0], mean(glacier_ref["V"])[V̂ .!= 0.0]; agg=sum))

    println("l_H: ", l_H)
    # println("l_V: ", l_V)

    # Zygote.ignore() do
    # #    hml = heatmap(mean(glacier_ref["V"]) .- V̂, title="Loss error - V")
    #    hml = heatmap(glacier_ref["H"][end] .- H, title="Loss error - H")
    #    display(hml)
    # end

    return l_H
end


"""
    iceflow!(H,p,t,t₁)

Forward ice flow model solving the Shallow Ice Approximation PDE 
"""
function iceflow!(dH, H, p,t)

    # Unpack parameters
    #A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year 
    current_year = @view p[end]
    A = @view p[1]
    
    # Get current year for MB and ELA
    year = floor(Int, t) + 1
    if year != current_year && year <= t₁

        #println("Year: ", year)
        #println("current_year before: ", current_year)
        #println("current_year before p: ", p[end])
        
        # Predict A with the fake A law
        #println("temps: ", temps)
        temp = @view p[7][year]
        A .= A_fake(temp)
        #println("A fake: ", ŶA)

        # Unpack and repack tuple with updated A value
        current_year .= year
        #println("current_year after: ", current_year)

    end

    # Compute the Shallow Ice Approximation in a staggered grid
    SIA!(dH, H, p)

end  


predict_A̅(UA, temp) = UA(temp) .* 1e-16


"""
    iceflow!(H,p,t,t₁)

Hybrid forward ice flow model combining the SIA PDE and neural networks with neural networks into an UDE
"""
function iceflow!(H, UA, p,t,t₁)

    # Retrieve input variables  
    let                  
    current_year = 0
    total_iter = 0
    Vx, Vx_buff, Vy, Vy_buff = zeros(nx-1,ny-1),zeros(nx-1,ny-1),zeros(nx-1,ny-1),zeros(nx-1,ny-1)
    t_step = 0
    temps = p[6]

    # Forward scheme implementation
    while t < t₁
        let
        iter = 1
        err = 2 * tolnl

        V = (zeros(nx-1,ny-1),zeros(nx-1,ny-1))
        Hold = copy(H)
        dHdt = zeros(nx, ny)

        # Get current year for MB and ELA
        year = floor(Int, t) + 1

        if year != current_year

            # println("Year: ", year)
        
            # Predict value of `A`
            temp = [temps[year]]'
                    
            ŶA = predict_A̅(UA, temp)

            # Zygote.ignore() do
            #     println("Current params: ", Flux.params(UA))

            #     println("ŶA: ", ŶA )

            #     display(heatmap(MB_avg[year], title="MB"))
            # end
        
            ## Unpack and repack tuple with updated A value
            Δx, Δy, Γ, A, B, temps, C, α = p
            p = (Δx, Δy, Γ, ŶA, B, temps, C, α)
            current_year = year
        end
           
        while err > tolnl && iter < itMax+1
        # while iter < itMax + 1
       
            Err = copy(H)

            # Compute the Shallow Ice Approximation in a staggered grid
            F, V, dτ = SIA(H, p)

            # Compute the residual ice thickness for the inertia
            @tullio ResH[i,j] := -(H[i,j] - Hold[i,j])/Δt + F[pad(i-1,1,1),pad(j-1,1,1)]

            dHdt_ = copy(dHdt)
            @tullio dHdt[i,j] := dHdt_[i,j]*damp + ResH[i,j]
                            
            # We keep local copies for tullio
            H_ = copy(H)
            
            # Update the ice thickness
            #@tullio H[i,j] := max(0.0, H_[i,j] + dHdt[i,j]*dτ[pad(i-1,1,1),pad(j-1,1,1)])
            @tullio H[i,j] := max(0.0, H_[i,j] + dHdt[i,j]*dτ)
            
            Zygote.ignore() do
                if mod(iter, nout) == 0
                    # Compute error for implicit method with damping
                    Err = Err .- H
                    err = maximum(Err)

                    if isnan(err)
                        error("""NaNs encountered.  Try a combination of:
                                    decreasing `damp` and/or `dtausc`, more smoothing steps""")
                    end
                end
            end

            iter += 1
            total_iter += 1

        end 
          
        t += Δt
        t_step += 1

        # Zygote.ignore() do
        #     @infiltrate
        # end
        
        # Fill buffers to handle Zygote "Mutating arrays" limitation
        Vx_buff = copy(Vx)
        Vy_buff = copy(Vy)

        @tullio Vx[i,j] := Vx_buff[i,j] + V[1][i,j]
        @tullio Vy[i,j] := Vy_buff[i,j] + V[2][i,j]
    
        end # let
    end 

    # Compute average surface velocity field
    V̂ = ((Vx./t_step).^2 + (Vy./t_step).^2).^(1/2)

    return H, V̂

    end   # let

end

"""
    SIA(H, p)

Compute a step of the Shallow Ice Approximation PDE in a forward model
"""

function SIA!(dH, H, p)
    
    # Retrieve parameters
    #A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, current_year  
    
    S = p[3]
    B = p[2]
    dSdx = p[4]
    dSdy = p[5]
    ∇S = p[10]
    D = p[6]
    dSdx_edges = p[8]
    dSdy_edges = p[9]
    Fx = p[11]
    Fy = p[12]
    Vx = p[13]
    Vy = p[14]
    
    # Update glacier surface altimetry
    S .= B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx .= diff(S, dims=1) / Δx
    dSdy .= diff(S, dims=2) / Δy
    ∇S .= (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 

    Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s 
    
    D .= Γ .* avg(H).^(n + 2) .* ∇S

    # Compute flux components
    dSdx_edges .= diff(S[:,2:end - 1], dims=1) / Δx
    dSdy_edges .= diff(S[2:end - 1,:], dims=2) / Δy
    Fx .= .-avg_y(D) .* dSdx_edges
    Fy .= .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    inn(dH) .= .-(diff(Fx, dims=1) / Δx .+ diff(Fy, dims=2) / Δy) # MB to be added here 
    #@tullio dH[i,j] = -(diff(Fx, dims=1)[pad(i-1,1,1),pad(j-1,1,1)] / Δx + diff(Fy, dims=2)[pad(i-1,1,1),pad(j-1,1,1)] / Δy)

    # Compute velocities    
    Vx .= -D./(avg(H) .+ ϵ).*avg_y(dSdx)
    Vy .= -D./(avg(H) .+ ϵ).*avg_x(dSdy)
    
    #@tullio dH[i,j] = H[i,j] + F[pad(i-1,1,1),pad(j-1,1,1)]
    #dH = H .+ F
    
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

# function A_fake(MB_buffer, shape, var_format)
#     # Matching point MB values to A values
#     maxA = 3e-16
#     minA = 1e-17

#     if var_format == "matrix"
#         MB_range = reverse(-15:0.01:8)
#     elseif var_format == "scalar"
#         MB_range = reverse(-3:0.01:0)
#     end

#     A_step = (maxA-minA)/length(MB_range)
#     A_range = sigmoid.(Flux.normalise(minA:A_step:maxA).*2.5f11).*5f-16 # add nonlinear relationship

#     if var_format == "matrix"
#         A = []
#         for MB_i in MB_buffer
#             push!(A, A_range[closest_index(MB_range, MB_i)])
#         end
#         A = reshape(A, shape)
#     elseif var_format == "scalar"
#         A = A_range[closest_index(MB_range, nanmean(MB_buffer))]
#     end

#     return A
# end

"""
    create_NNs()

Generates the hyperaparameters and the neural networks needed for the training of UDEs
"""
function create_NNs()
    ######### Define the network  ############
    # We determine the hyperameters for the training
    hyparams = Hyperparameters()

    # Leaky ReLu as activation function
    leakyrelu(x, a=0.01) = max(a*x, x)

    # Constraints A within physically plausible values
    minA = 0.3
    maxA = 8
    rangeA = minA:1e-3:maxA
    stdA = std(rangeA)*2
    relu_A(x) = min(max(minA, x), maxA)
    #relu_A(x) = min(max(minA, 0.00001 * x), maxA)
    sigmoid_A(x) = minA + (maxA - minA) / ( 1 + exp(-x) )

    A_init(custom_std, dims...) = randn(Float32, dims...) .* custom_std
    A_init(custom_std) = (dims...) -> A_init(custom_std, dims...)

    #UA = Chain(
    #    Dense(1,10), 
    #    #Dense(10,10, x->tanh.(x), init = A_init(stdA)), 
    #    Dense(10,10, x->tanh.(x)), #init = A_init(stdA)), 
    #    #Dense(10,5, x->tanh.(x), init = A_init(stdA)), 
    #    Dense(10,5, x->tanh.(x)), #init = A_init(stdA)), 
    #    Dense(5,1, sigmoid_A)
    #)

    UA = Chain(
        Dense(1,3, x->tanh.(x)),
        Dense(3,10, x->tanh.(x)),
        Dense(10,3, x->tanh.(x)),
        Dense(3,1, sigmoid_A)
    )

    return hyparams, UA
end

"""
    callback(l)

Callback to track evolution of the neural network's training. 
"""
# Callback to show the loss during training
callback(l) = begin
    # Container to track the losses
    losses = Float64[]
    push!(losses, l)
    if length(losses)%50==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end
