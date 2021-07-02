###################################################################
###   Functions for PDE solving in staggered grids with UDEs   ####
###################################################################

using Zygote
using PaddedViews
using Flux
using Flux: @epochs
using DiffEqOperators
using Tullio
include("utils.jl")

"""
    iceflow_UDE!(H,H_ref,UA,hyparams,p,t,t₁)

Hybrid ice flow model solving and optimizing the Shallow Ice Approximation (SIA) PDE using 
Universal Differential Equations (UDEs)
"""
function iceflow_UDE!(H,H_ref,UA,hyparams,p,t,t₁)
    println("Running forward UDE ice flow model...\n")
    # For now, train UDE only with the last timestamp with "observations"
    # Use batchsize similar to dataset size for easy training
    Y = vec(H_ref["H"][end]) # flatten matrix
    X = p[7] # flattened MB matrix
    # println("Y: ", size(Y))
    # println("X: ", size(X))

    # We define an optimizer
    #opt = RMSProp(hyparams.η, 0.95)
    opt = ADAM(hyparams.η)

    #data = Flux.Data.DataLoader((X, Y), batchsize=hyparams.batchsize, (X, Y), shuffle=false)
    # Train the UDE for a given number of epochs
    #H_buff = Zygote.Buffer(H)
    @epochs hyparams.epochs hybrid_train!(loss, UA, opt, H, p, t, t₁)

    #H = copy(H_buff)  
end

"""
    hybrid_train!(loss, ps_Up, ps_Ut, data, opt)

Train hybrid ice flow model based on UDEs.
"""
function hybrid_train!(loss, UA, opt, H, p, t, t₁)
    # Retrieve model parameters
    # We get the model parameters to be trained
    ps_UA = Flux.params(UA)
    # ps_UA = Params(ps_UA)

    # back is a method that computes the product of the gradient so far with its argument.
    println("Forward pass")
    #loss_UA, back_UA = Zygote.pullback(() -> loss(H, UA, p, t, t₁), ps_UA)
    loss_UA, back_UA = Zygote.pullback(UA -> loss(H, UA, p, t, t₁), UA)
    #loss_UA, back_UA = Zygote._pullback(ps_UA -> loss(H, UA, p, t, t₁), ps_UA)
    #loss_UA, back_UA = Zygote._pullback(UA -> iceflow!(H, UA, p,t,t₁), UA)
    #grad_UA = Zygote.jacobian(UA -> iceflow!(H, UA, p,t,t₁), UA)
    #loss_UA, back_UA = Zygote._pullback(UA -> SIA(H, p, y), UA)
    #@code_typed Zygote._pullback(() -> loss(data, H, p, t, t₁), ps_UA)
    # Callback to track the training
    #callback(train_loss_UA)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    println("Backpropagation")
    #@time ∇_UA = back_UA(one(loss_UA))
    ∇_UA = back_UA(one(loss_UA))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
    println("Updating NN weights")
    @infiltrate
    for p in ps_UA
        @show p, ∇_UA[p]
    end
    Flux.update!(opt, ps_UA, ∇_UA)
    # Here you might like to check validation set accuracy, and break out to do early stopping.
    
    lim = maximum( abs.(H .- H₀) )
    hmz = heatmap(H .- H₀, c = cgrad(:balance,rev=true), aspect_ratio=:equal,
        xlims=(0,180), ylims=(0,180), clim = (-lim, lim),
        title="Variation in ice thickness")
    display(hmz)

end

"""
    loss(batch)

Computes the loss function for a specific batch.
"""
# We determine the loss function
function loss(H, UA, p, t, t₁)
    l_H, l_A = 0.0f0, 0.0f0
   
    # Compute l_H as the difference between the simulated H with UA(x) and H_ref
    H = iceflow!(H, UA, p,t,t₁)
    #println("H: ", maximum(H))
    l_H = sqrt(Flux.Losses.mse(H, H_ref["H"][end]; agg=mean))
    # println("l_A: ", l_A)
    println("l_H: ", l_H)
    # l = l_H + l_A

    #Zygote.ignore() do
    #    hml = heatmap(H_ref["H"][end] .- H, title="Loss")
    #    display(hml)
    #end

    return l_H
end


"""
    iceflow!(H,p,t,t₁)

Forward ice flow model solving the Shallow Ice Approximation PDE 
"""
function iceflow!(H,H_ref::Dict, p,t,t₁)

    println("Running forward PDE ice flow model...\n")
    # Instantiate variables
    let             
    current_year = 0
    MB_avg = p[8]  
    total_iter = 0
    ts_i = 1

    # Manual explicit forward scheme implementation
    while t < t₁
        let
        iter = 1
        err = 2 * tolnl
        Hold = copy(H) # hold value of H for the other iteration in the implicit method
        # we need to define dHdt for iter = 1
        #dHdt = zeros(nx-2, ny-2) # with broadcasting
        dHdt = zeros(nx, ny) # with Tullio

        # Get current year for MB and ELA
        year = floor(Int, t) + 1
        if(year != current_year)
            
            # Predict A with the fake A law
            ŶA = A_fake(MB_avg[year], size(H))

            Zygote.ignore() do
                if(year == 1)
                    println("ŶA max: ", maximum(ŶA))
                    println("ŶA min: ", minimum(ŶA))
                end

            end
            # Unpack and repack tuple with updated A value
            Δx, Δy, Γ, A, B, v, MB, MB_avg, C, α = p
            p = (Δx, Δy, Γ, ŶA, B, v, MB, MB_avg, C, α)
            current_year = year
            println("Year ", year)
        end
        y = (year, current_year)

        if method == "explicit"

            F, dτ, current_year = SIA(H, p, y)
            inn(H) .= max.(0.0, inn(H) .+ Δt * F)
            t += Δt_exp
            total_iter += 1 

        elseif method == "implicit"

            while err > tolnl && iter < itMax
            
                Err = copy(H)

                # Compute the Shallow Ice Approximation in a staggered grid
                F, dτ, current_year = SIA(H, p, y)

                # Implicit method with broadcasting
                # Differentiate H via a Picard iteration method
                #ResH = -(inn(H) .- inn(Hold))/Δt .+ F  # with broadcasting
                #dHdt = damp .* dHdt .+ ResH # with broadcasting
                # Update the ice thickness
                #inn(H) .= max.(0.0, inn(H) .+ dτ .* dHdt) # with broadcasting 

                # implicit method with Tullio  
                @tullio ResH[i,j] := -(H[i,j] - Hold[i,j])/Δt + F[pad(i-1,1,1),pad(j-1,1,1)]
                
                dHdt_ = copy(dHdt)
                @tullio dHdt[i,j] := dHdt_[i,j]*damp + ResH[i,j]
                
                H_ = copy(H)
                @tullio H[i,j] := max(0.0, H_[i,j] + dHdt[i,j]*dτ[pad(i-1,1,1),pad(j-1,1,1)])
                

                if mod(iter, nout) == 0
                    # Compute error for implicit method with damping
                    Err = Err .- H
                    err = maximum(Err)
                    println("error at iter ", iter, ": ", err)

                    if isnan(err)
                        error("""NaNs encountered.  Try a combination of:
                                    decreasing `damp` and/or `dtausc`, more smoothing steps""")
                    end
                end
            
                iter += 1
                total_iter += 1
            end

            t += Δt

        end
        end

        # Store timestamps to be used for training of the UDEs
        if ts_i < length(H_ref["timestamps"])+1
            if t >= H_ref["timestamps"][ts_i]
                println("Saving H at year ", H_ref["timestamps"][ts_i])
                push!(H_ref["H"], H)
                ts_i += 1
            end          
        end        
    end 

    println("Total Number of iterartions: ", total_iter)
    end
    
    println("Saving reference data")
    save(joinpath(root_dir, "../../data/H_ref.jld"), "H_ref", H_ref)

    return H
end


"""
    iceflow!(H,p,t,t₁)

Hybrid forward ice flow model combining the SIA PDE and neural networks with neural networks into an UDE
"""
function iceflow!(H, UA, p,t,t₁)

    # Retrieve input variables  
    let                  
    current_year = 0
    MB_avg = p[8]  
    total_iter = 0
    global model = "UDE_A"

    # Forward scheme implementation
    while t < t₁
        let
        iter = 1
        err = 2 * tolnl
        Hold = copy(H)
        dHdt = zeros(nx, ny)

        # Get current year for MB and ELA
        year = floor(Int, t) + 1

        if(year != current_year)
            
            # Predict A with the NN
            ŶA = UA(vec(MB_avg[year])')

            Zygote.ignore() do
                println("Current params: ", params(UA))

                # if(year == 1)
                println("ŶA max: ", maximum(ŶA))
                println("ŶA min: ", minimum(ŶA))
                # end

                # display(heatmap(MB_avg[year], title="MB"))
            end
        
            # Unpack and repack tuple with updated A value
            Δx, Δy, Γ, A, B, v, MB, MB_avg, C, α = p
            p = (Δx, Δy, Γ, ŶA, B, v, MB, MB_avg, C, α)
            current_year = year
            println("Year ", year)
        end
        y = (year, current_year)

        if method == "implicit"
            
            while err > tolnl && iter < itMax

                #println("iter: ", iter)
            
                Err = copy(H)

                # Compute the Shallow Ice Approximation in a staggered grid
                F, dτ, current_year = SIA(H, p, y)

                # Compute the residual ice thickness for the inertia
                @tullio ResH[i,j] := -(H[i,j] - Hold[i,j])/Δt + F[pad(i-1,1,1),pad(j-1,1,1)]

                dHdt_ = copy(dHdt)
                @tullio dHdt[i,j] := dHdt_[i,j]*damp + ResH[i,j]
                              
                # We keep local copies for tullio
                H_ = copy(H)
                
                # Update the ice thickness
                @tullio H[i,j] := max(0.0, H_[i,j] + dHdt[i,j]*dτ[pad(i-1,1,1),pad(j-1,1,1)])

                #println("maximum H: ",maximum(H))
                #println("maximum H on borders: ", maximum([maximum(H[1,:]), maximum(H[:,1]), maximum(H[nx,:]), maximum(H[:,ny])]))

                #@show isderiving()
              
                if mod(iter, nout) == 0
                    # Compute error for implicit method with damping
                    Err = Err .- H
                    err = maximum(Err)
                    # println("error: ", err)
                    #@infiltrate

                    if isnan(err)
                        error("""NaNs encountered.  Try a combination of:
                                    decreasing `damp` and/or `dtausc`, more smoothing steps""")
                    end
                end
            
                iter += 1
                total_iter += 1

            end

            #println("t: ", t)
          
            t += Δt
        end
        end # let
    end   
    end # let

    # return sum(H)
    return H

end

"""
    SIA(H, p, y)

Compute a step of the Shallow Ice Approximation PDE in a forward model
"""

function SIA(H, p, y)
    Δx, Δy, Γ, A, B, v, MB, MB_avg, C, α = p
    year, current_year = y

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx  = diff(S, dims=1) / Δx
    dSdy  = diff(S, dims=2) / Δy
    ∇S = sqrt.(avg_y(dSdx).^2 .+ avg_x(dSdy).^2)

    # Compute diffusivity on secondary nodes
    # A here should have the same shape as H
    #                                     ice creep  +  basal sliding
    #D = (avg(pad(H)).^n .* ∇S.^(n - 1)) .* (A.*(avg(pad(H))).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
    # Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s # Γ from Jupyter notebook. To do: standarize this value in patameters.jl
    if(model == "standard")
        Γ = 2 * avg(A) * (ρ * g)^n / (n+2)
    elseif(model == "UDE_A")
        # Zygote.ignore() do
        #     display(heatmap(avg(reshape(A, size(H)))))
        # end
        Γ = 2 * avg(reshape(A, size(H))) * (ρ * g)^n / (n+2) # 1 / m^3 s # Γ from Jupyter notebook. To do: standarize this value in patameters.jl
        # Γ = 2 * avg(reshape(A, size(H))) * (ρ * g)^n / (n+2) # 1 / m^3 s # Γ from Jupyter notebook. To do: standarize this value in patameters.jl
       #Γ = 2 * A * (ρ * g)^n / (n+2)
    end
    D = Γ .* avg(H).^(n + 2) .* ∇S.^(n - 1) 
  

    #D = (Γ * avg(H).^n .* ∇S.^(n - 1)) .* (avg(reshape(A, size(H))) .* avg(H)).^(n-1) .+ (α*(n+2)*C)/(n-2)

    # Compute flux components
    dSdx_edges = diff(S[:,2:end - 1], dims=1) / Δx
    dSdy_edges = diff(S[2:end - 1,:], dims=2) / Δy
    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges    
    #  Flux divergence
    F = .-(diff(Fx, dims=1) / Δx .+ diff(Fy, dims=2) / Δy) # MB to be added here 

    # Compute dτ for the implicit method
    dτ = dτsc * min.( 10.0 , 1.0./(1.0/Δt .+ 1.0./(cfl./(ϵ .+ avg(D)))))

    return F, dτ, current_year
    #return sum(F)

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
function A_fake(MB_buffer, shape)
    # Matching point MB values to A values
    MB_range = -15:0.5:10
    A_step = (1.57e-16-1.57e-17)/length(MB_range)
    A_range = 1.57e-17:A_step:1.57e-16

    A = []
    for MB_i in MB_buffer
        push!(A, A_range[closest_index(MB_range, MB_i)])
    end
    
    return reshape(A, shape)
end

# function A_fake(ELA)
#     # Matching ELA values to A values
#     ELA_range = 2806:3470
#     A_step = (1.57e-16-1.57e-17)/length(ELA_range)
#     A_range = 1.57e-17:A_step:1.57e-16
    
#     return A_range[closest_index(ELA_range, ELA)]
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

    # Constraints A within physically plausible values
    relu_A(x) = min(max(1.58e-17, x), 1.58e-16)
    sigmoid_A(x) = 1.58e-17 + (1.58e-16 - 1.58e-17) / ( 1 + exp(-x) )

    # Define the networks 1->10->5->1
    UA = Chain(
        Dense(1,10,initb = Flux.zeros), 
        BatchNorm(10, leakyrelu),
        Dense(10,5,initb = Flux.zeros), 
        BatchNorm(5, leakyrelu),
        #Dense(5,1, relu_A, initb = Flux.zeros) 
        Dense(5,1, sigmoid_A, initb = Flux.zeros) 
    )

    return hyparams, UA
end

# Container to track the losses
losses = Float32[]

"""
    callback(l)

Callback to track evolution of the neural network's training. 
"""
# Callback to show the loss during training
callback(l) = begin
    push!(losses, l)
    if length(losses)%50==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end

