###################################################################
###   Functions for PDE solving in staggered grids with UDEs   ####
###################################################################

using Zygote
include("utils.jl")

"""
    iceflow_toy!(H,p,t,t₁)

Forward ice flow model solving the Shallow Ice Approximation PDE 
"""
function iceflow_toy!(H,H_ref, p,t,t₁)

    println("Running forward PDE ice flow model...\n")
    # Retrieve input variables                    
    Δx, Δy, Γ, A, B, v, MB, ELAs, C, α = p
    ts_i = 1

    # Manual explicit forward scheme implementation
    while t < t₁

        # Get current year for MB and ELA
        year = floor(Int, t) + 1

        # Update glacier surface altimetry
        S = B .+ H

        # All grid variables computed in a staggered grid
        # Compute surface gradients on edges
        dSdx  .= diff(S, dims=1) / Δx
        dSdy  .= diff(S, dims=2) / Δy
        ∇S .= sqrt.(avg_y(dSdx).^2 .+ avg_x(dSdy).^2)

        # Compute diffusivity on secondary nodes
        #smodel = "standard"
        if model == "stantard"
            #                                     ice creep  +  basal sliding
            D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (A.*avg(H).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
        elseif model == "fake A"
            # Diffusivity with fake A function
            D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (A_fake(buffer_mean(MB, year)) .* avg(H).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
        elseif model == "fake C"
            # Diffusivity with fake C function
            D .= (Γ * avg(H).^n.* ∇S.^(n-1)) .* (A.*avg(H).^(n-1) .+ (α*(n+2).*C_fake(MB[:,:,year],∇S))./(n-1)) 
        end
        #println("C_fake max: ", maximum(C_fake(MB, ∇S)))
        #println("C_fake min: ", minimum(C_fake(MB, ∇S)))

        # Compute flux components
        dSdx_edges = diff(S[:,2:end - 1], dims=1) / Δx
        dSdy_edges = diff(S[2:end - 1,:], dims=2) / Δy
        Fx .= .-avg_y(D) .* dSdx_edges
        Fy .= .-avg_x(D) .* dSdy_edges
        #  Flux divergence
        F .= .-(diff(Fx, dims=1) / Δx .+ diff(Fy, dims=2) / Δy) # MB to be added here 
        
        # To do: include 'method' and Δt as parameters of the function. 
        # Right now, this runs just as before.
        #method = "explicit-adaptive"

        if method == "explicit-adaptive"
            # Compute the maximum diffusivity in order to pick a temporal step that garantees estability 
            D_max = maximum(D)
            Δt = η * ( Δx^2 / (2 * D_max ))
        elseif method == "explicit"
            ## add this as parameter
            Δt = 0.001
            D_max = maximum(D)
            if Δt / ( Δx^2 / (2 * D_max )) > 1 
                println("Stability condition is not satisfied\n")
            end
        elseif method == "implicit"
            println("Implicit method not yet implemented\n")
        end
        append!(Δts, Δt)

        #  Update the glacier ice thickness
        # Only ice flux
        #dHdt = F .* Δt        
        # Ice flux + random mass balance  
        #MB = zeros(size(MB))  # test without mass balance              
        dHdt = (F .+ inn(MB[:,:,year])) .* Δt  
        global H[2:end - 1,2:end - 1] .= max.(0.0, inn(H) .+ dHdt)
        
        t += Δt
        # println("time: ", t)

        # Store timestamps to be used for training of the UDEs
        if ts_i < length(H_ref["timestamps"])+1
            if t >= H_ref["timestamps"][ts_i]
            println("Saving H at year ", H_ref["timestamps"][ts_i])
            push!(H_ref["H"], H)
            ts_i += 1
            end            
        end
        
    end 
    println("Saving reference data")
    save(joinpath(root_dir, "../../data/H_ref.jld"), "H_ref", H_ref)
end

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
    println("Y: ", size(Y))
    println("X: ", size(X))

    # We define an optimizer
    #opt = RMSProp(hyparams.η, 0.95)
    opt = ADAM(hyparams.η)

    # We get the model parameters to be trained
    ps_UA = Flux.params(UA)
    #data = Flux.Data.DataLoader((X, Y), batchsize=hyparams.batchsize, (X, Y), shuffle=false)
    data = Dict("X"=>X, "Y"=>Y)
    # Train the UDE for a given number of epochs
    @epochs hyparams.epochs hybrid_train!(loss, ps_UA, data, opt, H, p, t, t₁)
end

"""
    iceflow!(H,p,t,t₁)

Hybrid forward ice flow model combining the SIA PDE and neural networks with neural networks into an UDE
"""
function iceflow!(H, UA, p,t,t₁)

    # Retrieve input variables                    
    Δx, Δy, Γ, A, B, v, MB, ELAs, C, α = p
    ts_i = 1
    current_year = 0

    # Manual explicit forward scheme implementation
    while t < t₁

        # Get current year for MB and ELA
        year = floor(Int, t) + 1

        if(year != current_year)
            println("Year ", year)
            ŶA = UA(vec(buffer_mean(MB, year))')
            current_year = year
        end

        #println("t: ", t)

        # Update glacier surface altimetry
        S = B .+ H

        # All grid variables computed in a staggered grid
        # Compute surface gradients on edges
        dSdx  .= diff(S, dims=1) / Δx
        dSdy  .= diff(S, dims=2) / Δy
        ∇S .= sqrt.(avg_y(dSdx).^2 .+ avg_x(dSdy).^2)

        # Compute diffusivity on secondary nodes
        #                                     ice creep  +  basal sliding
        #D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (A.*avg(H) .+ (α*(n+2)*Uub)/(n-2)) 
        #println("MB buffer: ", buffer_mean(MB, year))
        
        #println("X: ", size(X))
        D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (avg(reshape(ŶA, size(H))) .* avg(H).^(n-1) .+ (α*(n+2)*C)/(n-2)) 

        # Compute flux components
        dSdx_edges = diff(S[:,2:end - 1], dims=1) / Δx
        dSdy_edges = diff(S[2:end - 1,:], dims=2) / Δy
        Fx .= .-avg_y(D) .* dSdx_edges
        Fy .= .-avg_x(D) .* dSdy_edges
        #  Flux divergence
        F .= .-(diff(Fx, dims=1) / Δx .+ diff(Fy, dims=2) / Δy) # MB to be added here 
            
        # Compute the maximum diffusivity in order to pick a temporal step that garantees estability 
        D_max = maximum(D)
        Δt = η * ( Δx^2 / (2 * D_max ))
        append!(Δts, Δt)

        #  Update the glacier ice thickness
        # Only ice flux
        #dHdt = F .* Δt        
        # Ice flux + mass balance                    
        dHdt = (F .+ inn(MB[:,:,year])) .* Δt  

        # Use Zygote.Buffer in order to mutate matrix while being able to compute gradients
        # TODO: Investigate how to correctly update H with Zygote.Buffer
        # Option 1
        H_buff = Zygote.Buffer(max.(0.0, inn(H) .+ dHdt))
        println("H_buff: ", size(H_buff))
        println("size(H_buff, 1): ", size(H_buff, 1))
        H_buff .= vcat(H_buff, Zygote.Buffer(zeros(size(H_buff, 1))))
        println("H_buff: ", size(H_buff))
        println("size(H_buff, 2): ", size(H_buff, 2))
        H_buff .= hcat(H_buff, Zygote.Buffer(zeros(size(H_buff, 2))))
        println("H_b: ", size(H_b))
        H = copy(H_buff)

        # Option 2 (incorrect)
        # H_buff = Zygote.Buffer(H)
        # H_buff[2:end - 1,2:end - 1] .= max.(0.0, inn(H_buff) .+ dHdt)
        # H = copy(H_buff)
        
        t += Δt
        # println("time: ", t)
        
    end     
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
function A_fake(MB_buffer)
    # Matching point MB values to A values
    MB_range = -15:0.5:10
    A_step = (1.57e-16-1.57e-17)/length(ELA_range)
    A_range = 1.57e-17:A_step:1.57e-16

    A = []
    for MB_i in MB_buffer
        push!(A, A_range[closest_index(MB_range, MB_i)])
    end
    
    return A
end
# function A_fake(ELA)
#     # Matching ELA values to A values
#     ELA_range = 2806:3470
#     A_step = (1.57e-16-1.57e-17)/length(ELA_range)
#     A_range = 1.57e-17:A_step:1.57e-16
    
#     return A_range[closest_index(ELA_range, ELA)]
# end

function create_NNs()
    ######### Define the network  ############
    # We determine the hyperameters for the training
    hyparams = Hyperparameters()

    # Leaky ReLu as activation function
    leakyrelu(x, a=0.01) = max(a*x, x)
    relu_acc(x) = min(max(0, x), 30)
    relu_abl(x) = min(max(0, x), 35)

    # Define the networks 1->5->5->1
    global UA = Chain(
        Dense(1,10,initb = Flux.zeros), 
        BatchNorm(10, leakyrelu),
        Dense(10,5,initb = Flux.zeros), 
        BatchNorm(5, leakyrelu),
        Dense(5,1, relu, initb = Flux.zeros)
    )

    return hyparams, UA
end

"""
    loss(batch)

Computes the loss function for a specific batch.
"""
# We determine the loss function
function loss(data, H, p, t, t₁)
    l_H, l_A = 0.0f0, 0.0f0
    num = 0

    # Make NN predictions
    w_pc=10e18
    for y in 1:size(data["X"])[3]
        ŶA = UA(vec(data["X"][:,:,y])')
        
        # Constrain A parameter within physically plausible values        
        l_A += sum(abs.(max.(ŶA .- 1.3e-24, 0).*w_pc)) + sum(abs.(min.(ŶA .- 0.5e-24, 0).*w_pc))
    end

    # Compute l_H as the difference between the simulated H with UA(x) and H_ref
    iceflow!(H, UA, p,t,t₁)
    l_H = sqrt(Flux.Losses.mse(H, H_ref["H"][end]; agg=mean))
    println("l_A: ", l_A)
    println("l_H: ", l_H)
    l = l_H + l_A

    return l
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

"""
    hybrid_train!(loss, ps_Up, ps_Ut, data, opt)

Train hybrid ice flow model based on UDEs.
"""

function hybrid_train!(loss, ps_UA, data, opt, H, p, t, t₁)
    # Retrieve model parameters
    ps_UA = Params(ps_UA)

    # back is a method that computes the product of the gradient so far with its argument.
    train_loss_UA, back_UA = Zygote.pullback(() -> loss(data, H, p, t, t₁), ps_UA)
    # Callback to track the training
    callback(train_loss_UA)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    gs_UA = back_UA(one(train_loss_UA))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
    Flux.update!(opt, ps_UA, gs_UA)
    # Here you might like to check validation set accuracy, and break out to do early stopping.
end
