###################################################################
###   Functions for PDE solving in staggered grids with UDEs   ####
###################################################################

include("utils.jl")

"""
    iceflow_toy!(H,p,t,t₁)

Forward ice flow model solving the Shallow Ice Approximation PDE 
"""
function iceflow_toy!(H,p,t,t₁)

    println("Running forward PDE ice flow model...\n")
    # Retrieve input variables                    
    Δx, Δy, Γ, A, B, v, MB, ELAs, C, α = p

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
        #                                     ice creep  +  basal sliding
        #D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (A.*avg(H).^(n-1) .+ (α*(n+2)*C)/(n-2)) 

        # Diffusivity with fake A function
        D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (A_fake(buffer_mean(ELAs, year)) .* avg(H).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
        
        #D .= (Γ * avg(H).^n.* ∇S.^(n-1)) .* (A.*avg(H).^(n-1) .+ (α*(n+2).*C_fake(MB[:,:,year],∇S))./(n-1)) 

        #println("C_fake max: ", maximum(C_fake(MB, ∇S)))
        #println("C_fake min: ", minimum(C_fake(MB, ∇S)))

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
        # Ice flux + random mass balance  
        #MB = zeros(size(MB))  # test without mass balance              
        dHdt = (F .+ inn(MB[:,:,year])) .* Δt  
        global H[2:end - 1,2:end - 1] .= max.(0.0, inn(H) .+ dHdt)
        
        t += Δt
        # println("time: ", t)
        
    end 
end

"""
    iceflow!(H,p,t,t₁)

Hybrid forward ice flow model combining the SIA PDE and neural networks into 
Universal Differential Equations (UDEs)
"""
function iceflow!(H,Uub, p,t,t₁)

    println("Running forward UDE ice flow model...\n")
    # Retrieve input variables                    
    Δx, Δy, Γ, A, B, v, MB, C, α = p

    # Manual explicit forward scheme implementation
    while t < t₁

        # Update glacier surface altimetry
        S = B .+ H

        # Random mass balance year
        y_rand = rand(1:97)

        # All grid variables computed in a staggered grid
        # Compute surface gradients on edges
        dSdx  .= diff(S, dims=1) / Δx
        dSdy  .= diff(S, dims=2) / Δy
        ∇S .= sqrt.(avg_y(dSdx).^2 .+ avg_x(dSdy).^2)

        # Compute diffusivity on secondary nodes
        #                                     ice creep  +  basal sliding
        D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (A.*avg(H) .+ (α*(n+2)*Uub)/(n-2)) 

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
        # Ice flux + random mass balance                      
        dHdt = (F .+ inn(MB[:,:,year])) .* Δt  
        global H[2:end - 1,2:end - 1] .= max.(0.0, inn(H) .+ dHdt)
        
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
function A_fake(ELA)
    # Matching ELA values to A values
    ELA_range = 2806:3470
    A_step = (1.57e-16-1.57e-17)/length(ELA_range)
    A_range = 1.57e-17:A_step:1.57e-16
    
    return A_range[closest_index(ELA_range, ELA)]
end

"""
    closest_index(x, val)

Return the index of the closest Array element
"""
function closest_index(x, val)
    ibest = eachindex(x)[begin]
    dxbest = abs(x[ibest]-val)
    for I in eachindex(x)
        dx = abs(x[I]-val)
        if dx < dxbest
            dxbest = dx
            ibest = I
        end
    end
    return ibest
    end 

"""
    buffer_mean(A, i)

Perform the mean of the last 5 elements of an Array
"""
function buffer_mean(A, i)
    if(i-5 < 1)
        j = 1
    else
        j = i-5
    end
    return mean(A[j:i])
end

function create_NNs()
    ######### Define the network  ############
    # We determine the hyperameters for the training
    hyparams = Hyperparameters()

    # Leaky ReLu as activation function
    leakyrelu(x, a=0.01) = max(a*x, x)
    relu_acc(x) = min(max(0, x), 30)
    relu_abl(x) = min(max(0, x), 35)

    # Define the networks 1->5->5->5->1
    global UA = Chain(
        Dense(1,10,initb = Flux.zeros), 
        BatchNorm(10, leakyrelu),
        Dense(10,10,initb = Flux.zeros), 
        BatchNorm(10, leakyrelu),
        Dense(10,5,initb = Flux.zeros), 
        BatchNorm(5, leakyrelu),
        Dense(5,1, relu, initb = Flux.zeros)
    )

    return hyparams, Uub
end

"""
    loss(batch)

Computes the loss function for a specific batch.
"""
# We determine the loss function
function loss(batch)
    l, l_acc, l_abl = 0.0f0, 0.0f0, 0.0f0
    num = 0
    for (x, y) in batch

        # Make NN predictions
        p_batch = x[1,:]'
        t_batch = x[2,:]'
        pdd_batch = max.(t_batch.-0, 0)
        Ŷp = Up(p_batch)
        Ŷt = Ut(pdd_batch)
        
        # We evaluate the MB as the combination of Accumulation - Ablation         
        w_pc=1000
        l_MB = sqrt(Flux.Losses.mse(MB(p_batch, t_batch, Up, Ut), y; agg=mean))
        l_range_acc = sum((max.((Ŷp/p_batch).-110, 0)).*w_pc)
        l_range_abl = sum((max.((Ŷt/pdd_batch).-110, 0)).*w_pc)

        #l += l_MB 
        l += l_MB + l_range_acc + l_range_abl
        l_acc += l_range_acc
        l_abl += l_range_abl
        num +=  size(x, 2)

        # println("Accumulation loss: ", l_range_acc)
        # println("Ablation loss: ", l_range_abl)

    end

    return l/num
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
function hybrid_train!(loss, ps_Up, ps_Ut, data, opt)
    # Retrieve model parameters
    ps_Up = Params(ps_Up)
    ps_Ut = Params(ps_Ut)

    for batch in data
    # back is a method that computes the product of the gradient so far with its argument.
    train_loss_Up, back_Up = Zygote.pullback(() -> loss(batch), ps_Up)
    train_loss_Ut, back_Ut = Zygote.pullback(() -> loss(batch), ps_Ut)
    # Callback to track the training
    callback(train_loss_Up)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    gs_Up = back_Up(one(train_loss_Up))
    gs_Ut = back_Ut(one(train_loss_Ut))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
    Flux.update!(opt, ps_Up, gs_Up)
    Flux.update!(opt, ps_Ut, gs_Ut)
    # Here you might like to check validation set accuracy, and break out to do early stopping.
    end
end
