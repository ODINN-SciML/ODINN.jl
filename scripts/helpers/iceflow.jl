###################################################################
###   Functions for PDE solving in staggered grids with UDEs   ####
###################################################################

using Zygote
using PaddedViews
using Flux
using Flux: @epochs
using DiffEqOperators
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

    # We get the model parameters to be trained
    ps_UA = Flux.params(UA)
    #data = Flux.Data.DataLoader((X, Y), batchsize=hyparams.batchsize, (X, Y), shuffle=false)
    data = Dict("X"=>X, "Y"=>Y)
    # Train the UDE for a given number of epochs
    H_buff = Zygote.Buffer(H)
    @epochs hyparams.epochs hybrid_train!(loss, ps_UA, data, opt, H_buff, p, t, t₁)

    H = copy(H_buff)  
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
    println("Forward pass")
    iceflow!(H, UA, p,t,t₁)
    #println("H: ", maximum(H))
    #@infiltrate
    l_H = sqrt(Flux.Losses.mse(H, H_ref["H"][end]; agg=mean))
    println("l_A: ", l_A)
    println("l_H: ", l_H)
    l = l_H + l_A

    return l
end


"""
    iceflow!(H,p,t,t₁)

Forward ice flow model solving the Shallow Ice Approximation PDE 
"""
function iceflow!(H,H_ref::Dict, p,t,t₁)

    println("Running forward PDE ice flow model...\n")
    # Instantiate variables
    let             
    Δx, Δy, Γ, A, B, v, MB, ELAs, C, α = p    
    current_year = 0
    ts_i = 1

    # Manual explicit forward scheme implementation
    while t < t₁

        # Get current year for MB and ELA
        year = floor(Int, t) + 1
        if(year != current_year)
            println("Year ", year)
            current_year = year
        end
        y = (year, current_year)
        ts = (t, ts_i)

        Δt, current_year, ts_i = update_store_H(H, H_ref, p, y, ts)
             
        t += Δt
        # println("Δt: ", Δt)
        # println("time: ", t)
         
    end 
    println("Saving reference data")
    save(joinpath(root_dir, "../../data/H_ref.jld"), "H_ref", H_ref)
    end
end


"""
    iceflow!(H,p,t,t₁)

Hybrid forward ice flow model combining the SIA PDE and neural networks with neural networks into an UDE
"""
function iceflow!(H, UA::Chain, p,t,t₁)

    # # Retrieve input variables  
    let                  
    current_year = 0
    # TODO: uncomment this once we have the pullbacks working and we're ready to train an UDE
    #global model = "UDE_A"

    # Manual explicit forward scheme implementation
    while t < t₁

        #println("time: ", t)

        # Get current year for MB and ELA
        year = floor(Int, t) + 1
        if(year != current_year)
            println("Year ", year)
            # TODO: uncomment once we're ready to train UDEs 
            # ŶA = UA(vec(buffer_mean(MB, year))')
            # # Unpack and repack tuple with updated A value
            # Δx, Δy, Γ, A, B, v, MB, ELAs, C, α = p
            # p = (Δx, Δy, Γ, ŶA, B, v, MB, ELAs, C, α)
            current_year = year
        end
        y = (year, current_year)

        Δt, current_year = update_H(H, p, y)
             
        t += Δt
        
    end   
    end
end

"""
    update_store_H(H, H_ref, p, y, ts_i)

Update the ice thickness by differentiating H based on the Shallow Ice Approximation and
create and store dataset to be used as a reference
"""
function update_store_H(H, H_ref, p, y, ts)

    # Compute the Shallow Ice Approximation in a staggered grid
    Δt, current_year = SIA!(H, p, y)
    t, ts_i = ts
    
    # Store timestamps to be used for training of the UDEs
    if ts_i < length(H_ref["timestamps"])+1
        if t >= H_ref["timestamps"][ts_i]
            println("Saving H at year ", H_ref["timestamps"][ts_i])
            push!(H_ref["H"], H)
            ts_i += 1
        end            
    end

    return Δt, current_year, ts_i
    
end 

"""
    update_H(H, p, y, Δ)

Update the ice thickness by differentiating H based on the Shallow Ice Approximation 
"""
function update_H(H, p, y)
    # Compute the Shallow Ice Approximation in a staggered grid
    Δt, current_year = SIA!(H, p, y)

    return Δt, current_year
    
end 

"""
    SIA!(H, p, y)

Compute a step of the Shallow Ice Approximation PDE in a forward model
"""
function SIA!(H, p, y)
    Δx, Δy, Γ, A, B, v, MB, ELAs, C, α = p
    year, current_year = y

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx  .= diff(padx(S), dims=1) / Δx
    dSdy  .= diff(pady(S), dims=2) / Δy
    ∇S .= sqrt.(avg_y(pady(dSdx)).^2 .+ avg_x(padx(dSdy)).^2)

    #@infiltrate

    # Compute diffusivity on secondary nodes
    if model == "standard"
        #                                     ice creep  +  basal sliding
        D .= (Γ * avg(pad(copy(H))).^n.* ∇S.^(n - 1)) .* (A.*avg(pad(copy(H))).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
    elseif model == "fake A"
        D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (avg(A_fake(buffer_mean(MB, year), size(H))) .* avg(H).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
    elseif model == "fake C"
        # Diffusivity with fake C function
        D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (A.*avg(H).^(n-1) .+ (α*(n+2).*C_fake(MB[:,:,year],∇S))./(n-1)) 
    elseif model == "UDE_A"
        # A here should have the same shape than H
        D .= (Γ * avg(H).^n.* ∇S.^(n - 1)) .* (avg(reshape(A, size(H))) .* avg(H).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
    else 
        println("ERROR: Model $model is incorrect")
        #throw(DomainError())
    end

    # Compute flux components
    Fx .= .-avg_y(pady(D)) .* dSdx
    Fy .= .-avg_x(padx(D)) .* dSdy
    #  Flux divergence
    F .= .-(diff(padx(Fx), dims=1) / Δx .+ diff(pady(Fy), dims=2) / Δy) 
    
    # Update or set time step for temporal discretization
    Δt = timestep!(Δts, Δx, D, method)

    #  Update the glacier ice thickness          
    dHdt .= (F .+ MB[:,:,year]) .* Δt  

    # Use Zygote.Buffer in order to mutate matrix while being able to compute gradients
    # TODO: Investigate how to correctly update H with Zygote.Buffer
    # Option 1
    #dH = max.(0.0, inn(H) .+ dHdt)
    #println("size(dH): ", size(dH))
    #H = PaddedView(0.0, Zygote.Buffer(dH), size(H), (2,2))
    #println("size(H): ", size(H))
    # println("H: ", size(H))
    # println("size(H_buff_0, 1): ", size(H_buff_0, 1))
    # H_buff_1 = vcat(H_buff_0, Zygote.Buffer(zeros(size(H_buff_0, 1))))
    # println("size(H_buff_1): ", size(H_buff_1))
    # H_buff = hcat(H_buff_1, Zygote.Buffer(zeros(size(H_buff_1, 2))))
    # println("H_buff: ", size(H_buff))
    #H = copy(H_buffpad)

    # Option 2 (incorrect)
    
    # println("H: ", size(H))
    # println("dHdt: ", size(dHdt))
    # for i in 1:size(dHdt,1)
    #     for j in 1:size(dHdt,2)
    #         H[i+1,j+1] = max(0.0, H[i+1,j+1] + dHdt[i,j])
    #     end
    # end

    H .= max.(0.0, H .+ dHdt)
    
    return Δt, current_year
end

# function rrule(::typeof(update_H), H, UA, year, current_year)
#     function update_pullback(H, UA)
#         b̄ = zero(ā)
#         for i in 1:length(a)
#             for d in 0:m-1
#                 if i-d > 0
#                     b̄[i] += (m-d) * ā[i-d]
#                 end
#             end
#         end
#         return NO_FIELDS, b̄, DoesNotExist()
#     end
#     return updatestate(a, m), update_pullback
# end;

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

