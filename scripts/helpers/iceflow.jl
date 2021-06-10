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

    # We get the model parameters to be trained
    ps_UA = Flux.params(UA)
    #data = Flux.Data.DataLoader((X, Y), batchsize=hyparams.batchsize, (X, Y), shuffle=false)
    data = Dict("X"=>X, "Y"=>Y)
    # Train the UDE for a given number of epochs
    #H_buff = Zygote.Buffer(H)
    @epochs hyparams.epochs hybrid_train!(loss, ps_UA, data, opt, H, p, t, t₁)

    #H = copy(H_buff)  
end

"""
    hybrid_train!(loss, ps_Up, ps_Ut, data, opt)

Train hybrid ice flow model based on UDEs.
"""
function hybrid_train!(loss, ps_UA, data, opt, H, p, t, t₁)
    # Retrieve model parameters
    ps_UA = Params(ps_UA)

    # back is a method that computes the product of the gradient so far with its argument.
    println("Forward pass")
    train_loss_UA, back_UA = Zygote.pullback(() -> loss(data, H, p, t, t₁), ps_UA)
    # Callback to track the training
    callback(train_loss_UA)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    println("Backpropagation")
    @time gs_UA = back_UA(one(train_loss_UA))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
    @infiltrate
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
    #w_pc=10e18
    w_pc = 1
    for y in 1:size(data["X"])[3]
        ŶA = UA(vec(data["X"][:,:,y])')
        
        # Constrain A parameter within physically plausible values        
        l_A += sum(abs.(max.(ŶA .- 1.3e-24, 0).*w_pc)) + sum(abs.(min.(ŶA .- 0.5e-24, 0).*w_pc))
    end

    # Compute l_H as the difference between the simulated H with UA(x) and H_ref
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
    MB = p[7]  
    current_year = 0
    total_iter = 0
    ts_i = 1

    # Manual explicit forward scheme implementation
    while t < t₁
        let
        iter = 1
        err = 2 * tolnl
        H_ = copy(H)
        #dHdt = zeros(nx, ny) # we need to define dHdt for iter = 1
        dHdt = zeros(nx-2, ny-2) # we need to define dHdt for iter = 1

        # Get current year for MB and ELA
        year = floor(Int, t) + 1
        if(year != current_year)
            println("Year ", year)
            current_year = year
        end
        y = (year, current_year)
        ts = (t, ts_i)

        my_method = "implicit" # To do: move this to parameters.jl
        #my_method = "explicit"

        if my_method == "explicit"

            F, dτ, current_year = SIA_old(H, p, y)
            Δt_exp = 0.0001 # Hardcoded. Based on the JupyterNotebook, this value should give stable results
            inn(H) .= max.(0.0, inn(H) .+ Δt_exp * F)
            # H .= max.(0.0, H .+ Δt_exp * F)
            #println("Fmax: ", maximum(abs.(F)))
            t += Δt_exp
            total_iter += 1 

        elseif my_method == "implicit"

            while err > tolnl && iter < itMax
            
                Err = copy(H)

                # Compute the Shallow Ice Approximation in a staggered grid
                #F, dτ, current_year = SIA(H, p, y)
                F, dτ, current_year = SIA_old(H, p, y)
                
                # Implicit method
                # Differentiate H via a Picard iteration method
                # Change update rule for SIA_old:
                #ResH = -(H .- H_)/Δt .+ F 
                #dHdt = damp .* dHdt .+ ResH
                ResH = -(inn(H) .- inn(H_))/Δt .+ F 
                dHdt = damp .* dHdt .+ ResH

                # Assert that dHdt has zeros on the borders
                #for i in 1:nx
                #    if dHdt[i, 1] != 0 || dHdt[i,ny] != 0
                #        println("Error: border of dHdt must be zero for boundary condition")
                #    end
                #end
                #for j in 1:ny
                #    if dHdt[1, j] != 0 || dHdt[nx, j] != 0
                #        println("Error: border of dHdt must be zero for boundary condition")
                #    end
                #end
                
                # println("F: ", maximum(F))
                # println("ResH: ", maximum(ResH))
                #println("dτ min: ", minimum(dτ))
                #println("dτ max: ", maximum(dτ))
                # println("dHdt: ", maximum(dHdt))
                # println("H: ", maximum(H))

                # Update the ice thickness
                #H .= max.(0.0, H .+ dτ .* dHdt)
                inn(H) .= max.(0.0, inn(H) .+ dτ .* dHdt)
                
                if mod(iter, nout) == 0
                    # Compute error for implicit method with damping
                    # There is no need to use a function for this, since Err is redefined inside the second while
                    Err = Err .- H
                    err = maximum(Err)
                    #err = computeErr!(Err, H)

                    println(" iter = $iter, error = $err \n")
                    if isnan(err)
                        error("""NaNs encountered.  Try a combination of:
                                    decreasing `damp` and/or `dtausc`, more smoothing steps""")
                    end
                end
            
                iter += 1
                total_iter += 1

                if total_iter == 50
                    println("Break!!!")
                    #break #REMOVE
                end

            end

            #break#REMOVE

            t += Δt

        end
        end

        # Store timestamps to be used for training of the UDEs
        if ts_i < length(H_ref["timestamps"])+1
            if t >= H_ref["timestamps"][ts_i]
                println("Saving H at year ", H_ref["timestamps"][ts_i])
                push!(H_ref["H"], H)
                ts_i += 1

                #hm3 = heatmap(H, c = :ice, title="Ice thickness (t=$t)")
                #display(hm3)  
            end          
        end

        #hm3 = heatmap(H, c = :ice, title="Ice thickness (t=$t)")
        #display(hm3)  

         
    end 

    println("Total Number of iterartions: ", total_iter)
    end
    
    println("Saving reference data")
    save(joinpath(root_dir, "../../data/H_ref.jld"), "H_ref", H_ref)
end


"""
    iceflow!(H,p,t,t₁)

Hybrid forward ice flow model combining the SIA PDE and neural networks with neural networks into an UDE
"""
function iceflow!(H, UA::Chain, p,t,t₁)

    # # Retrieve input variables  
    let                  
    current_year = 0
    MB = p[7]
    dHdt_ = zeros(nx, ny)
    H_ = zeros(nx, ny)
    # TODO: uncomment this once we have the pullbacks working and we're ready to train an UDE
    global model = "UDE_A"

    # Manual explicit forward scheme implementation
    while t < t₁

        #println("time: ", t)

        # Get current year for MB and ELA
        year = floor(Int, t) + 1
        if(year != current_year)
            println("Year ", year)
            # Predict A with the NN
            ŶA = UA(vec(buffer_mean(MB, year))')
            # Unpack and repack tuple with updated A value
            Δx, Δy, Γ, A, B, v, MB, ELAs, C, α = p
            p = (Δx, Δy, Γ, ŶA, B, v, MB, ELAs, C, α)
            current_year = year
        end
        y = (year, current_year)

        # Compute ice flux following the Shallow Ice Approximation PDE
        F, dτ, current_year = SIA(H, p, y)

        # Compute the residual ice thickness for the inertia
        @infiltrate
        @tullio ResH := -(H[i,j] - H_[i,j])/Δt + F[i,j] + MB[i,j,year]
        @tullio dHdt[i,j] := dHdt_[i,j]*damp + ResH[i,j]
        dHdt_ = copy(dHdt)

        println("dHdt: ", maximum(dHdt))
        # Update the ice thickness
        @tullio H[i,j] := max.(0.0, H_[i,j] .+ dHdt[i,j]*dτ)
        H_ = copy(H)
        #push!(Ht, dHdt)
        
        #@infiltrate
        t += Δt
        #append!(Δts,t)
        
    end   
    end
end

function computeErr!(Err, H)
    Err .= Err .- H # Telescopic sum: always equal to H_now - H_last
    #err = norm(Err) / length(Err) # cuadratic error
    err = maximum(Err) # Maximum error

    return err
end

"""
    SIA(H, p, y)

Compute a step of the Shallow Ice Approximation PDE in a forward model
"""
function SIA(H, p, y)
    Δx, Δy, Γ, A, B, v, MB, ELAs, C, α = p
    year, current_year = y

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    #dSdx  = diff(padx(S), dims=1) / Δx
    #dSdy  = diff(pady(S), dims=2) / Δy
    #∇S = sqrt.(avg_y(pady(dSdx)).^2 .+ avg_x(padx(dSdy)).^2)
    # Swich order of padding and diff:
    dSdx  = padx(diff(S, dims=1)) / Δx
    dSdy  = pady(diff(S, dims=2)) / Δy
    ∇S = sqrt.(pady(avg_y(dSdx)).^2 .+ padx(avg_x(dSdy)).^2)

    # Compute diffusivity on secondary nodes
    if model == "standard"
        # TODO: investigate instabilities on too large D numbers
        #                                     ice creep  +  basal sliding
        #D = (avg(pad(H)).^n .* ∇S.^(n - 1)) .* (A.*(avg(pad(H))).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
        Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s # Γ from Jupyter notebook. To do: standarize this value in patameters.jl
        # Swich order of padding and average:
        #D = Γ * avg(pad(H)).^(n + 2) .* ∇S.^(n - 1) # PROBLEM WITH ORDIN OF PADDING + AVg?
        D = Γ * pad(avg(H)).^(n + 2) .* ∇S.^(n - 1) # PROBLEM WITH ORDIN OF PADDING + AVg?
    
    elseif model == "fake A"
        D = (Γ * avg(H).^n .* ∇S.^(n - 1)) .* (avg(A_fake(buffer_mean(MB, year), size(H))) .* avg(H).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
    elseif model == "fake C"
        # Diffusivity with fake C function
        D = (Γ * avg(H).^n .* ∇S.^(n - 1)) .* (A.*avg(H).^(n-1) .+ (α*(n+2).*C_fake(MB[:,:,year],∇S))./(n-1)) 
    elseif model == "UDE_A"
        # A here should have the same shape than H
        D = (Γ * avg(pad(H)).^n .* ∇S.^(n - 1)) .* (avg(pad(reshape(A, size(H)))) .* avg(pad(H)).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
    else 
        error("Model $model is incorrect")
        #throw(DomainError())
    end

    # Compute flux components
    #Fx = .-avg_y(pady(D)) .* dSdx
    #Fy = .-avg_x(padx(D)) .* dSdy
    #  Flux divergence
    #F = .-(diff(padx(Fx), dims=1) / Δx .+ diff(pady(Fy), dims=2) / Δy)
    # Swich the order of padding and average:
    Fx = .-pady(avg_y(D)) .* dSdx
    Fy = .-padx(avg_x(D)) .* dSdy
    #  Flux divergence
    F = .-(padx(diff(Fx, dims=1)) / Δx .+ pady(diff(Fy, dims=2)) / Δy)
    
    # Update or set time step for temporal discretization
    #Δt = timestep!(Δts, Δx, D, method)  # explicit-adaptive timestep
    # Swich padding and average:
    #dτ = dτsc.*min.(10.0, 1.0./(1.0/Δt .+ 1.0./(cfl./(ϵ .+ avg(pad(D))))))  # semi-implicit timestep with damping
    dτ = dτsc.*min.(10.0, 1.0./(1.0/Δt .+ 1.0./(cfl./(ϵ .+ pad(avg(D))))))  # semi-implicit timestep with damping

    return F, dτ, current_year
end


function SIA_old(H, p, y)
    Δx, Δy, Γ, A, B, v, MB, ELAs, C, α = p
    year, current_year = y

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx  = diff(S, dims=1) / Δx
    dSdy  = diff(S, dims=2) / Δy
    ∇S = sqrt.(avg_y(dSdx).^2 .+ avg_x(dSdy).^2)

    # Compute diffusivity on secondary nodes
    Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s # Γ from Jupyter notebook
    D = Γ * avg(H).^(n + 2) .* ∇S.^(n - 1)
    #println(maximum(D))

    # Compute flux components
    dSdx_edges = diff(S[:,2:end - 1], dims=1) / Δx
    dSdy_edges = diff(S[2:end - 1,:], dims=2) / Δy
    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges
    

    #  Flux divergence
    F = .-(diff(Fx, dims=1) / Δx .+ diff(Fy, dims=2) / Δy) # MB to be added here 
    
    # hmf = heatmap(∇S.^(n - 1))
    # display(hmf)

    # Compute the maximum diffusivity in order to pick a temporal step that garantees estability 
    #D_max = maximum(D)
    #Δτ = η * ( Δx^2 / (2 * D_max ))
    dτ = dτsc * min.( 10.0 , 1.0./(1.0/Δt .+ 1.0./(cfl./(ϵ .+ avg(D)))))

    return F, dτ, current_year

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

