#= Glacier mass balance toy model

Test with ideal data of a hybrid MB model based on neural networks
for the ablation and accumulation components that respect a classic
temperature-index model, mixed with model interpretation using 
SINDy (Brunton et al., 2016).

=#

## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("../."); Pkg.instantiate()
using Infiltrator
using Debugger

using Flux, DiffEqFlux
using Flux: @epochs
using Zygote
using Plots
gr()
using Base: @kwdef
using Statistics
using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra
# Set a random seed for reproduceable behaviour
using Random

#Random.seed!(2345)

@kwdef mutable struct Hyperparameters
    batchsize::Int = 500    # batch size
    η::Float64 = 0.1   # learning rate
    epochs::Int = 500        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
end

###############################################
############  FUNCTIONS   #####################
###############################################

# Fake quadratic response to snowfall and temperature 
α(P) = P.^(2)
μ(T) = T.^(2)

# Temperature-index equation for point glacier mass balance
function MB(P, T, Up, Ut)
    T_melt = 0
    MB = Up(P) - Ut(max.(T.-T_melt, 0))
    return MB
end

# Toy function to create artificial reference data
function toy_MB(α, μ, P, T)
    T_melt = 0.0
    MB = α(P) - μ(max.(T.-T_melt, 0.0))
    return MB
end

function train()

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

    # Callback to show the loss during training
    callback(l) = begin
        push!(losses, l)
        if length(losses)%50==0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        false
    end

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

    #########################################
    ##########################################

    ######### Define the network  ############
    # We determine the hyperameters for the training
    hyparams = Hyperparameters()

    # Leaky ReLu as activation function
    leakyrelu(x, a=0.01) = max(a*x, x)
    relu_acc(x) = min(max(0, x), 30)
    relu_abl(x) = min(max(0, x), 35)

    # Define the networks 1->5->5->5->1
    global Up = Chain(
        Dense(1,10,initb = Flux.zeros), 
        BatchNorm(10, leakyrelu),
        Dense(10,10,initb = Flux.zeros), 
        BatchNorm(10, leakyrelu),
        Dense(10,5,initb = Flux.zeros), 
        BatchNorm(5, leakyrelu),
        Dense(5,1, relu, initb = Flux.zeros)
    )

    global Ut = Chain(
        Dense(1,10,initb = Flux.zeros), 
        BatchNorm(10, leakyrelu),
        Dense(10,10,initb = Flux.zeros), 
        BatchNorm(10, leakyrelu),
        Dense(10,5,initb = Flux.zeros), 
        BatchNorm(5, leakyrelu),
        Dense(5,1, relu, initb = Flux.zeros)
    )

    # We define an optimizer
    #opt = RMSProp(hyparams.η, 0.95)
    opt = ADAM(hyparams.η)

    # We get the model parameters to be trained
    ps_Up = Flux.params(Up)
    ps_Ut = Flux.params(Ut)

    sqnorm(x) = sum(abs2, x)

    #######  We generate toy data to train the model  ########
    global snowfall_toy = rand(0.0f0:100.0f0, 500)
    global temperature_toy = 100f0*sin.((1:500)./20)
    forcings = hcat(snowfall_toy, temperature_toy)

    MB_toy = toy_MB(α, μ, snowfall_toy, temperature_toy)

    # Plot the toy dataset 
    l1 = @layout [a b; c; d]
    p1₁ = plot(1:500, snowfall_toy, label="Snowfall", color="midnightblue")
    p1₂ = plot(1:500, temperature_toy, label="Temperature", color="darkred")
    hline!(p1₂, [0], c="black", label="")
    p1₃ = plot(1:500, MB_toy, label="Mass balance")
    hline!(p1₃, [0], c="black", label="")

    X = vcat(snowfall_toy', temperature_toy')
    Y = collect(MB_toy')

    # Very important:
    # Shuffle batches and use batchsize similar to dataset size for easy training
    data = Flux.Data.DataLoader((X, Y), batchsize=hyparams.batchsize, (X, Y), shuffle=true)

    # We train the mass balance hybrid model
    @epochs hyparams.epochs hybrid_train!(loss, ps_Up, ps_Ut, data, opt)

    # Perform MB simulations with trained model
    global T_melt = 0
    global snow_norm = X[1,:]
    global temp_norm = X[2,:]
    global PDD_toy = max.(temp_norm.-T_melt, 0)
    global Ŷ_MB = MB(snow_norm', temp_norm', Up, Ut)
    global Ŷ_acc = Up(snow_norm')
    global Ŷ_abl = Ut(PDD_toy')

    # Plot comparison with toy data
    p1₄ = plot(1:500, Ŷ_MB', label="Simulated mass balance")
    hline!(p1₄, [0], c="black", label="")
    p1 = plot(p1₁,p1₂,p1₃, p1₄,layout=l1)

    l2 = @layout [a b;c d]
    p2₁ = plot(1:500, α(snowfall_toy')', label="α")
    plot!(1:500, Ŷ_acc', label="Up")
    hline!(p2₁, [0], c="black", label="")
    p2₂ = plot(1:500, α(snowfall_toy')'./snowfall_toy, label="α/P")
    plot!(1:500, Ŷ_acc'./snowfall_toy, label="Up/P")
    hline!(p2₂, [0], c="black", label="")

    p2₃ = plot(1:500, μ(PDD_toy')', label="μ", lw=2)
    plot!(1:500, Ŷ_abl', label="Ut", lw=2)
    hline!(p2₃, [0], c="black", label="")
    p2₄ = plot(1:500, μ(PDD_toy')'./PDD_toy, label="μ/T", lw=2)
    plot!(1:500, Ŷ_abl'./PDD_toy, label="Ut/T", lw=2)
    hline!(p2₄, [0], c="black", label="")

    p2 = plot(p2₁, p2₂, p2₃, p2₄, layout=l2)

    if(!ispath(pwd(),"plots"))
        mkdir("plots")
    end

    display(p1)
    display(p2)
    savefig(p1,joinpath(pwd(),"plots","hybrid_MB_model.png"))
    savefig(p2,joinpath(pwd(),"plots","hybrid_acc_abl.png"))

    ### Let's take a look at the raw learnt dynamics
    X_raw = 1:150
    Ŷp_raw = Up(X_raw')
    Yp_raw = α(X_raw')
    Ŷt_raw = Ut(X_raw')
    Yt_raw = μ(X_raw')

    l3 = @layout [a b]
    p3₁ = plot(X_raw, Ŷp_raw', lw=3, xlabel="Snowfall (P)", ylabel="Accumulation (α(P))",label="Up (inferred function)")
    plot!(X_raw, Yp_raw', l2=3, lw=3, label="α (true function")
    p3₂ = plot(X_raw, Ŷt_raw', lw=3, xlabel="Temperature (T)", ylabel="Ablation (μ(T))",label="Ut (inferred function)")
    plot!(X_raw, Yt_raw', l2=3, lw=3, label="μ (true function")
    p3 = plot(p3₁, p3₂, layout=l3)
    display(p3)
    savefig(p3,joinpath(pwd(),"plots","recovered_functions.png"))
end

###################################################################
########################  SINDy    ################################
###################################################################

function sindy()

    ## Symbolic regression via sparse regression ( SINDy based )
    # Create a Basis
    Xp = Float64.(copy(reshape(snowfall_toy, (1,length((snowfall_toy))))))
    Xt = max.(temperature_toy.-T_melt, 0)
    @variables X_P[1:1]

    # Generate the basis functions, multivariate polynomials up to deg 5
    b = [polynomial_basis(X_P, 5);]
    basis_p = Basis(b, X_P)

    # Create an optimizer for the SINDy problem
    λ, ρ = 9999e-1, 5.0
    opt = ADMM(λ, ρ)

    #opt = STRRidge(0.1)# Create the thresholds which should be used in the search process

    # opt = SR3(9e-1, 1.0)
    # λs = exp10.(-5:0.1:-1)

    # Test on ideal data
    println("Accumulation:")
    println("SINDy on ideal data \n")
    Ψ_p_true = SINDy(Xp, α(snow_norm'), basis_p, opt, maxiter = 1000) # Succeed
    #Ψ_true = SINDy(Float32.(reshape(X_raw, (1, length(X_raw)))), Float32.(Yp_raw), basis_p, opt, maxiter = 100, normalize = true)
    println(Ψ_p_true)
    print_equations(Ψ_p_true, show_parameter=true)
    p̂_p_true = parameters(Ψ_p_true)
    println("Ψ_true Parameter guess : $(p̂_p_true) \n")

    println("SINDy on synthetic range predicted data \n")
    syn_range = 1:100
    Ψ_p_range = SINDy(Float64.(syn_range), Float64.(Up(syn_range')), basis_p, opt, p=[0], maxiter = 20000, denoise=true, normalize=true)
    println(Ψ_p_range)
    print_equations(Ψ_p_range)
    p̂_p_range = parameters(Ψ_p_range)
    println("Ψ First parameter guess : $(p̂_p_range)")

    println("SINDy on predicted data \n")
    Ψ_p = SINDy(Xp, Float64.(Ŷ_acc), basis_p, opt, p=[0], maxiter = 200000, denoise=false, normalize=true, convergence_error = 1e-8)
    println(Ψ_p)
    print_equations(Ψ_p)
    p̂_p = parameters(Ψ_p)
    println("Ψ First parameter guess : $(p̂_p) \n \n")

    println("Ablation: ")
    println("SINDy on ideal data \n")
    Ψ_t_true = SINDy(Xt, μ(temp_norm'), basis_p, opt, maxiter = 1000) # Succeed
    #Ψ_true = SINDy(Float32.(reshape(X_raw, (1, length(X_raw)))), Float32.(Yp_raw), basis_p, opt, maxiter = 100, normalize = true)
    println(Ψ_t_true)
    print_equations(Ψ_t_true, show_parameter=true)
    p̂_t_true = parameters(Ψ_t_true)
    println("Ψ_true Parameter guess : $(p̂_t_true) \n")

    println("SINDy on synthetic range predicted data \n")
    syn_range = 1:100
    Ψ_t_range = SINDy(Float64.(syn_range), Float64.(Ut(syn_range')), basis_p, opt, p=[0], maxiter = 20000, denoise=true, normalize=true)
    println(Ψ_t_range)
    print_equations(Ψ_t_range)
    p̂_t_range = parameters(Ψ_t_range)
    println("Ψ First parameter guess : $(p̂_t_range)")

    println("SINDy on predicted data \n")
    Ψ_t = SINDy(Xt, Float64.(Ŷ_abl), basis_p, opt, p=[0], maxiter = 200000, denoise=false, normalize=true, convergence_error = 1e-8)
    println(Ψ_t)
    print_equations(Ψ_t)
    p̂_t = parameters(Ψ_t)
    println("Ψ First parameter guess : $(p̂_t) \n")
end


###############################################################
###########################  MAIN #############################
###############################################################

train()

#sindy()