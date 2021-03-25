## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

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

###############################################
############  FUNCTIONS   #####################
###############################################

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

  
@kwdef mutable struct Hyperparameters
    batchsize::Int = 500    # batch size
    η::Float64 = 0.05    # learning rate
    epochs::Int = 1000        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
end

###############################################################
###########################  MAIN #############################
###############################################################

#function main()

######### Define the network  ############
# We determine the hyperameters for the training
hyparams = Hyperparameters()

# Leaky ReLu as activation function
leakyrelu(x, a=0.01) = max(a*x, x)
relu_acc(x) = min(max(0, x), 30)
relu_abl(x) = min(max(0, x), 35)

# Define the networks 1->5->5->5->1
Up = Chain(
    Dense(1,10,initb = Flux.zeros), 
    BatchNorm(10, leakyrelu),
    Dense(10,10,initb = Flux.zeros), 
    BatchNorm(10, leakyrelu),
    Dense(10,5,initb = Flux.zeros), 
    BatchNorm(5, leakyrelu),
    Dense(5,1, relu, initb = Flux.zeros)
)

Ut = Chain(
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
snowfall_toy = rand(0.0f0:100.0f0, 500)
temperature_toy = 100f0*sin.((1:500)./20)
forcings = hcat(snowfall_toy, temperature_toy)

α(P) = P.^(2)
μ(T) = T.^(2)
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
T_melt = 0
snow_norm = X[1,:]
temp_norm = X[2,:]
PDD_toy = max.(temp_norm.-T_melt, 0)
Ŷ_MB = MB(snow_norm', temp_norm', Up, Ut)
Ŷ_acc = Up(snow_norm')
Ŷ_abl = Ut(PDD_toy')

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
X_raw = 1:100
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

###################################################################
########################  SINDy    ################################
###################################################################

## Symbolic regression via sparse regression ( SINDy based )
# Create a Basis
Xp = Float32.(copy(reshape(snowfall_toy, (1,length((snowfall_toy))))))
Xt = max.(temperature_toy.-T_melt, 0)
@variables X_P[1:1]

# Generate the basis functions, multivariate polynomials up to deg 5
# and sine
b = [polynomial_basis(X_P, 5);]
basis_p = Basis(b, X_P)

# Create an optimizer for the SINDy problem
#opt = STRRidge(0.1)# Create the thresholds which should be used in the search process

# opt = SR3(Float32(1e-2), Float32(1e-2))
# # Create the thresholds which should be used in the search process
# λ = Float32.(exp10.(0:0.1:100))
# # Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
# g(x) = x[1] < 1 ? Inf : norm(x, 2)


# Test on ideal derivative data for unknown function ( not available )
println("SINDy on ideal data \n")
#Ψ_true = SINDy(Xp, Float32.(α(snow_norm')), basis_p, λ, opt, g = g, maxiter = 10000) # Succeed
Ψ_true = SINDy(Xp, Float32.(α(snow_norm')), basis_p, opt, maxiter = 100, normalize = true)
println(Ψ_true)
print_equations(Ψ_true)
p̂_true = parameters(Ψ_true)
println("Ψ_true Parameter guess : $(p̂_true) \n")

println("SINDy on predicted data \n")
Ψ = SINDy(Xp, Up(snow_norm'), basis_p, opt, maxiter = 100, normalize = true, denoise = true)
println(Ψ)
print_equations(Ψ)
p̂ = parameters(Ψ)
println("Ψ Parameter guess : $(p̂)")

# Ψ = SINDy(X̂, Ȳ, basis, λ, opt, g = g, maxiter = 10000) # Succeed
# println(Ψ)
# print_equations(Ψ)

# # Test on uode derivative data
# println("SINDy on learned, partial, available data")
# Ψ = SINDy(X̂, Ŷ, basis, λ,  opt, g = g, maxiter = 10000, normalize = true, denoise = true) # Succeed
# println(Ψ)
# print_equations(Ψ)

# # Extract the parameter
# p̂ = parameters(Ψ)
# println("First parameter guess : $(p̂)")

# # Just the equations
# b = Basis((u, p, t)->Ψ(u, [1.; 1.], t), u)

# # Retune for better parameters -> we could also use DiffEqFlux or other parameter estimation tools here.
# Ψf = SINDy(X̂, Ŷ, b, STRRidge(0.01), maxiter = 100, convergence_error = 1e-18) # Succeed
# println(Ψf)
# p̂ = parameters(Ψf)
# println("Second parameter guess : $(p̂)")
# println("Overall parameter guess : $(abs.([p̂; p_trained[1]]))")
# println("True paramter : $(p_[2:end])")


# #main()