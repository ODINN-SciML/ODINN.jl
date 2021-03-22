## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Flux, DiffEqFlux
using Flux: @epochs
using Zygote
using Plots
gr()
using Statistics
# Set a random seed for reproduceable behaviour
using Random

#Random.seed!(2345)

###############################################
############  FUNCTIONS   #####################
###############################################

# Temperature-index equation for point glacier mass balance
function MB!(MB, forcings, Up, Ut)
    T_melt = 0.0
    P, T = forcings
    MB = Up(P) - Ut(max(T-T_melt, 0.0))
end

# Toy function to create artificial reference data
function toy_MB(α, μ, P, T)
    T_melt = 0.0

    #print("forcings: ", forcings)
    print("P: ", P, "\n")
    print("T: ",T)
    MB = α(P) - μ(max.(T.-T_melt, 0.0))

    return MB
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
    ps_Up = Params(ps_Up)
    ps_Ut = Params(ps_Ut)
    for batch in data
      # back is a method that computes the product of the gradient so far with its argument.
      train_loss_Up, back_Up = Zygote.pullback(() -> loss(batch...), ps_Up)
      train_loss_Ut, back_Ut = Zygote.pullback(() -> loss(batch...), ps_Ut)
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


##########  MAIN ######################

#function main()

######### Define the network  ############
# Leaky ReLu as activation function
leakyrelu(x, a=0.01) = max(a*x, x)

# Define the networks 1->5->5->5->1
Up = Chain(
    Dense(1,5, leakyrelu), 
    Dense(5,5, leakyrelu), 
    Dense(5,5, leakyrelu), 
    Dense(5,1)
)

Ut = Chain(
    Dense(1,5, leakyrelu), 
    Dense(5,5, leakyrelu), 
    Dense(5,5, leakyrelu), 
    Dense(5,1)
)


# # Get the initial parameters, first is linear decay
# p_P = [rand(Float32); initial_params(Up)]
# p_T = [rand(Float32); initial_params(Ut)]

# We define an optimizer
#opt = RMSProp(0.002, 0.95)
opt = ADAM()

# We get the model parameters to be trained
ps_Up = Flux.params(Up)
ps_Ut = Flux.params(Ut)

sqnorm(x) = sum(abs2, x)

# We determine the loss function
function loss(x, y)
    # Start with a regularization on the network
    # We evaluate the MB as the combination of Accumulation - Ablation with L2
    l = sqrt(Flux.Losses.mse(Up(x[1,:]) - Ut(x[2,:]), y; agg=mean)) + sum(sqnorm, Flux.params(Up)) + sum(sqnorm, Flux.params(Ut))

    return l
end

#######  We generate toy data to train the model  ########
snowfall_toy = rand(0.0f0:10.0f0, 500)
temperature_toy = 10f0*sin.((1:500)./10)
forcings = hcat(snowfall_toy, temperature_toy)
α(P) = P.^(1.1)
μ(T) = T.^(1.3)
MB_toy = toy_MB(α, μ, snowfall_toy, temperature_toy)

# Plot the toy dataset 
l = @layout [a b; c; d]
p1 = plot(1:500, snowfall_toy, label="Snowfall", color="midnightblue")
p2 = plot(1:500, temperature_toy, label="Temperature", color="darkred")
hline!(p2, [0], c="black", label="")
p3 = plot(1:500, MB_toy, label="Mass balance")
hline!(p3, [0], c="black", label="")



X = vcat(snowfall_toy', temperature_toy')
Y = collect(MB_toy')

batch_size = 1
data = Flux.Data.DataLoader((X, Y), batchsize=batch_size)

#data = (X, Y)

# We train the mass balance hybrid model
number_epochs = 4
@epochs number_epochs hybrid_train!(loss, ps_Up, ps_Ut, data, opt)

# Perform MB simulations with trained model
mb_simulated = Up(snowfall_toy') - Ut(temperature_toy')

# Plot comparison with toy data
p4 = plot(1:500, mb_simulated', label="Simulated mass balance")
hline!(p3, [0], c="black", label="")

plot(p1,p2,p3, p4,layout=l)

#end

#main()