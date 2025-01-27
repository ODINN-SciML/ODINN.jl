
"""
get_NN()

Generates a neural network.
"""
function get_NN(θ_trained)
    UA = Flux.Chain(
        Dense(1,3, x->softplus.(x)),
        Dense(3,10, x->softplus.(x)),
        Dense(10,3, x->softplus.(x)),
        Dense(3,1, sigmoid_A)
    )
    UA = Flux.f64(UA)
    # See if parameters need to be retrained or not
    θ, UA_f = Flux.destructure(UA)
    if !isnothing(θ_trained)
        θ = θ_trained
    end
    return UA, θ, UA_f
end

"""
    predict_A̅(UA_f, θ, temp)

Predicts the value of A with a neural network based on the long-term air temperature.
"""
function predict_A̅(U, temp)
    return U(temp) .* 1e-18
end

function sigmoid_A(x) 
    minA_out = 8.0e-3 # /!\     # these depend on predict_A̅, so careful when changing them!
    maxA_out = 8.0
    return minA_out + (maxA_out - minA_out) / ( 1.0 + exp(-x) )
end

function sigmoid_A_inv(x) 
    minA_out = 8.0e-4 # /!\     # these depend on predict_A̅, so careful when changing them!
    maxA_out = 8.0e2
    return minA_out + (maxA_out - minA_out) / ( 1.0 + exp(-x) )
end

# Convert Pythonian date to Julian date
function jldate(pydate)
    return Date(pydate.dt.year.data[1], pydate.dt.month.data[1], pydate.dt.day.data[1])
end

function save_plot(plot, path, filename)
    Plots.savefig(plot,joinpath(path,"png","$filename-$(current_epoch[]).png"))
    Plots.savefig(plot,joinpath(path,"pdf","epoch$(current_epoch[]).pdf"))
end

function generate_plot_folders(path)
    if !isdir(joinpath(path,"png")) || !isdir(joinpath(path,"pdf"))
        mkpath(joinpath(path,"png"))
        mkpath(joinpath(path,"pdf"))
    end
end

# From Cuffey and Paterson
const A_values_sec = ([0.0 -2.0 -5.0 -10.0 -15.0 -20.0 -25.0 -30.0 -35.0 -40.0 -45.0 -50.0;
                              2.4e-24 1.7e-24 9.3e-25 3.5e-25 2.1e-25 1.2e-25 6.8e-26 3.7e-26 2.0e-26 1.0e-26 5.2e-27 2.6e-27]) # s⁻¹Pa⁻³
const A_values = hcat(A_values_sec[1,:], A_values_sec[2,:].*60.0*60.0*24.0*365.25)'

# Polynomial fit for Cuffey and Paterson data 
A_f = Polynomials.fit(A_values[1,:], A_values[2,:]) # degree = length(xs) - 1

const noise_A_magnitude = 5e-18  # magnitude of noise to be added to A
const rng_seed() = MersenneTwister(666)   # Random seed

function A_fake(temp, noise=false)
    # A = @. minA + (maxA - minA) * ((temp-minT)/(maxT-minT) )^2
    A = A_f.(temp) # polynomial fit
    if noise
        A_noise = rand(rng_seed()) .* noise_A_magnitude
        A = abs.(A .+ A_noise)
    end
    return A
end

function build_D_features(H::Matrix, temp, ∇S)
    ∇S_flat = ∇S[inn1(H) .!= 0.0] # flatten
    H_flat = H[H .!= 0.0] # flatten
    T_flat = repeat(temp,length(H_flat))
    X = Flux.normalise(hcat(H_flat,T_flat,∇S_flat))' # build feature matrix
    return X
end

function build_D_features(H::Float64, temp::Float64, ∇S::Float64)
    X = Flux.normalise(hcat([H],[temp],[∇S]))' # build feature matrix
    return X
end

function predict_diffusivity(UD_f, θ, X)
    UD = UD_f(θ)
    return UD(X)[1,:]
end

"""
    generate_batches(simulation::S; shuffle=true)

Generates batches for the UE inversion problem based on input data and feed them to the loss function.
"""
function generate_batches(simulation::S; shuffle=true) where {S <: Simulation}
    # Repeat simulations for batches
    batch_ids::Vector{Int} = collect(1:length(simulation.glaciers))
    rgi_ids::Vector{String} = [glacier.rgi_id for glacier in simulation.glaciers]
    batches = (batch_ids, rgi_ids)
    train_loader = Flux.DataLoader(batches, batchsize=simulation.parameters.hyper.batch_size, shuffle=shuffle)

    return train_loader
end

"""
    update_training_state(simulation, l)

Update training state to know if the training has completed an epoch.
If so, reset minibatches, update history loss and bump epochs.
"""
function update_training_state!(simulation, l)
    # Update minibatch count and loss for the current epoch
    simulation.parameters.hyper.current_minibatch += simulation.parameters.hyper.batch_size
    simulation.parameters.hyper.loss_epoch += l
    if simulation.parameters.hyper.current_minibatch >= length(simulation.glaciers)
        # Track evolution of loss per epoch
        push!(simulation.parameters.hyper.loss_history, simulation.parameters.hyper.loss_epoch)
        println("Epoch #$(simulation.parameters.hyper.current_epoch): ", simulation.parameters.hyper.loss_epoch)
        # Bump epoch and reset loss and minibatch count
        simulation.parameters.hyper.current_epoch += 1
        simulation.parameters.hyper.current_minibatch = 0
        simulation.parameters.hyper.loss_epoch = 0.0
    end
end