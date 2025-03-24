"""
    get_NN(θ_trained, ft)

Generates a neural network.

# Arguments
- `ft`: Float type.
- `θ_trained`: Pre-trained neural network parameters (optional).
- `ft`: Float type used. 

# Returns
- `UA`: `Lux.Chain` neural network architecture.
- `θ`: Neural network parameters.
- `st`: Lux state.
"""
function get_NN(θ_trained, ft)
    UA = Lux.Chain( # Light network for debugging
        Dense(1, 3, x -> softplus.(x)),
        Dense(3, 1, sigmoid_A)
    )
    # UA = Lux.Chain(
    #     Dense(1, 3, x -> softplus.(x)),
    #     Dense(3, 10, x -> softplus.(x)),
    #     Dense(10, 3, x -> softplus.(x)),
    #     # Dense(3, 1, sigmoid)
    #     Dense(3, 1, sigmoid_A)
    # )
    θ, st = Lux.setup(rng_seed(), UA)
    if !isnothing(θ_trained)
        θ = θ_trained
    end

    # TODO: To re-write with the new type stability fix 
    if ft == Float64
        UA = f64(UA)
        θ = f64(θ)
        st = f64(st)
    end

    θ = ComponentArray(θ=θ)
    return UA, θ, st
end

"""
    predict_A̅(U, temp)

Predicts the value of A with a neural network based on the long-term air temperature.
"""
function predict_A̅(U, temp)
    return only(U(temp)) * 1e-18
end

# function predict_A̅(U, temp)
#     # Scaled variables
#     minT = -30.0f0
#     maxT =   5.0f0 
#     minA = 6.0f-20
#     maxA = 8.0f-17
#     minLogA = log10(minA)
#     maxLogA = log10(maxA)
#     temp_scaled = 10 .* (temp .- (minT+maxT)) ./ (maxT-minT)
#     # Output of NN. This value will be O(1)
#     nn_output_log_raw = U(temp_scaled)
#     # Scale for sigmoid
#     nn_output_log_scaled = minLogA .+ (maxLogA-minLogA) ./ (1.0 .+ exp.(-nn_output_log_raw))
#     # nn_output_log_scaled = (minLogA + maxLogA)/2 .+ (maxLogA - minLogA) .* nn_output_log_raw
#     nn_output_scaled = 10.0f0.^nn_output_log_scaled
#     @assert minLogA .<= only(nn_output_log_scaled) .<= maxLogA
#     return nn_output_scaled
# end

"""
    sigmoid_A(x)

Sigmoid activation function for the neural network output.

# Arguments
- `x`: Input value.

# Returns
- Sigmoid-transformed output value.
"""
function sigmoid_A(x)
    minA_out = 8.0e-3 # /!\     # these depend on predict_A̅, so careful when changing them!
    maxA_out = 8.0
    return minA_out + (maxA_out - minA_out) / (1.0 + exp(-x))
end

"""
    sigmoid_A_inv(x)

Inverse sigmoid activation function for the neural network output.

# Arguments
- `x`: Input value.

# Returns
- Inverse sigmoid-transformed output value.
"""
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
    X = Lux.normalise(hcat(H_flat,T_flat,∇S_flat))' # build feature matrix
    return X
end

function build_D_features(H::Float64, temp::Float64, ∇S::Float64)
    X = Lux.normalise(hcat([H],[temp],[∇S]))' # build feature matrix
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
function generate_simulation_batches(simulation::FunctionalInversion)
    nbatches = 1 #simulation.parameters.hyper.batchsize
    ninstances = length(simulation.glaciers)
    @assert ninstances % nbatches == 0
    folds = Int(ninstances / nbatches)
    # Forward runs still don't have multiple results
    if length(simulation.results) < 1
        # TODO: Pass empty results object: machine learning should not be here!
        return [FunctionalInversion(Sleipnir.Model([simulation.model.iceflow[i]], [simulation.model.mass_balance[i]], simulation.model.machine_learning), [simulation.glaciers[i]], simulation.parameters, simulation.results, simulation.stats) for i in 1:ninstances]
        # return [FunctionalInversion(Sleipnir.Model(simulation.model.iceflow[(i-1)*nbatches+1:i*nbatches], simulation.model.mass_balance[(i-1)*nbatches+1:i*nbatches], simulation.model.machine_learning), simulation.glaciers[(i-1)*nbatches+1:i*nbatches], simulation.parameters, simulation.results, simulation.stats) for i in 1:folds]
    else
        return [FunctionalInversion(Sleipnir.Model([simulation.model.iceflow[i]], [simulation.model.mass_balance[i]], simulation.model.machine_learning), [simulation.glaciers[i]], simulation.parameters, [simulation.results[i]], simulation.stats) for i in 1:ninstances]
        # return [FunctionalInversion(Sleipnir.Model(simulation.model.iceflow[(i-1)*nbatches+1:i*nbatches], simulation.model.mass_balance[(i-1)*nbatches+1:i*nbatches], simulation.model.machine_learning), simulation.glaciers[(i-1)*nbatches+1:i*nbatches], simulation.parameters, [simulation.results[(i-1)*nbatches+1:i*nbatches]], simulation.stats) for i in 1:folds]
    end
end

function merge_batches(results::Vector)
    return reduce(vcat,results)
end

function generate_batches(simulation::S; shuffle=false) where {S <: Simulation}
    if shuffle
        @warn "You are shuffling the batches used for paralelization."
    end
    # Combined batch object
    simulations = [simulation]
    # Create train loeader use for simulations
    # batchsize is already set in generate simulation
    train_loader = DataLoader(simulations, batchsize=1, shuffle=shuffle)
    return train_loader
end


"""
    update_training_state!(simulation::S, l) where {S <: Simulation}

Update the training state to determine if the training has completed an epoch.
If an epoch is completed, reset the minibatches, update the history loss, and increment the epoch count.

# Arguments
- `simulation`: The current state of the simulation or training process.
- `l`: The current loss value or other relevant metric.

# Returns
- None. This function updates the state in place.
"""
function update_training_state!(simulation::S, l) where {S <: Simulation}
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