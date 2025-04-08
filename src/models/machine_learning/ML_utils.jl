"""
    get_NN(θ_trained, ft; lightNN=false)

Generates a neural network.

# Arguments
- `θ_trained`: Pre-trained neural network parameters (optional).
- `ft`: Float type used.
- `lightNN`: Boolean that determines if a light architecture is returned or not.

# Returns
- `UA`: `Lux.Chain` neural network architecture.
- `θ`: Neural network parameters.
- `st`: Lux state.
"""
function get_NN(θ_trained, ft; lightNN=false)
    if lightNN
        @warn "Using light mode of neural network"
        UA = Lux.Chain( # Light network for debugging
            Dense(1, 3, x -> softplus.(x)),
            Dense(3, 1, sigmoid)
        )
    else
        UA = Lux.Chain(
            Dense(1, 3, x -> softplus.(x)),
            Dense(3, 10, x -> softplus.(x)),
            Dense(10, 3, x -> softplus.(x)),
            Dense(3, 1, sigmoid)
        )
    end
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
    predict_A̅(U, temp, lims::Tuple{F, F}) where {F <: AbstractFloat}

Predicts the value of A with a neural network based on the long-term air temperature
and on the bounds value to normalize the output of the neural network.

# Arguments
- `U`: Neural network.
- `temp`: Temperature to be fed as an input of the neural network.
- `lims::Tuple{F, F}`: Bounds to use for the affine transformation of the neural
    network output.
"""
function predict_A̅(U, temp, lims::Tuple{F, F}) where {F <: AbstractFloat}
    return only(normalize_A(U(temp), lims))
end

"""
    normalize_A(x, lims::Tuple{F, F}) where {F <: AbstractFloat}

Normalize a variable by using an affine transformation defined by some lower and
upper bounds (m, M). The returned value is m+(M-m)*x.

# Arguments
- `x`: Input value.
- `lims::Tuple{F, F}`: Lower and upper bounds to use in the affine transformation.

# Returns
- The input variable scaled by the affine transformation.
"""
function normalize_A(x, lims::Tuple{F, F}) where {F <: AbstractFloat}
    minA_out = lims[1]
    maxA_out = lims[2]
    return minA_out .+ (maxA_out - minA_out) .* x
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

"""
    A_law_PatersonCuffey()

Returns a law of the coefficient A as a polynomial of the temperature.
The values used to fit the polynomial come from Peterson & Cuffey.
"""
function A_law_PatersonCuffey()
    # Law of A(T) from Peterson & Cuffey
    A_values_sec = ([0.0 -2.0 -5.0 -10.0 -15.0 -20.0 -25.0 -30.0 -35.0 -40.0 -45.0 -50.0;
                                2.4e-24 1.7e-24 9.3e-25 3.5e-25 2.1e-25 1.2e-25 6.8e-26 3.7e-26 2.0e-26 1.0e-26 5.2e-27 2.6e-27]) # s⁻¹Pa⁻³
    A_values = hcat(A_values_sec[1,:], A_values_sec[2,:].*60.0*60.0*24.0*365.25)'
    return Polynomials.fit(A_values[1,:], A_values[2,:])
end

# Polynomial fit for Cuffey and Paterson data
A_f = A_law_PatersonCuffey() # degree = length(xs) - 1

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
    generate_ground_truth(glacier::G, fakeA::Function, params, model, tstops::Vector{F})

Generates ground truth data and populate glacier with the ground truth observation
given a fake law of A.

Arguments:
- `glacier::G`: Glacier instance.
- `fakeA::Function`: Function that maps a temperature to A.
- `params::`: The simulation parameters.
- `model`:: The model that includes iceflow and a machine learning model.
- `tstops`:: Vector of time points where the solver should stop.
"""
function generate_ground_truth(
    glacier::G,
    fakeA::Function,
    params,
    model,
    tstops::Vector{F}
) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}
    T = mean(glacier.climate.longterm_temps)
    A = fakeA(T)
    # Generate a fake forward model for the simulation
    generate_glacier_prediction!(glacier, params, model; A = A, tstops=tstops)
end

function build_simulation_batch(simulation::FunctionalInversion, i::I, nbatches::I=1) where {I <: Integer}
    iceflow = simulation.model.iceflow[i]
    massbalance = simulation.model.mass_balance[i]
    ml = simulation.model.machine_learning
    model = Sleipnir.Model{typeof(iceflow), typeof(massbalance), typeof(ml)}([iceflow], [massbalance], ml)
    ####
    # iceflow = simulation.model.iceflow[(i-1)*nbatches+1:i*nbatches]
    # massbalance = simulation.model.mass_balance[(i-1)*nbatches+1:i*nbatches]
    # ml = simulation.model.machine_learning
    # Sleipnir.Model{typeof(iceflow[1]), typeof(massbalance[1]), typeof(ml)}(iceflow, massbalance, ml)
    ####
    if length(simulation.results) < 1
        # TODO: Pass empty results object: machine learning should not be here!
        return FunctionalInversion{typeof(simulation.glaciers[i]), typeof(model), typeof(simulation.parameters)}(model, [simulation.glaciers[i]], simulation.parameters, simulation.results, simulation.stats)
        # return FunctionalInversion{typeof(simulation.glaciers[i*nbatches]), typeof(model), typeof(simulation.parameters)}(model, simulation.glaciers[(i-1)*nbatches+1:i*nbatches], simulation.parameters, simulation.results, simulation.stats)
    else
        return FunctionalInversion{typeof(simulation.glaciers[i]), typeof(model), typeof(simulation.parameters)}(model, [simulation.glaciers[i]], simulation.parameters, [simulation.results[i]], simulation.stats)
        # return FunctionalInversion{typeof(simulation.glaciers[i*nbatches]), typeof(model), typeof(simulation.parameters)}(model, simulation.glaciers[(i-1)*nbatches+1:i*nbatches], simulation.parameters, [simulation.results[(i-1)*nbatches+1:i*nbatches]], simulation.stats)
    end
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
    return [build_simulation_batch(simulation, i, nbatches) for i in 1:ninstances]
    # return [build_simulation_batch(simulation, i, nbatches) for i in 1:folds]
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
- None. This function updates the state in-place.
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