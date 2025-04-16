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
function get_default_NN(θ_trained, ft; lightNN = false)
    architecture = build_default_NN(; lightNN = lightNN)
    return set_NN(architecture; θ_trained = θ_trained, ft = ft)
end

function build_default_NN(; n_input = 1, lightNN = false)
    if lightNN
        @warn "Using light mode of neural network"
        architecture = Lux.Chain( # Light network for debugging
            Dense(n_input, 3, x -> softplus.(x)),
            Dense(3, 1, sigmoid)
        )
    else
        architecture = Lux.Chain(
            Dense(n_input, 3, x -> softplus.(x)),
            Dense(3, 10, x -> softplus.(x)),
            Dense(10, 3, x -> softplus.(x)),
            Dense(3, 1, sigmoid)
        )
    end
    return architecture
end

function set_NN(architecture; θ_trained = nothing, ft = nothing)
    # Set neural network using Lux
    θ, st = Lux.setup(rng_seed(), architecture)

    # Set pre-trained weights if provided
    if !isnothing(θ_trained)
        θ = θ_trained
    end

    # TODO: To re-write with the new type stability fix
    if ft == Float64
        architecture = f64(architecture)
        θ = f64(θ)
        st = f64(st)
    end

    # Build parameter as component array
    θ = ComponentArray(θ=θ)
    return architecture, θ, st
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
    generate_ground_truth(glaciers::Vector{G}, law::Symbol, params, model, tstops::Vector{F}) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}

Generate ground truth data for a glacier simulation by applying a specified flow law and running a forward model.

# Arguments
- `glaciers::Vector{G}`: A vector of glacier objects of type `G`, where `G` is a subtype of `Sleipnir.AbstractGlacier`.
- `law::Symbol`: The flow law to use for the simulation. Currently supports `:PatersonCuffey`.
- `params`: Simulation parameters, typically of type `Sleipnir.Parameters`.
- `model`: The model to use for the simulation, typically of type `Sleipnir.Model`.
- `tstops::Vector{F}`: A vector of time steps (of type `F <: AbstractFloat`) at which the simulation will be evaluated.

# Description
1. Applies the specified flow law (`law`) to generate a polynomial function for the flow rate factor `A`.
2. Generates a fake flow rate factor `A` for each glacier based on the long-term temperature of the glacier.
3. Runs a forward model simulation for the glaciers using the provided parameters, model, and time steps.

# Notes
- If an unsupported flow law is provided, an error is logged.
- The function modifies the `glaciers` vector in place by updating their flow rate factor `A` and running the forward model.

# Example
```julia
glaciers = [glacier1, glacier2] # dummy example
law = :PatersonCuffey
params = Sleipnir.Parameters(...) # to be filled
model = Sleipnir.Model(...) # to be filled
tstops = 0.0:1.0:10.0

generate_ground_truth(glaciers, law, params, model, tstops)
```
"""
function generate_ground_truth(
    glaciers::Vector{G},
    law::Union{Symbol, Function, Polynomials.Polynomial},
    params::Sleipnir.Parameters,
    model::Sleipnir.Model,
    tstops::Vector{F}
) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}
    # Generate a fake forward model for the simulation
    fakeA = get_rheology_law(law)

    # Generate a fake A for the glaciers
    generate_fake_A!(glaciers, fakeA)

    # Generate a fake forward model for the simulation
    generate_glacier_prediction!(glaciers, params, model, tstops)
end

"""
    get_rheology_law(law::Symbol)

Retrieve the rheology law function for the flow rate factor `A` based on the specified law.

# Arguments
- `law::Symbol`: A symbol representing the rheology law to use. Currently supports `:PatersonCuffey`.

# Returns
- A function `fakeA(T)` that computes the flow rate factor `A` for a given temperature `T` using the specified rheology law.

# Description
This function retrieves the parametrization law for the glacier's flow rate factor `A`. If the specified law is `:PatersonCuffey`, it uses the `A_law_PatersonCuffey` polynomial to define the flow rate factor as a function of temperature. If an unsupported law is provided, an error is logged.

# Notes
- The returned function `fakeA(T)` can be used to compute the flow rate factor for a given temperature `T`.
- If an unknown law is provided, the function logs an error and does not return a valid function.
"""
function get_rheology_law(law::Symbol)
    # Get the parametrization law for the glacier
    if law == :PatersonCuffey
        A_poly = A_law_PatersonCuffey()
        fakeA(T) = A_poly(T)
        return fakeA
    else
        @error "Unknown law of A: $law"
    end
end

"""
    get_rheology_law(law::Polynomial)

Convert a polynomial into a rheology law function for the flow rate factor `A`.

# Arguments
- `law::Polynomial`: A polynomial representing the rheology law for the flow rate factor `A`.

# Returns
- A function `fakeA(T)` that computes the flow rate factor `A` for a given temperature `T` using the provided polynomial.
"""
function get_rheology_law(law::Polynomials.Polynomial)
    # Convert polynomial into function
    fakeA(T) = law(T)
    return fakeA(T)
end

"""
    get_rheology_law(law::Function)

Return the provided rheology law function without modification. 
This just uses multiple dispatch to handle cases where the rheology law is already a function.

# Arguments
- `law::Function`: A function representing the rheology law for the flow rate factor `A`.

# Returns
- The input function `law`, unchanged.

# Description
This function is a simple bypass that uses multiple dispatch to handle cases where the rheology law is already provided as a function. It directly returns the input function without any modifications.
"""
function get_rheology_law(law::Function)
    # Just bypass using multiple dispatch
    return law
end

"""
    generate_fake_A!(glaciers::Vector{G}, fakeA::Function) where {G <: Sleipnir.AbstractGlacier}

Generate and assign a fake flow rate factor `A` for a vector of glaciers based on their long-term temperatures.

# Arguments
- `glaciers::Vector{G}`: A vector of glacier objects of type `G`, where `G` is a subtype of `Sleipnir.AbstractGlacier`.
- `fakeA::Function`: A function that computes the flow rate factor `A` based on the mean long-term temperature of a glacier.

# Description
This function iterates over a vector of glaciers and computes the flow rate factor `A` for each glacier using the provided `fakeA` function. The flow rate factor is computed based on the mean of the glacier's long-term temperature (`longterm_temps`) and is assigned to the glacier's `A` property.

# Notes
- The `fakeA` function should take a single argument (temperature) and return the corresponding flow rate factor.
- This function modifies the `glaciers` vector in place by updating the `A` property of each glacier.
"""
function generate_fake_A!(glaciers::Vector{G}, fakeA::Function) where {G <: Sleipnir.AbstractGlacier}
    # Generate a fake A for the glaciers 
    for glacier in glaciers
        T = glacier.climate.longterm_temps
        glacier.A = fakeA(mean(T))
    end
end

"""
    store_thickness_data!(prediction::Prediction, tstops::Vector{F}) where {F <: AbstractFloat}

Store the simulated thickness data in the corresponding glaciers within a `Prediction` object.

# Arguments
- `prediction::Prediction`: A `Prediction` object containing the simulation results and associated glaciers.
- `tstops::Vector{F}`: A vector of time steps (of type `F <: AbstractFloat`) at which the simulation was evaluated.

# Description
This function iterates over the glaciers in the `Prediction` object and stores the simulated thickness data (`H`) and corresponding time steps (`t`) in the `data` field of each glacier. If the `data` field is empty (`nothing`), it initializes it with the thickness data. Otherwise, it appends the new thickness data to the existing data.

# Notes
- The function asserts that the time steps (`ts`) in the simulation results match the provided `tstops`. If they do not match, an error is raised.
- T
"""
function store_thickness_data!(prediction::Prediction, tstops::Vector{F}) where {F <: AbstractFloat}

    # Store the thickness data in the glacier
    for i in 1:length(prediction.glaciers)
        ts = prediction.results[i].t
        Hs = prediction.results[i].H
    
        @assert ts ≈ tstops "Timestops of simulated PDE solution and UDE solution do not match."

        if isnothing(prediction.glaciers[i].data)
            prediction.glaciers[i].data = [Sleipnir.ThicknessData(ts, Hs)]
        else
            append!(prediction.glaciers[i].data, Sleipnir.ThicknessData(ts, Hs))
        end
    end
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
    generate_simulation_batches(simulation::FunctionalInversion)

Generate batches of simulations from a `FunctionalInversion` object for parallel or batched processing.

# Arguments
- `simulation::FunctionalInversion`: A `FunctionalInversion` object containing the model, glaciers, parameters, results, and statistics for the simulation.

# Returns
- A vector of `FunctionalInversion` objects, each representing a batch of simulations. Each batch contains a subset of glaciers, models, and results from the original simulation.

# Description
This function splits the glaciers and associated data in the `simulation` object into smaller batches for processing. Each batch is represented as a new `FunctionalInversion` object. The number of batches is determined by the `nbatches` variable (currently set to 1). If the simulation results are empty, the function creates batches with empty results. Otherwise, it includes the corresponding results for each glacier in the batches.

# Notes
- The number of glaciers (`ninstances`) must be divisible by the number of batches (`nbatches`). An assertion is used to enforce this condition.
- The function currently defaults to `nbatches = 1`, meaning no actual batching is performed. This can be updated to use `simulation.parameters.hyper.batchsize` for dynamic batching.
- If the simulation results are empty, the function creates batches with empty results objects.
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

"""
    merge_batches(results::Vector)

Merge simulation results from multiple batches into a single collection.

# Arguments
- `results::Vector`: A vector where each element is a collection of results (e.g., arrays or vectors) from a batch.

# Returns
- A single collection containing all the merged results from the input batches.

# Description
This function takes a vector of results from multiple simulation batches and merges them into a single collection using vertical concatenation (`vcat`). It is useful for combining results that were processed in parallel or in separate batches.
"""
function merge_batches(results::Vector)
    return reduce(vcat,results)
end

"""
    generate_batches(simulation::S; shuffle=false) where {S <: Simulation}

Generate a data loader for batching simulations, optionally shuffling the batches.

# Arguments
- `simulation::S`: A `Simulation` object (or subtype of `Simulation`) containing the data to be batched.
- `shuffle::Bool=false`: A flag indicating whether to shuffle the batches. Defaults to `false`.

# Returns
- A `DataLoader` object that provides batched access to the simulation data.

# Description
This function creates a `DataLoader` for batching the provided simulation object. The `DataLoader` allows for efficient iteration over the simulation data in batches. The batch size is set to `1` by default, and the `shuffle` flag determines whether the batches are shuffled. If `shuffle` is enabled, a warning is logged to indicate that the batches used for parallelization are being shuffled.

# Notes
- The batch size is fixed at `1` in this implementation. To modify the batch size, you may need to adjust the `DataLoader` initialization.
- Shuffling the batches may affect reproducibility and parallelization behavior.
"""
function generate_batches(simulation::S; shuffle=false) where {S <: Simulation}
    if shuffle
        @warn "You are shuffling the batches used for paralelization."
    end
    # Combined batch object
    simulations = [simulation]
    # Create train loader use for simulations
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