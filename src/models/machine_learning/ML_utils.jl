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
function get_default_NN(θ_trained, ft; lightNN = false, seed = nothing)
    architecture = build_default_NN(; lightNN = lightNN)
    return set_NN(architecture; θ_trained = θ_trained, ft = ft, seed = seed)
end

function build_default_NN(; n_input = 1, lightNN = false)
    if lightNN
        @warn "Using light mode of neural network"
        architecture = Lux.Chain( # Light network for debugging
            Dense(n_input, 3, x -> softplus.(x)),
            Dense(3, 1, sigmoid)
            # Dense(n_input, 3, x -> sigmoid.(x); init_weight = Lux.glorot_normal),
            # Dense(3, 1, sigmoid; init_weight = Lux.glorot_normal)
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

function set_NN(architecture; θ_trained = nothing, ft = nothing, seed = nothing)
    # Set neural network using Lux
    if isnothing(seed)
        θ, st = Lux.setup(rng_seed(), architecture)
    else
        θ, st = Lux.setup(seed, architecture)
    end

    # Set pre-trained weights if provided
    if !isnothing(θ_trained)
        θ = θ_trained
    else
        # Build parameter as component array
        θ = ComponentArray(θ=θ)
    end

    # TODO: To re-write with the new type stability fix
    if ft == Float64
        architecture = f64(architecture)
        θ = f64(θ)
        st = f64(st)
    end

    return architecture, θ, st
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

const rng_seed() = MersenneTwister(666)   # Random seed

function build_simulation_batch(
    simulation::FunctionalInversion,
    i::I,
    nbatches::I = 1
    ) where {I <: Integer}

    iceflow = simulation.model.iceflow
    massbalance = simulation.model.mass_balance
    ml = simulation.model.machine_learning

    # TODO: in the future we could avoid a copy of model since it is stateless
    # but we need to pay attention that there is no side effect with multiprocessing
    model = Sleipnir.Model(iceflow, massbalance, ml)

    # Each element of the batch has access only to the current glacier, so glacier_idx=1
    cache = init_cache(model, simulation, 1, simulation.parameters)
    glacier = simulation.glaciers[i]
    if length(simulation.results.simulation) < 1
        return FunctionalInversion{typeof(model), cache_type(model), typeof(glacier), typeof(simulation.results)}(model, cache, [glacier], simulation.parameters, simulation.results)
    else
        # TODO: Notice this assumes there is just one vector in results! Probably needs a fix
        # results = Results([simulation.results.simulation[i]], simulation.results.stats)
        results = Results([only(simulation.results.simulation)], simulation.results.stats)
        return FunctionalInversion{typeof(model), cache_type(model), typeof(glacier), typeof(simulation.results)}(model, cache, [glacier], simulation.parameters, results)
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
    # TODO: we need to change how this is done here, with a manual =1 batch size
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