"""
    run!(simulation::FunctionalInversion)

Run the training process for a given `FunctionalInversion` simulation.

# Arguments
- `simulation::FunctionalInversion`: The simulation object containing the parameters and settings for the functional inversion process.

# Description
This function initiates the training of a Universal Differential Equation (UDE) for the provided simulation. It prints a message indicating the start of the training process, calls the `train_UDE!` function to perform the training, and collects the results in `results_list`. The results are intended to be saved using `Sleipnir.save_results_file!`, but this step is currently commented out and will be enabled once the optimization is working. Finally, the garbage collector is triggered to free up memory.

# Notes
- The `Sleipnir.save_results_file!` function call is currently commented out and should be enabled once the optimization process is confirmed to be working.
- The garbage collector is explicitly run using `GC.gc()` to manage memory usage.
"""
function run!(
    simulation::FunctionalInversion;
    path::Union{String, Nothing} = nothing,
    file_name::Union{String, Nothing} = nothing,
    save_every_iter::Bool = false,
)

    println("Running training of UDE...\n")

    # Set expected total number of epochs from beginning for the callback
    simulation.stats.niter = sum(simulation.parameters.hyper.epochs)

    if !(typeof(simulation.parameters.hyper.optimizer) <: Vector)
        # One single optimizer
        sol = train_UDE!(simulation; save_every_iter=save_every_iter)
    else
        # Multiple optimizers
        optimizers = simulation.parameters.hyper.optimizer
        epochs = simulation.parameters.hyper.epochs
        @assert length(optimizers) == length(epochs) "Provide number of epochs as a vector with the same length of optimizers"
        for i in 1:length(epochs)
            # Construct a new simulation for each optimizer
            simulation.parameters.hyper.optimizer = optimizers[i]
            simulation.parameters.hyper.epochs = epochs[i]
            if i !== 1
                θ_trained = sol.u
                simulation.model.machine_learning.θ = θ_trained
            end
            sol = train_UDE!(simulation; save_every_iter=save_every_iter)
        end
    end

    # Setup final results
    simulation.stats.niter = length(simulation.stats.losses)
    # simulation.stats.retcode = sol.
    simulation.stats.θ = sol.u

    simulation.model.machine_learning.θ = sol.u

    # TODO: Save when optimization is working
    # Save results in path is provided
    if !isnothing(path) & !isnothing(file_name)
        ODINN.save_inversion_file!(sol, simulation; path = path, file_name = file_name)
    end

    @everywhere GC.gc() # run garbage collector

end

"""
train_UDE!(simulation::FunctionalInversion; save_every_iter::Bool=false)

Trains UDE based on the current FunctionalInversion.
"""
function train_UDE!(simulation::FunctionalInversion; save_every_iter::Bool=false)
    optimizer = simulation.parameters.hyper.optimizer
    iceflow_trained = train_UDE!(simulation, optimizer; save_every_iter=save_every_iter)
    return iceflow_trained
end

"""
BFGS Training
"""
function train_UDE!(simulation::FunctionalInversion, optimizer::Optim.FirstOrderOptimizer; save_every_iter::Bool=false)

    @info "Training with BFGS optimizer"

    # Create batches for inversion training
    simulation_train_loader = generate_batches(simulation)
    # simulation_batch_ids = train_loader.data[1]

    θ = simulation.model.machine_learning.θ

    # Get the available workers
    # Workers are always the number of allocated cores to Julia minus one
    workers_list = workers()
    if simulation.parameters.simulation.multiprocessing
        @assert length(workers_list) == (simulation.parameters.simulation.workers - 1) "Number of workers does not match"
    end

    # Simplify API for optimization problem and include data loaded in argument for minibatch
    loss_function(_θ, _simulation) = loss_iceflow_transient(_θ, only(_simulation), pmap)

    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        # Enzyme.API.strictAliasing!(false)
        optf = OptimizationFunction(loss_function, simulation.parameters.UDE.optim_autoAD)
    else
        @info "Training with custom $(typeof(simulation.parameters.UDE.grad)) method"
        loss_function_grad!(_dθ, _θ, _simulation) = SIA2D_grad!(_dθ, _θ, only(_simulation))
        optf = OptimizationFunction(loss_function, NoAD(), grad=loss_function_grad!)
    end

    # optprob = OptimizationProblem(optf, θ, (simultion_batch_ids))
    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Training diagnosis callback
    cb(θ, l) = callback_diagnosis(θ, l, simulation; save=save_every_iter)

    iceflow_trained = solve(
        optprob,
        simulation.parameters.hyper.optimizer,
        maxiters = simulation.parameters.hyper.epochs,
        allow_f_increases = true,
        callback = cb,
        progress = false
        )

    return iceflow_trained
end

"""
ADAM Training
"""
function train_UDE!(simulation::FunctionalInversion, optimizer::AR; save_every_iter::Bool=false) where {AR <: Optimisers.AbstractRule}

    @info "Training with ADAM optimizer"

    # Create batches for inversion training
    simulation_train_loader = generate_batches(simulation)

    θ = simulation.model.machine_learning.θ

    # Get the available workers
    # Workers are always the number of allocated cores to Julia minus one
    workers_list = workers()
    if simulation.parameters.simulation.multiprocessing
        @assert length(workers_list) == (simulation.parameters.simulation.workers - 1) "Number of workers does not match"
    end

    # Simplify API for optimization problem and include data loaded in argument for minibatch
    # glacier_data_batch is a pair of the data sampled (e.g, glacier_data_batch = (id, glacier))
    # _glacier_data_batch has a simulation!
    loss_function(_θ, simulation_loader) = loss_iceflow_transient(_θ, simulation_loader[1], pmap)

    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        # Enzyme.API.strictAliasing!(false)
        optf = OptimizationFunction(loss_function, simulation.parameters.UDE.optim_autoAD)
    else
        @info "Training with custom $(typeof(simulation.parameters.UDE.grad)) method"
        loss_function_grad!(_dθ, _θ, simulation_loader) = SIA2D_grad!(_dθ, _θ, simulation_loader[1])
        optf = OptimizationFunction(loss_function, NoAD(), grad=loss_function_grad!)
    end

    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Training diagnosis callback
    cb(θ, l) = callback_diagnosis(θ, l, simulation; save=save_every_iter)

    iceflow_trained = solve(
        optprob,
        simulation.parameters.hyper.optimizer,
        maxiters = simulation.parameters.hyper.epochs,
        callback = cb,
        progress = false
        )

    return iceflow_trained
end

function loss_iceflow_transient(θ, simulation::FunctionalInversion, mappingFct)

    predict_iceflow!(θ, simulation, mappingFct)

    loss_function = simulation.parameters.UDE.empirical_loss_function

    l = 0.0

    for i in 1:length(simulation.glaciers)
        glacier = simulation.glaciers[i]

        H_ref = glacier.thicknessData.H
        t = simulation.results[i].t
        Δt = diff(t)
        H = simulation.results[i].H
        @assert size(H_ref[begin]) == size(H[begin]) "Initial state of reference and predicted ice thickness do not match."
        @assert length(H_ref) == length(H) "Size of reference and prediction datasets do not match."
        @assert t == glacier.thicknessData.t "Reference and prediction need to be evaluated with the same timestamps."

        if isa(loss_function, LossHV) || isa(loss_function, LossV)
            @assert !isnothing(glacier.velocityData) "Using $(typeof(loss_function)) but no velocityData in the glacier $(glacier.rgi_id)"
            @assert length(glacier.velocityData.date)>0 "Using $(typeof(loss_function)) but no reference velocity in the results"
        end

        β = 2.0
        for τ in 2:length(H)
            # println("t[τ] = ", t[τ], "  τ=",τ)
            normalization = 1.0
            # normalization = std(H_ref[τ][H_ref[τ] .> 0.0])^β
            mean_error = loss(
                loss_function,
                H[τ],
                H_ref[τ],
                t[τ],
                glacier,
                θ,
                simulation;
                normalization=prod(size(H_ref[τ]))*normalization,
            )
            l += Δt[τ-1] * mean_error
        end
    end

    return l

end

function predict_iceflow!(θ, simulation::FunctionalInversion, mappingFct)
    simulations = generate_simulation_batches(simulation)
    results = mappingFct(simulation -> batch_iceflow_UDE(θ, simulation), simulations)
    simulation.results = ODINN.merge_batches(results)
end

function batch_iceflow_UDE(θ, simulation::FunctionalInversion)
    return [_batch_iceflow_UDE(θ, simulation, glacier_idx) for glacier_idx in 1:length(simulation.glaciers)]
end

function _batch_iceflow_UDE(θ, simulation::FunctionalInversion, glacier_idx::I) where {I <: Integer}

    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]

    simulation.cache = init_cache(simulation.model, simulation, glacier_idx, params)
    simulation.model.machine_learning.θ = θ

    # Create mass balance callback
    tstops = Huginn.define_callback_steps(params.simulation.tspan, params.solver.step)
    stop_condition(u,t,integrator) = Sleipnir.stop_condition_tstops(u,t,integrator, tstops)
    params.solver.tstops = tstops

    mb_action! = let model = simulation.model, cache = simulation.cache, glacier = glacier, step = params.solver.step
        function (integrator)
            if params.simulation.use_MB
                # Compute mass balance
                MB_timestep!(cache, model, glacier, step, integrator.t)
                apply_MB_mask!(integrator.u, cache.iceflow)
            end
        end
    end
    cb_MB = DiscreteCallback(stop_condition, mb_action!)

    # Create iceflow law callback
    cb_iceflow = Huginn.build_callback(simulation.model.iceflow, simulation.cache.iceflow, simulation.cache.iceflow.glacier_idx, θ)

    cb = CallbackSet(cb_MB, cb_iceflow)

    # Run iceflow UDE for this glacier
    iceflow_sol = simulate_iceflow_UDE!(θ, simulation, cb, glacier_idx)

    # Update simulation results
    result = Sleipnir.create_results(
        simulation, glacier_idx, iceflow_sol, nothing;
        tstops = simulation.parameters.solver.tstops,
        light = !simulation.parameters.solver.save_everystep,
    )

    return result
end


"""
    simulate_iceflow_UDE!(
        θ,
        simulation::SIM,
        cb::SciMLBase.DECallback,
        glacier_idx::I,
    ) where {I <: Integer, SIM <: Simulation}

Make forward simulation of the iceflow UDE.
"""
function simulate_iceflow_UDE!(
    θ,
    simulation::SIM,
    cb::SciMLBase.DECallback,
    glacier_idx::I,
) where {I <: Integer, SIM <: Simulation}

    model = simulation.model
    cache = simulation.cache
    params = simulation.parameters

    # Define closure with apply_parametrization inside the function call
    SIA2D_UDE_closure(H, θ, t) = SIA2D_UDE(H, θ, t, simulation)

    iceflow_prob = ODEProblem(
        SIA2D_UDE_closure,
        cache.iceflow.H₀,
        params.simulation.tspan,
        θ;
        tstops=params.solver.tstops,
    )

    iceflow_sol = solve(
        iceflow_prob,
        params.solver.solver,
        callback=cb,
        sensealg=params.UDE.sensealg,
        reltol=params.solver.reltol,
        progress=false,
        maxiters=params.solver.maxiters,
    )
    @assert iceflow_sol.retcode==ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(iceflow_sol.retcode)\""

    # Compute average ice surface velocities for the simulated period
    cache.iceflow.H .= iceflow_sol.u[end]
    cache.iceflow.H .= ifelse.(cache.iceflow.H .> 0.0, cache.iceflow.H , 0.0)

    # Average surface velocity
    Huginn.avg_surface_V(simulation, iceflow_sol.t[end])

    glacier = simulation.glaciers[glacier_idx]

    # Surface topography
    cache.iceflow.S .= glacier.B .+ cache.iceflow.H

    return iceflow_sol
end

"""
Wrapper to pass a parametrization to the SIA2D
"""
function SIA2D_UDE(H::Matrix{R}, θ, t::R, simulation::SIM) where {R <: Real, SIM <: Simulation}
    simulation.model.machine_learning.θ = θ
    return Huginn.SIA2D(H, simulation, t)
end

"""
currently just use for Enzyme
"""
function SIA2D_UDE!(_θ, _dH::Matrix{R}, _H::Matrix{R}, simulation::FunctionalInversion, t::R) where {R <: Real}
    simulation.model.machine_learning.θ = _θ
    Huginn.SIA2D!(_dH, _H, simulation, t)
    return nothing
end
