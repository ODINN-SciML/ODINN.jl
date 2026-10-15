import Huginn.precompute_all_VJPs_laws!, Huginn.run!

"""
    run!(simulation::Inversion)

Run the training process for a given `Inversion` simulation.

# Arguments

  - `simulation::Inversion`: The simulation object containing the parameters and settings for the inversion process.

# Description

This function initiates the training of a Universal Differential Equation (UDE) for the provided simulation. It prints a message indicating the start of the training process, calls the `train_UDE!` function to perform the training, and collects the results in `results_list`. The results are intended to be saved using `Sleipnir.save_results_file!`, but this step is currently commented out and will be enabled once the optimization is working. Finally, the garbage collector is triggered to free up memory.

# Notes

  - The `Sleipnir.save_results_file!` function call is currently commented out and should be enabled once the optimization process is confirmed to be working.
  - The garbage collector is explicitly run using `GC.gc()` to manage memory usage.
"""
function run!(
        simulation::Inversion;
        path::Union{String, Nothing} = nothing,
        file_name::Union{String, Nothing} = nothing,
        save_every_iter::Bool = false,
        path_tb_logger::Union{String, Nothing} = joinpath(
            ODINN.root_dir,
            ".log/",
            Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
        )
)
    # Set expected total number of epochs from beginning for the callback
    simulation.results.stats.niter = sum(simulation.parameters.hyper.epochs)
    # The table of the loss terms costs one forward simulation per glacier
    show_terms = !simulation.parameters.simulation.test_mode
    show_terms && print_loss_terms(stdout,
        loss_terms(simulation.model.trainable_components.θ, simulation),
        "Loss terms at the start:")

    logger = isnothing(path_tb_logger) || simulation.parameters.simulation.test_mode ?
             nothing : TBLogger(path_tb_logger)
    if !(typeof(simulation.parameters.hyper.optimizer) <: Vector)
        # One single optimizer
        sol = train_UDE!(simulation; save_every_iter = save_every_iter, logger = logger)
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
                simulation.model.trainable_components.θ = θ_trained
            end
            sol = train_UDE!(simulation; save_every_iter = save_every_iter, logger = logger)
            # Clear results of previous simulation for fresh start
            if i < length(epochs)
                simulation.results.simulation = Sleipnir.Results{Float64, Int64}[]
            end
        end
        # Put back the original epochs and optimizer values
        simulation.parameters.hyper.optimizer = optimizers
        simulation.parameters.hyper.epochs = epochs
    end

    # Setup final results
    simulation.model.trainable_components.θ = sol.u
    show_terms && print_loss_terms(stdout, loss_terms(sol.u, simulation),
        "Loss terms at the end:")

    simulation.results.stats.niter = length(simulation.results.stats.losses)
    # Final parameters of the optimization
    simulation.results.stats.θ = sol.u
    # Final initial conditions for the simulation
    if haskey(sol.u, :IC)
        simulation.results.stats.initial_conditions = Dict()
        for glacier_id in 1:length(simulation.glaciers)
            glacier = simulation.glaciers[glacier_id]
            simulation.results.stats.initial_conditions[String(glacier.rgi_id)] = initial_thickness(
                simulation.model.trainable_components.θ, simulation, glacier_id)
        end
    end

    # TODO: Save when optimization is working
    # Save results in path is provided
    if !isnothing(path) & !isnothing(file_name)
        ODINN.save_inversion_file!(sol, simulation; path = path, file_name = file_name)
    end

    @everywhere GC.gc() # run garbage collector
end

"""
    train_UDE!(
        simulation::Inversion;
        save_every_iter::Bool = false,
        logger::Union{<: TBLogger, Nothing} = nothing
    )

Trains UDE based on the current Inversion.
"""
function train_UDE!(
        simulation::Inversion;
        save_every_iter::Bool = false,
        logger::Union{<: TBLogger, Nothing} = nothing
)
    optimizer = simulation.parameters.hyper.optimizer
    iceflow_trained = train_UDE!(simulation, optimizer; save_every_iter = save_every_iter, logger = logger)
    return iceflow_trained
end

"""
BFGS optim
"""
function train_UDE!(
        simulation::Inversion,
        optimizer::Union{Optim.LBFGS, Optim.BFGS};
        save_every_iter::Bool = false,
        logger::Union{<: TBLogger, Nothing} = nothing
)
    @info "Optimizing with $(nameof(typeof(optimizer)))"

    # Create batches for inversion training
    simulation_train_loader = generate_batches(simulation)
    # simulation_batch_ids = train_loader.data[1]

    θ = simulation.model.trainable_components.θ
    allowed_keys = (:A, :C, :n, :Y, :U)
    @assert length(intersect(keys(θ), allowed_keys))==1 "The vector of parameters θ should contain at most only one of the following keys: $(allowed_keys)"

    # Get the available workers
    # Workers are always the number of allocated cores to Julia minus one
    workers_list = workers()
    if simulation.parameters.simulation.multiprocessing
        @assert length(workers_list) == (simulation.parameters.simulation.workers - 1) "Number of workers does not match"
    end

    # Simplify API for optimization problem and include data loaded in argument for minibatch
    # The line search of BFGS and LBFGS can try a point where the forward simulation fails
    loss_function(_θ, _simulation) = loss_or_inf(
        loss_iceflow_transient, _θ, only(_simulation.data), pmap)

    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        @assert simulation.parameters.UDE.optim_autoAD == Optimization.AutoZygote() "For the moment only Zygote is supported for the differentiation of the loss function but params.UDE.optim_autoAD = $(simulation.parameters.UDE.optim_autoAD)."
    else
        @info "Optimizing with custom $(typeof(simulation.parameters.UDE.grad)) method"
    end
    loss_function_grad!(_dθ, _θ, _simulation) =
        if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
            grad_loss_iceflow!(_dθ, _θ, only(_simulation), pmap)
        else
            SIA2D_grad!(_dθ, _θ, only(_simulation))
        end
    optf = OptimizationFunction(loss_function, NoAD(), grad = loss_function_grad!)

    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Optim diagnosis callback
    cb(θ, l) =
        let simulation=simulation, logger=logger, save_every_iter=save_every_iter
            callback_diagnosis(θ, l, simulation; save = save_every_iter, tbLogger = logger)
        end

    iceflow_trained = solve(
        optprob,
        simulation.parameters.hyper.optimizer,
        maxiters = simulation.parameters.hyper.epochs,
        allow_f_increases = true,
        callback = cb,
        progress = false
    )

    θ_trained = iceflow_trained.u
    simulation.results.simulation = create_results(
        θ_trained, simulation, pmap; processVelocity = Huginn.V_from_H)

    return iceflow_trained
end

"""
ADAM optim
"""
function train_UDE!(
        simulation::Inversion,
        optimizer::Union{Optim.Adam, Optimisers.Adam};
        save_every_iter::Bool = false,
        logger::Union{<: TBLogger, Nothing} = nothing
)
    @info "Optimizing with ADAM"

    # Create batches for inversion training
    simulation_train_loader = generate_batches(simulation)

    # The variable θ includes all variables to being optimized, including initial conditions
    θ = simulation.model.trainable_components.θ
    allowed_keys = (:A, :C, :n, :Y, :U)
    @assert length(intersect(keys(θ), allowed_keys))==1 "The vector of parameters θ should contain at most only one of the following keys: $(allowed_keys)"

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
        @assert simulation.parameters.UDE.optim_autoAD == Optimization.AutoZygote() "For the moment only Zygote is supported for the differentiation of the loss function but params.UDE.optim_autoAD = $(simulation.parameters.UDE.optim_autoAD)."
    else
        @info "Optimizing with custom $(typeof(simulation.parameters.UDE.grad)) method"
    end
    loss_function_grad!(_dθ, _θ, simulation_loader) =
        if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
            grad_loss_iceflow!(_dθ, _θ, simulation_loader[1], pmap)
        else
            SIA2D_grad!(_dθ, _θ, simulation_loader[1])
        end
    optf = OptimizationFunction(loss_function, NoAD(), grad = loss_function_grad!)

    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Optim diagnosis callback
    cb(θ, l) =
        let simulation=simulation, logger=logger, save_every_iter=save_every_iter
            callback_diagnosis(θ, l, simulation; save = save_every_iter, tbLogger = logger)
        end

    iceflow_trained = solve(
        optprob,
        simulation.parameters.hyper.optimizer,
        maxiters = simulation.parameters.hyper.epochs,
        callback = cb,
        progress = false
    )

    θ_trained = iceflow_trained.u
    simulation.results.simulation = create_results(
        θ_trained, simulation, pmap; processVelocity = Huginn.V_from_H)

    return iceflow_trained
end

"""
    create_results(θ, simulation::Inversion, mappingFct; processVelocity::Union{Nothing, Function} = nothing)

Given the parameters θ, solve the iceflow problem for all the glaciers and aggregate
the results for all of them.
This function is typically used at the end of a training once θ has been optimized
and one wants to run one last forward simulation in order to retrieve statistics
about each of the iceflow problems.

Arguments:

  - `θ`: Parameters to use for the forward simulation.
  - `simulation::Inversion`: Simulation structure that contains all the required information about the inversion.
  - `mappingFct`: Function to use to process the glaciers. Either `map` for a sequential processing or `pmap` for multiprocessing.
  - `processVelocity::Union{Nothing, Function}`: Post processing function to map the ice thickness to the surface velocity. It is called before creating the results. It takes as inputs simulation, ice thickness (matrix) and the associated time and returns 3 variables Vx, Vy, V which are all matrix. Defaults is nothing which means no post processing is applied.
"""
function create_results(θ, simulation::Inversion, mappingFct;
        processVelocity::Union{Nothing, Function} = nothing)
    simulation.model.trainable_components.θ = θ
    simulations = generate_simulation_batches(simulation)
    results = mappingFct(simulations) do simulation
        container = InversionBinder(simulation, simulation.model.trainable_components.θ)
        [_batch_iceflow_UDE(
             container, glacier_idx,
             define_iceflow_prob(simulation.model.trainable_components.θ, simulation, glacier_idx);
             processVelocity = processVelocity
         ) for glacier_idx in 1:length(container.simulation.glaciers)]
    end
    results = merge_batches(results)
    return results
end

"""
    loss_iceflow_transient(θ, simulation::Inversion, mappingFct)

Given the parameters θ, this function:

 1. Solves the iceflow problem for all the glaciers.
 2. Computes the loss function defined as the sum of the loss functions for each of the glaciers.
    The loss function of each glacier depends on the type of loss. Refer to `empirical_loss_function` in
    the UDE parameters for more information. The loss function is transient meaning that the state of the
    glacier is compared to a reference at different time steps over the simulated period.
 3. Return the value of the loss function.

Arguments:

  - `θ`: Parameters to use for the forward simulation.
  - `simulation::Inversion`: Simulation structure that contains all the required information about the inversion.
  - `mappingFct`: Function to use to process the glaciers. Either `map` for a sequential processing or `pmap` for multiprocessing.
"""
function loss_iceflow_transient(θ, simulation::Inversion, mappingFct)
    simulation.model.trainable_components.θ = θ
    simulations = generate_simulation_batches(simulation)
    losses = mappingFct(
        simulation -> parallel_loss_iceflow_transient(
            simulation.model.trainable_components.θ, simulation
        ), simulations)
    losses = merge_batches(losses)
    return sum(losses)
end

"""
    ForwardSolveFailure(retcode)

Error thrown when the solver of the forward ice flow simulation does not succeed.
"""
struct ForwardSolveFailure <: Exception
    retcode::ReturnCode.T
end
function Base.showerror(io::IO, e::ForwardSolveFailure)
    print(io,
        "ForwardSolveFailure: there was an error in the iceflow solver. Returned code is \"$(e.retcode)\"")
end

# `pmap` wraps the error of a task or of a worker
root_exception(e) = e
root_exception(e::RemoteException) = root_exception(e.captured)
root_exception(e::CapturedException) = root_exception(e.ex)

"""
    loss_or_inf(loss, args...)

Call `loss(args...)` and return `Inf` if the forward simulation fails.

A trial point that is too far can make the forward fail: the surface leaves the range of the
mass balance (`Muninn.ElevationOutOfRange`), or the solver fails (`ForwardSolveFailure`).
With an infinite loss, the line search of BFGS and LBFGS takes a shorter step instead of
stopping the inversion. Other errors are not caught.
"""
function loss_or_inf(loss, args...)
    try
        return loss(args...)
    catch err
        ex = root_exception(err)
        ex isa Union{ForwardSolveFailure, Muninn.ElevationOutOfRange} || rethrow()
        @warn "The forward simulation failed, using an infinite loss: $(sprint(showerror, ex))" maxlog=5
        return Inf
    end
end

"""
    grad_loss_iceflow!(dθ, θ, simulation::Inversion, mappingFct)

Compute the gradient with respect to θ for all the glaciers and assign the result in-place to `dθ`.

Arguments:

  - `dθ`: Gradient of the parameters where the computed gradient should be stored.
  - `θ`: Parameters to differentiate.
  - `simulation::Inversion`: Simulation structure that contains all the required information about the inversion.
  - `mappingFct`: Function to use to process the glaciers. Either `map` for a sequential processing or `pmap` for multiprocessing.
"""
function grad_loss_iceflow!(dθ, θ, simulation::Inversion, mappingFct)
    dθ .= grad_loss_iceflow!(θ, simulation::Inversion, mappingFct)
end

"""
    grad_loss_iceflow!(θ, simulation::Inversion, mappingFct)

Compute the gradient with respect to θ for all the glaciers and return the result out-of-place.
See the in-place implementation for more information.
"""
function grad_loss_iceflow!(θ, simulation::Inversion, mappingFct)
    simulation.model.trainable_components.θ = θ
    simulations = generate_simulation_batches(simulation)
    grads = mappingFct(simulations) do simulation
        [grad_parallel_loss_iceflow!(simulation.model.trainable_components.θ, simulation, glacier_idx)
         for glacier_idx in 1:length(simulation.glaciers)]
    end
    # One gradient per glacier, with the layout of the `θ` of its batch: the parameters of each
    # glacier go to the place of the glacier, and the other ones are added
    return aggregate∇θ(merge_batches(grads), θ, simulation.model.trainable_components)
end

"""
    grad_parallel_loss_iceflow!(θ, simulation::Inversion, glacier_idx::Integer)

Compute the gradient with respect to θ for a particular glacier and return the computed gradient.
This function defines the iceflow problem and then calls Zygote to differentiate `batch_loss_iceflow_transient` with respect to θ.
It uses the SciMLSensitivity implementation under the hood to compute the adjoint of the ODE.

The iceflow problem is built inside the differentiated region, from the θ that is
differentiated, so that `u0` (the initial condition) gets a gradient. `u0` must come from
this θ and not from `container.θ`, otherwise the gradient of `θ.C` is zero.
"""
function grad_parallel_loss_iceflow!(θ, simulation::Inversion, glacier_idx::Integer)
    ret, = Zygote.gradient(
        _θ -> batch_loss_iceflow_transient(
            _θ,
            InversionBinder(simulation, _θ),
            glacier_idx,
            define_iceflow_prob(_θ, simulation, glacier_idx)
        )[1], θ)
    return ret
end

"""
    parallel_loss_iceflow_transient(θ, simulation::Inversion)

Loop over a list of glaciers to process.
When multiprocessing is enabled, each call of this function has a dedicated process.
This function calls `batch_loss_iceflow_transient` which returns both the loss and the result structure. The function keeps only the loss.
"""
function parallel_loss_iceflow_transient(θ, simulation::Inversion)
    return [batch_loss_iceflow_transient(
                θ,
                InversionBinder(simulation, θ),
                glacier_idx,
                define_iceflow_prob(θ, simulation, glacier_idx)
            )[1]
            for glacier_idx in 1:length(simulation.glaciers)]
end

"""
    batch_loss_iceflow_transient(
        θ,
        container::InversionBinder,
        glacier_idx::Integer,
        iceflow_prob::ODEProblem,
    )

Solve the ODE, retrieve the results and compute the loss.

`θ` must be the parameter vector that is differentiated, and the loss terms must receive it.
Reading it from `container` drops the part of the gradient that goes through the ODE, with
the same loss value (see `sensitivity.md`).

Arguments:

  - `θ`: Parameter vector being differentiated. This is what the loss terms must receive.
  - `container::InversionBinder`: SciMLStruct that contains the simulation structure and the vector of parameters to optimize. Solver-facing: never read `θ` back out of it.
  - `glacier_idx::Integer`: Index of the glacier.
  - `iceflow_prob::ODEProblem`: Iceflow problem defined as an ODE with respect to time.
"""
function batch_loss_iceflow_transient(
        θ,
        container::InversionBinder,
        glacier_idx::Integer,
        iceflow_prob::ODEProblem
)
    result = _batch_iceflow_UDE(container, glacier_idx, iceflow_prob)

    # Take the simulation off the binder once, untracked: reading it from `container` in the
    # loss gives a wrong gradient (see the `InversionBinder` docstring)
    simulation = @ignore_derivatives container.simulation

    ℓ = observation_loss(
        simulation.parameters.UDE.empirical_loss_function, result, θ, simulation, glacier_idx)
    return ℓ, result
end

"""
    observation_loss(loss_function, result, θ, simulation, glacier_idx)

Value of the loss `loss_function` for the simulated `result` of a glacier: the sum of the loss
at each time of the simulation and of the time aggregated losses.
"""
function observation_loss(loss_function, result, θ, simulation, glacier_idx)
    glacier = simulation.glaciers[glacier_idx]
    t = result.t
    H = result.H

    if loss_uses_velocity(loss_function)
        @assert !isnothing(glacier.velocityData) "Using $(typeof(loss_function)) but no velocityData in the glacier $(glacier.rgi_id)"
        @assert length(glacier.velocityData.date) > 0 "Using $(typeof(loss_function)) but no reference velocity in the results"
    end

    # Discretization for the ice thickness loss term
    tH_ref = tdata(glacier.thicknessData) # If thicknessData is nothing, then tH_ref is an empty vector
    useThickness = length(tH_ref)>0
    H_ref = useThickness ? glacier.thicknessData.H : nothing

    # Discretization for the surface velocity loss term
    tV_ref = tdata(glacier.velocityData, simulation.parameters.simulation.mapping) # If velocityData is nothing, then tV_ref is an empty vector
    useVelocity = length(tV_ref)>0
    Vabs_ref = useVelocity ? glacier.velocityData.vabs : nothing
    Vx_ref = useVelocity ? glacier.velocityData.vx : nothing
    Vy_ref = useVelocity ? glacier.velocityData.vy : nothing

    # Weight of each observation, provided to the loss as a named tuple with one vector per term
    Δt_HV = @ignore_derivatives observation_weights(
        loss_function, tH_ref, tV_ref, glacier_tspan(simulation, glacier_idx))

    if useThickness
        @assert size(H[begin]) == size(H_ref[begin]) "Size of reference and prediction datasets do not match."
    end
    if useVelocity
        @assert size(H[begin]) == size(Vabs_ref[begin]) "Size of reference and prediction datasets do not match."
    end

    β = 2.0
    losses = map(1:length(H)) do τ
        normalization = 1.0
        # normalization = std(H_ref[τ][H_ref[τ] .> 0.0])^β

        tj = t[τ]
        indThickness = findfirst(==(tj), tH_ref)
        indVelocity = findfirst(==(tj), tV_ref)

        # Ignore these parts of the computational graph, otherwise AD fails
        Hr = @ignore_derivatives(isnothing(indThickness) ? nothing : H_ref[indThickness])
        Vr = @ignore_derivatives(isnothing(indVelocity) ? nothing : Vabs_ref[indVelocity])
        Vxr = @ignore_derivatives(isnothing(indVelocity) ? nothing : Vx_ref[indVelocity])
        Vyr = @ignore_derivatives(isnothing(indVelocity) ? nothing : Vy_ref[indVelocity])
        Δtj = @ignore_derivatives((;
            H = isnothing(indThickness) ? 0.0 : Δt_HV.H[indThickness],
            V = isnothing(indVelocity) ? 0.0 : Δt_HV.V[indVelocity]
        ))

        loss(
            loss_function,
            H[τ],
            Hr,
            Vr, Vxr, Vyr,
            t[τ],
            glacier_idx,
            θ,
            simulation,
            prod(size(H[τ]))*normalization,
            Δtj
        )
    end
    time_aggregated_losses = time_aggregated_loss(
        loss_function, H, nothing, nothing, nothing, nothing, t,
        glacier_idx, θ, simulation, prod(size(H[begin]))*1.0, (;))
    return sum(losses) + time_aggregated_losses
end

"""
    loss_terms(θ, simulation::Inversion)

Value of each term of the loss for the parameters `θ`, summed over the glaciers. It runs one
forward simulation per glacier, with no gradient.

Returns a vector with one named tuple `(; term, λ, value, share)` per term of the loss: the
loss term, its weight, its value without the weight, and its part of the total loss
(`λ * value / total`).
"""
function loss_terms(θ, simulation::Inversion)
    lossType = simulation.parameters.UDE.empirical_loss_function
    terms = lossType isa MultiLoss ? lossType.losses : (lossType,)
    λs = lossType isa MultiLoss ? lossType.λs : (1.0,)
    values = zeros(length(terms))
    for glacier_idx in eachindex(simulation.glaciers)
        result = _batch_iceflow_UDE(InversionBinder(simulation, θ), glacier_idx,
            define_iceflow_prob(θ, simulation, glacier_idx))
        for (k, term) in enumerate(terms)
            values[k] += observation_loss(term, result, θ, simulation, glacier_idx)
        end
    end
    total = sum(λs .* values)
    return [(; term, λ, value, share = total > 0 ? λ * value / total : 0.0)
            for (term, λ, value) in zip(terms, λs, values)]
end

# Table of the terms of the loss
function print_loss_terms(io::IO, terms, title)
    println(io, title)
    @printf(io, "  %-32s %8s %12s %7s\n", "term", "λ", "value", "share")
    for t in terms
        @printf(io, "  %-32s %8.3g %12.4g %6.1f%%\n",
            nameof(typeof(t.term)), t.λ, t.value, 100 * t.share)
    end
end

"""
    _batch_iceflow_UDE(
        container::InversionBinder,
        glacier_idx::Integer,
        iceflow_prob::ODEProblem;
        processVelocity::Union{Nothing, Function} = nothing
    )

Define the callbacks to be called by the ODE solver, solve the ODE and create the results.
"""
function _batch_iceflow_UDE(
        container::InversionBinder,
        glacier_idx::Integer,
        iceflow_prob::ODEProblem;
        processVelocity::Union{Nothing, Function} = nothing
)
    params = container.simulation.parameters
    glacier = container.simulation.glaciers[glacier_idx]
    step = params.solver.step
    step_MB = params.simulation.step_MB

    container.simulation.cache = init_cache(
        container.simulation.model, container.simulation, glacier_idx, container.θ)
    container.simulation.model.trainable_components.θ = container.θ

    tspan = glacier_tspan(params, glacier)

    # Define tstops. `define_callback_steps` mutates an array when the window is not a whole
    # number of steps, and the stops do not depend on `θ`
    tstops = @ignore_derivatives Huginn.define_callback_steps(tspan, step)
    tstops = unique(vcat(tstops, params.solver.tstops)) # Merge time steps controlled by `step` with the user provided time steps
    tstopsIceThickness = tdata(glacier.thicknessData)
    tstopsVelocity = tdata(glacier.velocityData, params.simulation.mapping)
    tstopsDiscreteLoss = unique(discreteLossSteps(params.UDE.empirical_loss_function, tspan))
    tstopsAggregatedLoss = unique(discretePostIntegralLossSteps(
        params.UDE.empirical_loss_function, container.simulation, glacier_idx))
    tstops = sort(unique(vcat(tstops, tstopsIceThickness, tstopsVelocity,
        tstopsDiscreteLoss, tstopsAggregatedLoss)))
    # Observation times can fall outside the simulated window, and the solver never reaches
    # them. `create_results` looks every tstop up in `solution.t` and indexes with the
    # result, so keeping an unreachable one makes it index with `nothing`.
    tstops = filter(t -> tspan[1] <= t <= tspan[2], tstops)

    # Create iceflow law callback
    cb_iceflow = Huginn.build_callback(
        container.simulation.model.iceflow,
        container.simulation.cache.iceflow,
        container.simulation.cache.iceflow.glacier_idx,
        container.θ,
        tspan
    )

    # Mass balance is a source term of the ice flow RHS, so there is no callback for it
    cb = CallbackSet(cb_iceflow)

    # Run iceflow UDE for this glacier
    iceflow_sol = simulate_iceflow_UDE!(container, cb, iceflow_prob, tstops)

    # Compute simulation results
    # No need to generate the velocities since this is automatically computed directly inside the loss function when needed
    # Diagnostics only, and it writes into the mass balance cache, so keep it out of the
    # differentiated region
    MB,
    t_MB = Zygote.@ignore_derivatives Huginn.MB_diagnostics(
        container.simulation, iceflow_sol)
    # The bed is only for the results, it is not used by the loss
    B = @ignore_derivatives Huginn.bed(container.simulation, glacier_idx, container.θ)
    return Sleipnir.create_results(
        container.simulation, glacier_idx, iceflow_sol, tstops;
        MB = MB,
        t_MB = t_MB,
        processVelocity = processVelocity,
        B = B
    )
end

"""
    simulate_iceflow_UDE!(
        container::InversionBinder,
        cb::SciMLBase.DECallback,
        iceflow_prob::ODEProblem,
        tstops,
    )

Make a forward simulation of the iceflow UDE.
"""
function simulate_iceflow_UDE!(
        container::InversionBinder,
        cb::SciMLBase.DECallback,
        iceflow_prob::ODEProblem,
        tstops
)
    params = container.simulation.parameters
    # The stops do not depend on `θ`, and a window that is not a whole number of `step_MB`
    # makes them mutate an array
    solver_tstops,
    saveat = @ignore_derivatives Huginn.MB_solver_stops(
        container.simulation, tstops)
    # `dt` only when stepping is fixed: in adaptive mode supplying one overrides the solver's
    # own initial step, and supplying zero aborts the solve.
    step_kw = params.solver.adaptive ? NamedTuple() : (; dt = params.solver.dt)
    # `u0` already carries a derivative: the problem is built from the differentiated θ by the
    # caller, so only `p` is swapped here. Do not rebuild `u0` from `container.θ` — see
    # `grad_parallel_loss_iceflow!`.
    iceflow_prob_remake = remake(iceflow_prob; p = container)
    iceflow_sol = solve(
        iceflow_prob_remake,
        params.solver.supply_eigen_est ?
        Huginn.with_eigen_est(params.solver.solver, container.simulation) :
        params.solver.solver;
        callback = cb,
        sensealg = params.UDE.sensealg,
        reltol = params.solver.reltol,
        abstol = params.solver.scale_abstol ?
                 Huginn.effective_abstol(params.solver.abstol, iceflow_prob.tspan;
            verbose = false) : params.solver.abstol,
        adaptive = params.solver.adaptive,
        saveat = saveat,
        progress = false,
        maxiters = params.solver.maxiters,
        tstops = solver_tstops,
        step_kw...
    )
    iceflow_sol.retcode == ReturnCode.Success ||
        throw(ForwardSolveFailure(iceflow_sol.retcode))

    return iceflow_sol
end

"""
    define_iceflow_prob(
        θ,
        simulation::Inversion,
        glacier_idx::Integer,
    )

Given a `simulation` struct and a `glacier_idx`, build the iceflow problem that has to be solved in the ODE solver.
In practice, the returned iceflow problem is used inside `simulate_iceflow_UDE!` through `remake`.

It is called inside the differentiated region, so that `u0 = evaluate_H₀(θ, …)` has a
derivative and the initial condition can be trained.
"""
function define_iceflow_prob(
        θ,
        simulation::Inversion,
        glacier_idx::Integer
)
    params = simulation.parameters
    # Define initial condition for the inverse simulation
    if haskey(θ, :IC)
        H₀ = initial_thickness(θ, simulation, glacier_idx)
        @assert size(H₀) == size(simulation.glaciers[glacier_idx].H₀)
        # An abstract eltype of `u0` fails far from here (see `σ_zang`), so check it once
        u0_ok = Zygote.@ignore_derivatives isconcretetype(eltype(H₀)) &&
                                           eltype(H₀) <: AbstractFloat
        Zygote.@ignore_derivatives @assert u0_ok "evaluate_H₀ returned eltype $(eltype(H₀)) for glacier $(glacier_idx); u0 must be a concrete float array. Check `initial_condition_filter = $(simulation.parameters.UDE.initial_condition_filter)` for branches that disagree on their return type."

    else
        H₀ = simulation.glaciers[glacier_idx].H₀
    end
    iceflow_prob = ODEProblem(
        SIA2D_UDE!,
        H₀,
        glacier_tspan(simulation, glacier_idx),
        simulation;
        tstops = params.solver.tstops
    )
    return iceflow_prob
end

"""
    precompute_all_VJPs_laws!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation::Inversion,
        glacier_idx::Integer,
        t::Real,
        θ,
    )

Precomputes the vector-Jacobian products (VJPs) for all laws used in
the SIA2D ice flow model for a given glacier, time, and model parameters.

Depending on which target (`U`, `Y`, or neither) is provided in
`SIA2D_model`, this function checks if the corresponding law supports
VJP precomputation and, if so, triggers the appropriate precompute
routine for that law. If neither `U` nor `Y` is provided, precomputes
VJPs for the `A`, `C`, and `n` laws.

# Arguments

  - `SIA2D_model::SIA2Dmodel`: The model containing the configuration and
    laws used for SIA2D ice flow.
  - `SIA2D_cache::SIA2DCache`: A cache object holding intermediate values
    and storage relevant for precomputations.
  - `simulation::Inversion`: Simulation object containing global simulation parameters.
  - `glacier_idx::Integer`: Index of the glacier being simulated.
  - `t::Real`: Current time in the simulation.
  - `θ`: Model parameters or state variables for the simulation step.

# Notes

  - This routine is intended as a preparatory step for manual adjoint.
  - Only laws supporting VJP precomputation are processed.
"""
function precompute_all_VJPs_laws!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation::Inversion,
        glacier_idx::Integer,
        t::Real,
        θ
)
    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint) ||
       isa(simulation.parameters.UDE.grad, DummyAdjoint)
        return nothing
    end
    if isa(simulation.parameters.UDE.grad.VJP_method, EnzymeVJP)
        return nothing
    end
    if SIA2D_model.U_is_provided
        if is_precomputable_law_VJP(SIA2D_model.U)
            precompute_law_VJP(SIA2D_model.U, SIA2D_cache.U, SIA2D_cache.U_prep_vjps,
                simulation, glacier_idx, t, θ)
        end
    elseif SIA2D_model.Y_is_provided
        if is_precomputable_law_VJP(SIA2D_model.Y)
            precompute_law_VJP(SIA2D_model.Y, SIA2D_cache.Y, SIA2D_cache.Y_prep_vjps,
                simulation, glacier_idx, t, θ)
        end
    else
        if is_precomputable_law_VJP(SIA2D_model.A)
            precompute_law_VJP(SIA2D_model.A, SIA2D_cache.A, SIA2D_cache.A_prep_vjps,
                simulation, glacier_idx, t, θ)
        end
        if is_precomputable_law_VJP(SIA2D_model.C)
            precompute_law_VJP(SIA2D_model.C, SIA2D_cache.C, SIA2D_cache.C_prep_vjps,
                simulation, glacier_idx, t, θ)
        end
        if is_precomputable_law_VJP(SIA2D_model.n)
            precompute_law_VJP(SIA2D_model.n, SIA2D_cache.n, SIA2D_cache.n_prep_vjps,
                simulation, glacier_idx, t, θ)
        end
    end
end

"""
Currently just used for Enzyme
"""
function SIA2D_UDE!(_dH::Matrix{<: Real}, _H::Matrix{<: Real}, container::InversionBinder, t::Real)
    Huginn.SIA2D!(_dH, _H, container.simulation, t, container.θ)
    return nothing
end
function SIA2D_UDE!(
        _θ, _dH::Matrix{<: Real}, _H::Matrix{<: Real}, simulation::Inversion, t::Real)
    Huginn.SIA2D!(_dH, _H, simulation, t, _θ)
    return nothing
end
