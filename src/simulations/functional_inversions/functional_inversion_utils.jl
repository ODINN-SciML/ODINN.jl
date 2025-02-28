

"""
run!(simulation::FunctionalInversion)

In-place run of the model. 
"""
function run!(simulation::FunctionalInversion)

    println("Running training of UDE...\n")
    results_list = train_UDE!(simulation)

    # Setup final results
    simulation.stats.niter = length(simulation.stats.losses)

    # TODO: Save when optimization is working
    # Sleipnir.save_results_file!(results_list, simulation)

    @everywhere GC.gc() # run garbage collector

end

"""
train_UDE!(simulation::FunctionalInversion) 

Trains UDE based on the current FunctionalInversion.
"""
function train_UDE!(simulation::FunctionalInversion) 
    
    # Create batches for inversion training 
    train_loader = generate_batches(simulation)

    θ = simulation.model.machine_learning.θ

    # Simplify API for optimization problem and include data loaded in argument for minibatch
    loss_function(θ, glacier_data_loader) = loss_iceflow(θ, glacier_data_loader.data[1], simulation)
    
    if isnothing(simulation.parameters.UDE.grad)
        optf = OptimizationFunction(loss_function, simulation.parameters.UDE.optim_autoAD)
    else
        @warn "Using custom grad function."
        # Custom grad API for optimization problem
        loss_iceflow_grad!(dθ, θ, glacier_data_loader) = simulation.parameters.UDE.grad(dθ, θ; simulation=simulation)
        optf = OptimizationFunction(loss_function, NoAD(), grad=loss_iceflow_grad!)
    end

    optprob = OptimizationProblem(optf, θ, train_loader)

    # Plot callback 
    if isnothing(simulation.parameters.UDE.target)
        cb_plots = (θ, l) -> false 
    elseif simulation.parameters.UDE.target == "A"
        cb_plots = (θ, l) -> false 
        # This other option returns weird error right now, commenting for now
        # cb_plots(θ, l) = callback_plots_A(θ, l, simulation) # TODO: make this more customizable 
    else
        raise("Simulation target not defined.")
    end
    # Training diagnosis callback
    cb_diagnosis(θ, l) = callback_diagnosis(θ, l, simulation)
    # Combined callback
    cb(θ, l) = CallbackOptimizationSet(θ, l; callbacks=(cb_plots, cb_diagnosis))
  
    println("Training iceflow UDE...")
    
    iceflow_trained = solve(optprob, 
                            simulation.parameters.hyper.optimizer, 
                            maxiters=simulation.parameters.hyper.epochs,
                            callback=cb,
                            progress=true)

    return iceflow_trained
end

function loss_iceflow(θ, batch_ids::Vector{I}, simulation::FunctionalInversion) where {I <: Integer} 

    # simulation.model.machine_learning.θ = θ # update model parameters
    predict_iceflow!(θ, simulation, batch_ids)

    # Compute loss function for the full batch
    let l_V = 0.0, l_H =  0.0
    for result in simulation.results
        # Get ice thickness from the reference dataset
        H_ref = result.H_glathida
        # isnothing(H_ref) ? continue : nothing
        # Get ice velocities for the reference dataset
        Vx_ref = result.Vx_ref
        Vy_ref = result.Vy_ref
        # Get ice thickness from the UDE predictions
        H = result.H
        # Get ice velocities prediction from the UDE
        Vx_pred = result.Vx
        Vy_pred = result.Vy

        if simulation.parameters.UDE.scale_loss 
            # Ice thickness
            if !isnothing(H_ref)
                normHref = mean(H_ref.^2)^0.5
                l_H_loc = Flux.Losses.mse(H, H_ref; agg=mean) 
                l_H += normHref^(-1) * l_H_loc
            end
            # Ice surface velocities
            normVref = mean(Vx_ref.^2 .+ Vy_ref.^2)^0.5
            l_V_loc = Flux.Losses.mse(Vx_pred, inn1(Vx_ref); agg=mean) + Flux.Losses.mse(Vy_pred, inn1(Vy_ref); agg=mean)
            l_V += normVref^(-1) * l_V_loc
        else
            # Ice thickness
            if !isnothing(H_ref)
                l_H += Flux.Losses.mse(H[H_ref .!= 0.0], H_ref[H_ref.!= 0.0]; agg=mean) 
            end
            # Ice surface velocities
            l_V += Flux.Losses.mse(V_pred[V_ref .!= 0.0], V_ref[V_ref.!= 0.0]; agg=mean)
        end
    end # for

    loss_type = simulation.parameters.UDE.loss_type
    @assert (loss_type == "H" || loss_type == "V" || loss_type == "HV") "Invalid `loss_type`. Needs to be 'H', 'V' or 'HV'"
    if loss_type == "H"
        l_tot = l_H/length(simulation.results)
    elseif loss_type == "V"
        l_tot = l_V/length(simulation.results) 
    elseif loss_type == "HV"
        l_tot = (l_V + l_H)/length(simulation.results)
    end

    return l_tot
    end # let
end

function predict_iceflow!(θ, simulation::FunctionalInversion, batch_ids::Vector{I}) where {I <: Integer}
    # Train UDE in parallel
    simulation.results = pmap((batch_id) -> batch_iceflow_UDE(θ, simulation, batch_id), batch_ids)
    # println("All batches finished")
end

function batch_iceflow_UDE(θ, simulation::FunctionalInversion, batch_id::I) where {I <: Integer}
    
    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[batch_id]

    # glacier_id = isnothing(glacier.gdir) ? "unnamed" : glacier.rgi_id
    # println("Processing glacier: ", glacier_id)
    
    # Initialize glacier ice flow model
    initialize_iceflow_model(model.iceflow[batch_id], batch_id, glacier, params)

    params.solver.tstops =  @ignore_derivatives Huginn.define_callback_steps(params.simulation.tspan, params.solver.step)
    stop_condition(u,t,integrator) = Sleipnir.stop_condition_tstops(u,t,integrator, params.solver.tstops) #closure
    function action!(integrator)
        if params.simulation.use_MB 
            # Compute mass balance
            @ignore_derivatives begin 
                MB_timestep!(model, glacier, params.solver.step, integrator.t; batch_id = batch_id)
                apply_MB_mask!(integrator.u, glacier, model.iceflow[batch_id])
            end
        end
        # Apply parametrization
        apply_UDE_parametrization!(θ, simulation, integrator, batch_id)
    end
    
    cb_MB = DiscreteCallback(stop_condition, action!)

    # Run iceflow UDE for this glacier
    du = params.simulation.use_iceflow ? Huginn.SIA2D : Huginn.noSIA2D
    iceflow_sol = simulate_iceflow_UDE!(θ, simulation, model, params, cb_MB, batch_id; du = du)

    # println("simulation finished for $batch_id")

    # Update simulation results
    results =  Sleipnir.create_results(simulation, batch_id, iceflow_sol, nothing; light=true, batch_id = batch_id)

    # println("Batch $batch_id finished!")

    return results
end


"""
function simulate_iceflow_UDE!(
    θ,
    simulation::SIM, 
    model::Sleipnir.Model, 
    params::Sleipnir.Parameters, 
    cb::DiscreteCallback,
    batch_id::I; 
    du = Huginn.SIA2D) where {I <: Integer, SIM <: Simulation}

Make forward simulation of the iceflow UDE determined in `du`.
"""
function simulate_iceflow_UDE!(
    θ,
    simulation::SIM, 
    model::Sleipnir.Model, 
    params::Sleipnir.Parameters, 
    cb::DiscreteCallback,
    batch_id::I; 
    du = Huginn.SIA2D) where {I <: Integer, SIM <: Simulation}

    # TODO: make this more general
    apply_UDE_parametrization!(θ, simulation, nothing, batch_id)
    SIA2D_UDE_closure(H, θ, t) = SIA2D_UDE(H, θ, t, simulation, batch_id)

    iceflow_prob = ODEProblem(SIA2D_UDE_closure, model.iceflow[batch_id].H, params.simulation.tspan, tstops=params.solver.tstops, θ)
    iceflow_sol = solve(iceflow_prob, 
                        params.solver.solver, 
                        callback=cb,
                        tstops=params.solver.tstops, 
                        u0=model.iceflow[batch_id].H₀, 
                        p=θ,
                        sensealg=params.UDE.sensealg,
                        reltol=params.solver.reltol, 
                        save_everystep=false,  
                        progress=false)

    # Compute average ice surface velocities for the simulated period
    model.iceflow[batch_id].H = iceflow_sol.u[end]
    model.iceflow[batch_id].H = ifelse.(model.iceflow[batch_id].H .> 0.0, model.iceflow[batch_id].H , 0.0)

    # Average surface velocity
    Huginn.avg_surface_V(simulation; batch_id = batch_id)

    glacier = simulation.glaciers[batch_id]

    # Surface topography
    model.iceflow[batch_id].S = glacier.B .+ model.iceflow[batch_id].H

    return iceflow_sol
end

function apply_UDE_parametrization!(θ, simulation::FunctionalInversion, integrator, batch_id::I) where {I <: Integer}
    # We load the ML model with the parameters
    U = simulation.model.machine_learning.NN_f(θ)
    # We generate the ML parametrization based on the target
    if simulation.parameters.UDE.target == "A"
        A = predict_A̅(U, [mean(simulation.glaciers[batch_id].climate.longterm_temps)])[1]
        simulation.model.iceflow[batch_id].A[] = A
    # elseif simulation.parameters.UDE.target == "D"
    #     parametrization = U()
    end
end

# Wrapper to pass a parametrization to the SIA2D
function SIA2D_UDE(H::Matrix{R}, θ, t::R, simulation::SIM, batch_id::I) where {R <: Real, I <: Integer, SIM <: Simulation}
    return Huginn.SIA2D(H, simulation, t; batch_id = batch_id)
end