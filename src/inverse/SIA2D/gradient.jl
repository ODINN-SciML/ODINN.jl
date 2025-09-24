export SIA2D_grad!

"""
Inverse with batch
"""
function SIA2D_grad!(dθ, θ, simulation::FunctionalInversion)

    @assert simulation.parameters.solver.save_everystep "Forward solution needs to be stored in dense mode (ie save_everystep should be set to true), for gradient computation."

    simulations = generate_simulation_batches(simulation)
    loss_grad = pmap(simulation -> SIA2D_grad_batch!(θ, simulation), simulations)

    # Retrieve loss function
    losses = getindex.(loss_grad, 1)
    loss = sum(losses)
    # Retrieve gradient
    dθs  = getindex.(loss_grad, 2)
    dθs = ODINN.merge_batches(dθs)

    if maximum(norm.(dθs)) > 1e7
        glacier_ids = findall(>(1e7), norm.(dθs))
        for id in glacier_ids
            @warn "Potential unstable gradient for glacier $(simulation.glaciers[id].rgi_id): ‖dθ‖=$(norm(dθs)[id]) \n Try reducing the temporal stepsize Δt used for reverse simulation."
        end
    end

    @assert typeof(θ) == typeof(sum(dθs))
    # @assert norm(sum(dθs)) > 0.0 "‖∑dθs‖=$(norm(sum(dθs))) but should be greater than 0"

    dθ .= sum(dθs)
end

"""
Compute gradient glacier per glacier
"""
function SIA2D_grad_batch!(θ, simulation::FunctionalInversion)

    # Run forward simulation to build the results
    container = FunctionalInversionBinder(simulation, θ)
    loss_results = [batch_loss_iceflow_transient(
            container,
            glacier_idx,
            define_iceflow_prob(θ, simulation, glacier_idx),
        ) for glacier_idx in 1:length(container.simulation.glaciers)]
    loss_val = sum(getindex.(loss_results, 1))
    results = getindex.(loss_results, 2)
    simulation.results.simulation = results
    tspan = simulation.parameters.simulation.tspan

    # Let's compute the forward loss inside gradient
    ℓ = 0.0
    dLdθs_vector = []
    loss_function = simulation.parameters.UDE.empirical_loss_function

    for i in 1:length(simulation.glaciers)

        simulation.cache = init_cache(simulation.model, simulation, i, simulation.parameters)
        simulation.model.machine_learning.θ = θ

        result = simulation.results.simulation[i]

        # Results from forward simulation
        t = result.t
        Δt = diff(t)
        H = result.H
        glacier = simulation.glaciers[i]

        # Reference data
        t_ref = glacier.thicknessData.t
        H_ref = glacier.thicknessData.H

        @assert t ≈ t_ref "Reference times of simulation and reference data do not coincide."
        @assert length(H) == length(H_ref)
        @assert size(H[begin]) == size(H_ref[begin])

        # Dimensions
        N = size(result.B)
        k = length(H)
        normalization = 1.0
        dLdθ = Enzyme.make_zero(θ)

        if typeof(simulation.parameters.UDE.grad) <: DiscreteAdjoint

            tstopsMB = if simulation.parameters.simulation.use_MB
                nSteps = Int(round((tspan[2]-tspan[1])/simulation.parameters.solver.step))
                tstopsMB = (tspan[1] .+ collect(1:nSteps) .* simulation.parameters.solver.step)
                @assert all(map(ti -> ti in t, tstopsMB)) "When using the DiscreteAdjoint the tstops of the MB callback must all be included in the tstops from the results."
                tstopsMB
            else [] end

            # Adjoint setup
            # Define empty object to store adjoint in reverse mode
            λ  = [Enzyme.make_zero(result.B) for _ in 1:k]

            res_backward_loss = map(j -> backward_loss(
                    loss_function,
                    H[j],
                    H_ref[j],
                    t[j],
                    glacier,
                    θ,
                    simulation;
                    normalization = prod(N) * normalization,
                ), 1:k)
            # Unzip ∂L∂H, ∂L∂θ at each timestep
            ∂L∂H, ∂L∂θ = map(x -> collect(x), zip(res_backward_loss...))

            for j in reverse(2:k)
                tj = t[j]

                if simulation.parameters.simulation.use_MB && (tj in tstopsMB)
                    λ[j] .+= VJP_λ_∂MB∂H(simulation.parameters.UDE.grad.MB_VJP, λ[j], H[j], simulation, glacier, tj)
                end

                # Compute derivative of local contribution to loss function
                ∂ℓ∂H = ∂L∂H[j]
                ∂ℓ∂θ = ∂L∂θ[j]

                # Compute loss function for verification purpose
                ℓi = loss(
                    loss_function,
                    H[j],
                    H_ref[j],
                    t[j],
                    glacier,
                    θ,
                    simulation;
                    normalization = prod(N) * normalization,
                )
                ℓ += Δt[j-1]*ℓi

                ### Custom VJP to compute the adjoint
                λ_∂f∂H, dH_H = VJP_λ_∂SIA∂H(simulation.parameters.UDE.grad.VJP_method, λ[j], H[j], θ, simulation, tj)

                ### Update adjoint
                λ[j-1] .= λ[j] .+ Δt[j-1] .* (λ_∂f∂H .+ ∂ℓ∂H)

                ### Custom VJP for grad of loss function
                λ_∂f∂θ = VJP_λ_∂SIA∂θ(simulation.parameters.UDE.grad.VJP_method, λ[j-1], H[j], θ, dH_H, simulation, tj)

                ### Update gradient
                # @assert ℓ ≈ loss_val "Loss in forward and reverse do not coincide: $(ℓ) != $(loss_val)"

                # Contribution to the loss
                dLdθ .+= Δt[j-1] .* (isnothing(∂ℓ∂θ) ? λ_∂f∂θ : λ_∂f∂θ .+ ∂ℓ∂θ)
            end

            # Contribution of initial condition to loss function
            if haskey(θ, :IC)
                λ₀ = λ[begin]
                # This contribution will come from the regularization on the initial condition
                ∂L∂H₀ = 0.0
                glacier_id = Symbol("$(glacier.rgi_id)")
                s₀ = evaluate_∂H₀(
                    θ,
                    glacier,
                    simulation.parameters.UDE.initial_condition_filter
                    )
                dLdθ.IC[glacier_id] .+= λ₀ .* s₀ .+ ∂L∂H₀
            end

        elseif typeof(simulation.parameters.UDE.grad) <: ContinuousAdjoint
            """
            Construct continuous interpolator for solution of forward PDE
            TODO: For now we do linear, but of course we can use something more sophisticated (although I don't think will make a huge difference for ice)
            TODO: For an uniform grid, we don't need the Gridded, and actually this is more efficient based on docs
            We construct H_itp with t_ref rather than t since t_ref can have small
            numerical errors that make the interpolator to evaluate outside the interval
            Notice this should not be an issue since t ≈ t_ref
            """
            if simulation.parameters.UDE.grad.interpolation == :Linear
                H_itp = interpolate((t_ref,), H, Gridded(Linear()))
                H_ref_itp = interpolate((t_ref,), H_ref, Gridded(Linear()))
            else
                throw("Interpolation method for continuous adjoint not defined.")
            end

            # Nodes and weights for numerical quadrature
            t_nodes, weights = GaussQuadrature(tspan..., simulation.parameters.UDE.grad.n_quadrature)

            ### Define the reverse ODE problem
            if !( (typeof(simulation.parameters.UDE.grad.VJP_method) <: DiscreteVJP) | (typeof(simulation.parameters.UDE.grad.VJP_method) <: EnzymeVJP) | (typeof(simulation.parameters.UDE.grad.VJP_method) <: ContinuousVJP) )
                throw("VJP method $(simulation.parameters.UDE.grad.VJP_method) is not supported yet.")
            end
            f_adjoint_rev = let simulation=simulation, H_itp=H_itp, θ=θ
                function (dλ, λ, p, τ)
                    t = -τ
                    λ_∂f∂H, _ = VJP_λ_∂SIA∂H(simulation.parameters.UDE.grad.VJP_method, λ, H_itp(t), θ, simulation, t)
                    dλ .= λ_∂f∂H
                end
            end

            ### Definition of callback to introduce contribution of loss function to adjoint
            t_ref_inv = .-reverse(t_ref)
            stop_condition_loss(λ, t, integrator) = Sleipnir.stop_condition_tstops(λ, t, integrator, t_ref_inv)
            effect_loss! = let loss_function=loss_function, H_itp=H_itp, H_ref_itp=H_ref_itp, glacier=glacier, θ=θ, simulation=simulation, normalization=normalization, N=N
                function (integrator)
                    t = - integrator.t
                    ∂ℓ∂H, ∂ℓ∂θ = backward_loss(
                        loss_function,
                        H_itp(t),
                        H_ref_itp(t),
                        t,
                        glacier,
                        θ,
                        simulation;
                        normalization = prod(N) * normalization
                    )
                    integrator.u .= integrator.u .+ simulation.parameters.simulation.step .* ∂ℓ∂H
                end
            end
            cb_adjoint_loss = DiscreteCallback(stop_condition_loss, effect_loss!)
            effect_MB! = let simulation=simulation, glacier=glacier, H_itp=H_itp
                function (integrator)
                    t = - integrator.t
                    λ_∂MB∂H = VJP_λ_∂MB∂H(simulation.parameters.UDE.grad.MB_VJP, integrator.u, H_itp(t), simulation, glacier, t)
                    integrator.u .+= λ_∂MB∂H
                end
            end
            cb_adjoint_MB = if simulation.parameters.simulation.use_MB
                nSteps = Int(round((tspan[2]-tspan[1])/simulation.parameters.solver.step))
                tstopsMB = - (tspan[1] .+ collect(1:nSteps) .* simulation.parameters.solver.step)
                stop_condition_MB(λ, t, integrator) = Sleipnir.stop_condition_tstops(λ, t, integrator, tstopsMB)
                # For the moment the time stepping used in the loss, and the one for the MB gradient computation must match
                # The plan in the future is to be able to customize the time stepping for the MB gradient computation
                # Cf https://github.com/ODINN-SciML/ODINN.jl/issues/373
                DiscreteCallback(stop_condition_MB, effect_MB!)
            else
                CallbackSet()
            end
            cb = CallbackSet(cb_adjoint_MB, cb_adjoint_loss)

            # Final condition
            λ₁ = Enzyme.make_zero(H[end])

            # Include contribution of loss from last step since this is not accounted for in the discrete callback
            if tspan[2] ∈ t_ref
                t_final = tspan[2]
                ∂ℓ∂H, ∂ℓ∂θ = backward_loss(
                    loss_function,
                    H_itp(t_final),
                    H_ref_itp(t_final),
                    t_final,
                    glacier,
                    θ,
                    simulation;
                    normalization = prod(N) * normalization
                    )
                λ₁ .+= simulation.parameters.simulation.step .* ∂ℓ∂H
            end
            # Define ODE Problem with time in reverse
            adjoint_PDE_rev = ODEProblem(
                f_adjoint_rev,
                λ₁,
                .-reverse(tspan)
                )

            # Solve reverse adjoint PDE with dense output
            sol_rev = solve(
                adjoint_PDE_rev,
                callback = cb,
                # saveat=t_nodes_rev, # dont use this!
                dense = true,
                save_everystep = true,
                tstops = t_ref_inv,
                simulation.parameters.UDE.grad.solver,
                dtmax = simulation.parameters.UDE.grad.dtmax,
                reltol = simulation.parameters.UDE.grad.reltol,
                abstol = simulation.parameters.UDE.grad.abstol,
                maxiters = simulation.parameters.solver.maxiters,
                )
            @assert sol_rev.retcode == ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(sol_rev.retcode)\""

            ### Numerical integration using quadrature to compute gradient
            # Contribution of the loss function due to ∂l∂θ
            res_backward_loss = map(t ->
                backward_loss(
                    loss_function,
                    H_itp(t),
                    H_ref_itp(t),
                    t,
                    glacier,
                    θ,
                    simulation;
                    normalization = prod(N) * normalization,
                    ),
                t_nodes)
            # Unzip ∂L∂H, ∂L∂θ at each timestep
            _, ∂L∂θ = map(x -> collect(x), zip(res_backward_loss...))

            # Final integration of the loss
            if (typeof(simulation.parameters.UDE.grad.VJP_method) <: DiscreteVJP) | (typeof(simulation.parameters.UDE.grad.VJP_method) <: EnzymeVJP) | (typeof(simulation.parameters.UDE.grad.VJP_method) <: ContinuousVJP)
                for j in 1:length(t_nodes)
                    λ_sol = sol_rev(- t_nodes[j])
                    _H = H_itp(t_nodes[j])
                    λ_∂f∂θ = VJP_λ_∂SIA∂θ(simulation.parameters.UDE.grad.VJP_method, λ_sol, _H, θ, nothing, simulation, t_nodes[j])
                    dLdθ .+= weights[j] .* (λ_∂f∂θ .+ ∂L∂θ[j])
                end
            else
                throw("VJP method $(simulation.parameters.UDE.grad.VJP_method) is not supported yet.")
            end

            # Contribution of initial condition to loss function
            if haskey(θ, :IC)
                λ₀ = sol_rev(-tspan[1])
                # This contribution will come from the regularization on the initial condition
                ∂L∂H₀ = 0.0
                glacier_id = Symbol("$(glacier.rgi_id)")
                s₀ = evaluate_∂H₀(
                    θ,
                    glacier,
                    simulation.parameters.UDE.initial_condition_filter
                    )
                dLdθ.IC[glacier_id] .+= λ₀ .* s₀ .+ ∂L∂H₀
            end

        elseif typeof(simulation.parameters.UDE.grad) <: DummyAdjoint
            if isnothing(simulation.parameters.UDE.grad.grad_function)
                dLdθ .+= maximum(abs.(θ)) .* rand(Float64, size(θ))
            else
                dLdθ .+= simulation.parameters.UDE.grad.grad_function(θ)
            end
        else
            throw("Adjoint method $(simulation.parameters.UDE.grad) is not supported yet.")
        end

        # Return final evaluations of gradient
        push!(dLdθs_vector, dLdθ)
    end

    return loss_val, dLdθs_vector

end

"""
Gauss Quadratrue for numerical integration
"""
function GaussQuadrature(t₀, t₁, n_quadrature::Int)
    # Ignore AD here since FastGaussQuadrature is using mutating arrays
    nodes, weigths = gausslegendre(n_quadrature)
    nodes = (t₀+t₁)/2 .+ nodes * (t₁-t₀)/2
    weigths = (t₁-t₀) / 2 * weigths
    return nodes, weigths
end
