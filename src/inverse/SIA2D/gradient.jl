export SIA2D_grad!

"""
Inverse with batch
"""
function SIA2D_grad!(dθ, θ, simulation::Inversion)
    simulation.model.trainable_components.θ = θ
    simulations = generate_simulation_batches(simulation)
    loss_grad = pmap(
        simulation -> SIA2D_grad_batch!(simulation.model.trainable_components.θ, simulation), simulations)

    # Retrieve loss function
    losses = getindex.(loss_grad, 1)
    loss = sum(losses)
    # Retrieve gradient
    dθs = getindex.(loss_grad, 2)
    dθs = ODINN.merge_batches(dθs)

    if maximum(norm.(dθs)) > 1e7
        glacier_ids = findall(>(1e7), norm.(dθs))
        for id in glacier_ids
            @warn "Potential unstable gradient for glacier $(simulation.glaciers[id].rgi_id): ‖dθ‖=$(norm(dθs[id])) \n Try reducing the temporal stepsize Δt used for reverse simulation."
        end
    end
    dθs = aggregate∇θ(dθs, θ, simulation.model.trainable_components)

    @assert typeof(θ) == typeof(dθs)
    # @assert norm(sum(dθs)) > 0.0 "‖∑dθs‖=$(norm(sum(dθs))) but should be greater than 0"

    dθ .= dθs
end

"""
    safe_slice(obj, ind::Integer)

Return a sliced object `obj` if `ind > 0`, otherwise return 0.0.
"""
@inline function safe_slice(obj, ind::Integer)
    return ind>0 ? obj[ind] : 0.0
end

"""
Compute gradient glacier per glacier
"""
function SIA2D_grad_batch!(θ, simulation::Inversion)

    # Run forward simulation to build the results
    container = InversionBinder(simulation, θ)
    loss_results = [batch_loss_iceflow_transient(
                        container,
                        glacier_idx,
                        define_iceflow_prob(θ, simulation, glacier_idx)
                    ) for glacier_idx in 1:length(container.simulation.glaciers)]
    loss_per_glacier = getindex.(loss_results, 1)
    loss_val = sum(loss_per_glacier)
    results = getindex.(loss_results, 2)
    simulation.results.simulation = results
    params = simulation.parameters
    tspan = params.simulation.tspan

    dLdθs_vector = Vector{typeof(θ)}()
    loss_function = params.UDE.empirical_loss_function

    for i in 1:length(simulation.glaciers)
        simulation.cache = init_cache(simulation.model, simulation, i, θ)
        simulation.model.trainable_components.θ = θ

        result = simulation.results.simulation[i]

        ## 1- Results from forward simulation
        t = result.t
        Δt = diff(t)
        H = result.H
        glacier = simulation.glaciers[i]

        ## 2- Reference data

        # Discretization for the ice thickness loss term
        tH_ref = tdata(glacier.thicknessData) # If thicknessData is nothing, then tH_ref is an empty vector
        ΔtH = diff(tH_ref)
        useThickness = length(tH_ref)>0
        H_ref = useThickness ? glacier.thicknessData.H : nothing

        # Discretization for the surface velocity loss term
        tV_ref = tdata(glacier.velocityData, params.simulation.mapping) # If velocityData is nothing, then tV_ref is an empty vector
        ΔtV = diff(tV_ref)
        useVelocity = length(tV_ref)>0
        Vabs_ref = useVelocity ? glacier.velocityData.vabs : nothing
        Vx_ref = useVelocity ? glacier.velocityData.vx : nothing
        Vy_ref = useVelocity ? glacier.velocityData.vy : nothing

        # Discretization provided to the loss as a named tuple with the discretization for each term
        Δt_HV = (; H = ΔtH, V = ΔtV)

        ## 3- Determine tstops in the same way as what is done in the forward and check that this matches
        tstops = Huginn.define_callback_steps(tspan, params.solver.step)
        tstopsDiscreteLoss = unique(discreteLossSteps(params.UDE.empirical_loss_function, tspan))
        tstopsAggregatedLoss = unique(discretePostIntegralLossSteps(
            params.UDE.empirical_loss_function, simulation, i))
        tstops = sort(unique(vcat(
            tstops, params.solver.tstops, tH_ref, tV_ref,
            tstopsDiscreteLoss, tstopsAggregatedLoss)))

        @assert length(t) == length(tstops) "The size of tstops does not match with the size of the reference times."
        @assert isapprox(t, tstops, rtol = 1e-7) "Times in tstops and reference times in result do not coincide. Maximum difference is $(maximum(abs.(t-tstops)))"
        if useThickness
            @assert size(H[begin]) == size(H_ref[begin])
        end
        if useVelocity
            @assert size(H[begin]) == size(Vabs_ref[begin])
        end

        # Dimensions
        N = size(result.B)
        k = length(H)
        normalization = 1.0
        dLdθ = zero(θ)

        # Let's compute the forward loss inside the gradient computation for verification purpose
        ℓ = 0.0

        apply_all_callback_laws!(
            simulation.model.iceflow, simulation.cache.iceflow, simulation, i, tspan[2], θ)
        feed_input_cache!(
            simulation.model.iceflow, simulation.cache.iceflow, simulation, i, θ, result)
        precompute_all_VJPs_laws!(
            simulation.model.iceflow, simulation.cache.iceflow, simulation, i, tspan[2], θ)

        if typeof(simulation.parameters.UDE.grad) <: DiscreteAdjoint
            tstopsMB = if simulation.parameters.simulation.use_MB
                tstopsMB = Huginn.define_callback_steps(tspan, simulation.parameters.simulation.step_MB)[2:end] # Discard first time step to be aligned with the forward
                @assert all(map(ti -> ti in t, tstopsMB)) "When using the DiscreteAdjoint the tstops of the MB callback must all be included in the tstops from the results."
                tstopsMB
            else
                []
            end

            # Adjoint setup
            # Define empty object to store adjoint in reverse mode
            λ = [zero(result.B) for _ in 1:k]

            res_backward_loss = map(1:k) do j
                tj = t[j]
                indThickness = findfirst(==(tj), tH_ref)
                indVelocity = findfirst(==(tj), tV_ref)
                Δtj = (;
                    H = isnothing(indThickness) ? 0.0 : safe_slice(Δt_HV.H, indThickness-1),
                    V = isnothing(indVelocity) ? 0.0 : safe_slice(Δt_HV.V, indVelocity-1)
                )
                backward_loss(
                    loss_function,
                    H[j],
                    isnothing(indThickness) ? nothing : H_ref[indThickness],
                    isnothing(indVelocity) ? nothing : Vabs_ref[indVelocity],
                    isnothing(indVelocity) ? nothing : Vx_ref[indVelocity],
                    isnothing(indVelocity) ? nothing : Vy_ref[indVelocity],
                    tj,
                    i,
                    θ,
                    simulation,
                    prod(N) * normalization,
                    Δtj
                )
            end
            # Unzip ∂L∂H, ∂L∂θ at each timestep
            ∂L∂H = first.(res_backward_loss)
            ∂L∂θ = last.(res_backward_loss)

            # Contribution of time aggregated losses
            ∂L∂H_aggregated_loss,
            ∂L∂θ_aggregated_loss = if length(tstopsAggregatedLoss)>0
                indPostIntegralLoss = Sleipnir.indFromT(tspan, tstopsAggregatedLoss, t)
                backward_time_aggregated_loss(
                    loss_function,
                    H[indPostIntegralLoss],
                    nothing,
                    nothing,
                    nothing,
                    nothing,
                    t[indPostIntegralLoss],
                    i,
                    θ,
                    simulation,
                    prod(N) * normalization,
                    (;)
                )
            else
                Vector{Matrix{typeof(H[begin])}}(), zero(θ)
            end

            for j in reverse(1:k)
                tj = t[j]

                indThickness = findfirst(==(tj), tH_ref)
                indVelocity = findfirst(==(tj), tV_ref)
                Δtj = (;
                    H = isnothing(indThickness) ? 0.0 : safe_slice(Δt_HV.H, indThickness-1),
                    V = isnothing(indVelocity) ? 0.0 : safe_slice(Δt_HV.V, indVelocity-1)
                )

                if simulation.parameters.simulation.use_MB && (tj in tstopsMB)
                    λ[j] .+= VJP_λ_∂MB∂H(simulation.parameters.UDE.grad.MB_VJP,
                        λ[j], H[j], simulation, glacier, tj)
                end

                # Compute derivative of local contribution to loss function
                ∂ℓ∂H = ∂L∂H[j]
                ∂ℓ∂θ = ∂L∂θ[j]
                if tj in tstopsAggregatedLoss
                    idx = findfirst(==(tj), tstopsAggregatedLoss)
                    ∂ℓ∂H .+= ∂L∂H_aggregated_loss[idx]
                end

                # Compute loss function for verification purpose
                ℓi = loss(
                    loss_function,
                    H[j],
                    isnothing(indThickness) ? nothing : H_ref[indThickness],
                    isnothing(indVelocity) ? nothing : Vabs_ref[indVelocity],
                    isnothing(indVelocity) ? nothing : Vx_ref[indVelocity],
                    isnothing(indVelocity) ? nothing : Vy_ref[indVelocity],
                    t[j],
                    i,
                    θ,
                    simulation,
                    prod(N) * normalization,
                    Δtj
                ) # Δt is included in each loss call
                ℓ += ℓi

                ### Custom VJP to compute the adjoint
                λ_∂f∂H,
                dH_H = VJP_λ_∂SIA∂H(simulation.parameters.UDE.grad.VJP_method,
                    λ[j], H[j], θ, simulation, tj)

                ### Update adjoint
                if j>1
                    # For ∂ℓ∂H, time discretization is already included in the loss, so no need to multiply by Δt
                    λ[j - 1] .= λ[j] .+ Δt[j - 1] * λ_∂f∂H .+ something(∂ℓ∂H, 0.0)

                    ### Custom VJP for grad of loss function
                    λ_∂f∂θ = VJP_λ_∂SIA∂θ(simulation.parameters.UDE.grad.VJP_method,
                        λ[j - 1], H[j], θ, dH_H, simulation, tj)

                    ### Contribution to the loss
                    dLdθ .+= Δt[j - 1] * λ_∂f∂θ
                end
                # For ∂ℓ∂θ, time discretization is already included in the loss, so no need to multiply by Δt
                dLdθ .+= something(∂ℓ∂θ, 0.0)
            end
            ℓ_agg_loss = time_aggregated_loss(loss_function, H, nothing, nothing, nothing,
                nothing, t, i, θ, simulation, prod(N) * normalization, (;))
            ℓ += ℓ_agg_loss

            ### Check consistency between forward and reverse
            @assert isapprox(ℓ, loss_per_glacier[i]; atol = 0, rtol = 1e-8) "Loss in forward and reverse do not coincide: $(ℓ) (reverse) != $(loss_per_glacier[i]) (forward)"

            # Compute gradient wrt initial condition because this is not taken into account in the loop above
            if haskey(θ, :IC)
                λ₀ = λ[begin]
                s₀ = evaluate_∂H₀(
                    θ,
                    glacier,
                    simulation.parameters.UDE.initial_condition_filter,
                    i
                )
                dLdθ.IC[Symbol("$(i)")] .+= λ₀ .* s₀
            end

            # Contributions of time aggregated loss function terms
            dLdθ .+= ∂L∂θ_aggregated_loss

        elseif typeof(simulation.parameters.UDE.grad) <: ContinuousAdjoint
            """
            Construct continuous interpolator for solution of forward PDE
            TODO: For now we do linear, but of course we can use something more sophisticated (although I don't think will make a huge difference for ice)
            TODO: For an uniform grid, we don't need the Gridded, and actually this is more efficient based on docs
            We construct H_itp with tH_ref rather than t since tH_ref can have small
            numerical errors that make the interpolator to evaluate outside the interval
            Notice this should not be an issue since t ≈ tH_ref
            """
            if simulation.parameters.UDE.grad.interpolation == :Linear
                H_itp = interpolate((t,), H, Gridded(Linear()))
                H_ref_itp = useThickness ?
                            interpolate((tH_ref,), H_ref, Gridded(Linear())) : nothing
                # When there is only one reference velocity data we use a constant interpolator
                Vabs_ref_itp = useVelocity ?
                               (length(tV_ref)>1 ?
                                interpolate((tV_ref,), Vabs_ref, Gridded(Linear())) :
                                t -> only(Vabs_ref)) : nothing
                Vx_ref_itp = useVelocity ?
                             (length(tV_ref)>1 ?
                              interpolate((tV_ref,), Vx_ref, Gridded(Linear())) :
                              t -> only(Vx_ref)) : nothing
                Vy_ref_itp = useVelocity ?
                             (length(tV_ref)>1 ?
                              interpolate((tV_ref,), Vy_ref, Gridded(Linear())) :
                              t -> only(Vy_ref)) : nothing
            else
                throw("Interpolation method for continuous adjoint not defined.")
            end

            # Nodes and weights for numerical quadrature
            t_nodes,
            weights = GaussQuadrature(tspan..., simulation.parameters.UDE.grad.n_quadrature)

            ### Define the reverse ODE problem
            if !((typeof(simulation.parameters.UDE.grad.VJP_method) <: DiscreteVJP) |
                 (typeof(simulation.parameters.UDE.grad.VJP_method) <: EnzymeVJP) |
                 (typeof(simulation.parameters.UDE.grad.VJP_method) <: ContinuousVJP))
                throw("VJP method $(simulation.parameters.UDE.grad.VJP_method) is not supported yet.")
            end
            f_adjoint_rev = let simulation=simulation, H_itp=H_itp, θ=θ
                function (dλ, λ, p, τ)
                    t = -τ
                    λ_∂f∂H,
                    _ = VJP_λ_∂SIA∂H(simulation.parameters.UDE.grad.VJP_method,
                        λ, H_itp(t), θ, simulation, t)
                    dλ .= λ_∂f∂H
                end
            end

            ### Definition of callback to introduce contribution of loss function to adjoint
            t_ref_inv = .-reverse(tstops)
            stop_condition_loss(λ, t,
                integrator) = Sleipnir.stop_condition_tstops(λ, t, integrator, t_ref_inv)

            effect_loss! = let loss_function=loss_function, H_itp=H_itp,
                useThickness=useThickness, useVelocity=useVelocity, H_ref_itp=H_ref_itp,
                Vabs_ref_itp=Vabs_ref_itp, Vx_ref_itp=Vx_ref_itp, Vy_ref_itp=Vy_ref_itp,
                i=i, θ=θ, simulation=simulation, normalization=normalization, N=N,
                tH_ref=tH_ref, tV_ref=tV_ref

                function (t, u)
                    indThickness = findfirst(==(t), tH_ref)
                    indVelocity = findfirst(==(t), tV_ref)
                    Δtj = (;
                        H = isnothing(indThickness) ? 0.0 :
                            safe_slice(Δt_HV.H, indThickness - 1),
                        V = isnothing(indVelocity) ? 0.0 :
                            safe_slice(Δt_HV.V, indVelocity - 1)
                    )
                    ∂ℓ∂H,
                    ∂ℓ∂θ = backward_loss(
                        loss_function,
                        H_itp(t),
                        (useThickness && Δtj.H > 0.0) ? H_ref_itp(t) : nothing,
                        (useVelocity && Δtj.V > 0.0) ? Vabs_ref_itp(t) : nothing,
                        (useVelocity && Δtj.V > 0.0) ? Vx_ref_itp(t) : nothing,
                        (useVelocity && Δtj.V > 0.0) ? Vy_ref_itp(t) : nothing,
                        t,
                        i,
                        θ,
                        simulation,
                        prod(N) * normalization,
                        Δtj
                    )
                    # For ∂ℓ∂H, time discretization is already included in the loss, so no need to multiply by the time step
                    u .+= ∂ℓ∂H
                end
            end
            cb_adjoint_loss = DiscreteCallback(
                stop_condition_loss, integrator -> effect_loss!(-integrator.t, integrator.u))

            # Contribution of aggregated losses
            cb_adjoint_aggregated_loss = if length(tstopsAggregatedLoss)>0
                indPostIntegralLoss = Sleipnir.indFromT(tspan, tstopsAggregatedLoss, t)
                ∂L∂H_aggregated_loss,
                ∂L∂θ_aggregated_loss = backward_time_aggregated_loss(
                    loss_function,
                    H[indPostIntegralLoss],
                    nothing,
                    nothing,
                    nothing,
                    nothing,
                    t[indPostIntegralLoss],
                    i,
                    θ,
                    simulation,
                    prod(N) * normalization,
                    (;)
                )

                effect_aggregated_loss! = let θ=θ, simulation=simulation,
                    ∂L∂H_aggregated_loss=∂L∂H_aggregated_loss,
                    tstopsAggregatedLoss=tstopsAggregatedLoss

                    function (t, u)
                        ind = Sleipnir.indFromT(
                            simulation.parameters.simulation.tspan, [t], tstopsAggregatedLoss)[1]
                        u .+= ∂L∂H_aggregated_loss[ind]
                    end
                end

                stop_condition_aggregated_loss = let tstopsAggregatedLoss=tstopsAggregatedLoss
                    function (λ, t, integrator)
                        Sleipnir.stop_condition_tstops(λ, t, integrator, .-reverse(tstopsAggregatedLoss))
                    end
                end

                DiscreteCallback(
                    stop_condition_aggregated_loss, integrator -> effect_aggregated_loss!(
                        -integrator.t, integrator.u))
            else
                CallbackSet()
            end

            # Mass balance contribution
            effect_MB! = let simulation=simulation, glacier=glacier, H_itp=H_itp
                function (integrator)
                    t = - integrator.t
                    λ_∂MB∂H = VJP_λ_∂MB∂H(simulation.parameters.UDE.grad.MB_VJP,
                        integrator.u, H_itp(t), simulation, glacier, t)
                    integrator.u .+= λ_∂MB∂H
                end
            end
            cb_adjoint_MB = if simulation.parameters.simulation.use_MB
                # For the moment the time stepping used in the loss, and the one for the MB gradient computation must match
                # The plan in the future is to be able to customize the time stepping for the MB gradient computation
                # Cf https://github.com/ODINN-SciML/ODINN.jl/issues/373
                PeriodicCallback(effect_MB!, simulation.parameters.simulation.step_MB;
                    initial_affect = true, final_affect = false) # Exchange the role of initial_affect/final_affect in comparison to the forward
            else
                CallbackSet()
            end
            cb = CallbackSet(cb_adjoint_MB, cb_adjoint_loss, cb_adjoint_aggregated_loss)

            # Final condition
            λ₁ = zero(H[end])

            # Include contribution of loss from last step since this is not accounted for in the discrete callback
            if tspan[2] ∈ tstops
                effect_loss!(tspan[2], λ₁)
            end
            # Define ODE Problem with time in reverse
            adjoint_PDE_rev = ODEProblem(
                f_adjoint_rev,
                λ₁,
                .-reverse(tspan)
            )

            tstops_adjoint = sort(unique(vcat(t_ref_inv, - t_nodes)))

            # Solve reverse adjoint PDE with dense output
            sol_rev = solve(
                adjoint_PDE_rev,
                callback = cb,
                tstops = tstops_adjoint,
                simulation.parameters.UDE.grad.solver,
                dtmax = simulation.parameters.UDE.grad.dtmax,
                reltol = simulation.parameters.UDE.grad.reltol,
                abstol = simulation.parameters.UDE.grad.abstol,
                maxiters = simulation.parameters.solver.maxiters
            )
            @assert sol_rev.retcode == ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(sol_rev.retcode)\""

            ### Numerical integration using quadrature to compute gradient
            # Contribution of the loss function due to ∂l∂θ
            Δtj = (; H = 1.0, V = 1.0) # Don't need to provide the time steps since we are using a quadrature and this is weighted
            res_backward_loss = map(
                t -> backward_loss(
                    loss_function,
                    H_itp(t),
                    useThickness ? H_ref_itp(t) : nothing,
                    useVelocity ? Vabs_ref_itp(t) : nothing,
                    useVelocity ? Vx_ref_itp(t) : nothing,
                    useVelocity ? Vy_ref_itp(t) : nothing,
                    t,
                    i,
                    θ,
                    simulation,
                    prod(N) * normalization,
                    Δtj
                ),
                t_nodes)
            # Unzip ∂L∂θ at each timestep
            ∂L∂θ = last.(res_backward_loss)

            # Final integration of the loss
            if typeof(simulation.parameters.UDE.grad.VJP_method) <:
               Union{DiscreteVJP, EnzymeVJP, ContinuousVJP}
                for j in 1:length(t_nodes)
                    λ_sol = sol_rev(- t_nodes[j])
                    _H = H_itp(t_nodes[j])
                    λ_∂f∂θ = VJP_λ_∂SIA∂θ(simulation.parameters.UDE.grad.VJP_method,
                        λ_sol, _H, θ, nothing, simulation, t_nodes[j])
                    dLdθ .+= weights[j] .* (λ_∂f∂θ .+ ∂L∂θ[j])
                end
            else
                throw("VJP method $(simulation.parameters.UDE.grad.VJP_method) is not supported yet.")
            end

            # Compute gradient wrt initial condition because this is not taken into account in the quadrature
            if haskey(θ, :IC)
                λ₀ = sol_rev(-tspan[1])
                s₀ = evaluate_∂H₀(
                    θ,
                    glacier,
                    simulation.parameters.UDE.initial_condition_filter,
                    i
                )
                dLdθ.IC[Symbol("$(i)")] .+= λ₀ .* s₀
            end

            # Contributions of discrete loss function terms such as the regularization on the initial condition
            for t in tstopsDiscreteLoss
                Δtj = (; H = 0.0, V = 0.0) # Set to zero because we want to compute only the contributions of the discrete loss terms
                dLdθ .+= backward_loss(
                    loss_function,
                    H_itp(t),
                    useThickness ? H_ref_itp(t) : nothing,
                    useVelocity ? Vabs_ref_itp(t) : nothing,
                    useVelocity ? Vx_ref_itp(t) : nothing,
                    useVelocity ? Vy_ref_itp(t) : nothing,
                    t,
                    i,
                    θ,
                    simulation,
                    prod(N) * normalization,
                    Δtj
                )[2]
            end

            # Contributions of time aggregated loss function terms
            dLdθ .+= ∂L∂θ_aggregated_loss

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
