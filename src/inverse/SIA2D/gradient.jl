export SIA2D_grad!
export generate_glacier_prediction!

"""
Inverse with batch
"""
function SIA2D_grad!(dθ, θ, simulation::FunctionalInversion)

    # Run forward simulation in dense mode
    # batch_ids = glacier_data_loader[1] # e.g., [2, 1]
    # Maybe this first forward run is not neccesary, since the adjoint will compute this same stuff.
    # l = loss_iceflow_transient(θ, simulation)
    #l = loss_iceflow_transient(θ, glaciers, results, SIA2D_models, simulation.params)

    @assert simulation.parameters.solver.save_everystep "Forward solution needs to be stored in dense mode (ie save_everystep should be set to true), for gradient computation."

    # glacier_results_ids = map(batch_id -> Sleipnir.get_result_id_from_rgi(batch_id, simulation), batch_ids)
    # TODO: move out this from here and make the re-arangement of the results outside

    # results = simulation.results[glacier_results_ids]
    # glaciers = simulation.glaciers
    # SIA2D_models = simulation.model.iceflow

    simulations = ODINN.generate_simulation_batches(simulation)
    loss_grad = pmap(simulation -> SIA2D_grad_batch!(θ, simulation), simulations)

    # Retrieve loss function
    losses = getindex.(loss_grad, 1)
    loss = sum(losses)
    # Retrive gradient
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
Inverse by glacier
"""
function SIA2D_grad_batch!(θ, simulation::FunctionalInversion)

    # Run forward simulation to trigger Result
    loss_val = loss_iceflow_transient(θ, simulation)
    # Let's compute the forward loss inside gradient
    ℓ = 0.0
    # Extract relevant data
    # glacier_results_id = Sleipnir.get_result_id_from_rgi(batch_id, simulation)
    dLdθs_vector = []

    for i in 1:length(simulation.glaciers)

        result = simulation.results[i]

        # Results from forward simulation
        t = result.t
        Δt = diff(t)
        H = result.H

        # Reference data
        t_ref = simulation.glaciers[i].thicknessData.t
        H_ref = simulation.glaciers[i].thicknessData.H

        @assert t ≈ t_ref "Reference times of simulation and reference data do not coincide."
        @assert length(H) == length(H_ref)
        @assert size(H[begin]) == size(H_ref[begin])

        # Dimensions
        N = size(result.B)
        k = length(H)
        t₀ = simulation.parameters.simulation.tspan[1]
        normalization = 1.0
        dLdθ = Enzyme.make_zero(θ)

        if typeof(simulation.parameters.UDE.grad) <: DiscreteAdjoint

            # Adjoint setup
            # Define empty object to store adjoint in reverse mode
            λ  = [Enzyme.make_zero(result.B) for _ in 1:k]
            dH_λ = [Enzyme.make_zero(H[1]) for _ in 1:k]

            ∂L∂H = backward_loss(simulation.parameters.UDE.empirical_loss_function, H, H_ref; normalization=prod(N)*normalization)

            for j in reverse(2:k)

                # β = 2.0
                # normalization = std(H_ref[j][H_ref[j] .> 0.0])^β
                # Compute derivative of local contribution to loss function
                ∂ℓ∂H = ∂L∂H[j]
                ℓ += Δt[j-1] * loss(simulation.parameters.UDE.empirical_loss_function, H[j], H_ref[j]; normalization=prod(N)*normalization)

                ### Custom VJP to compute the adjoint
                λ_∂f∂H, dH_H = VJP_λ_∂SIA∂H(simulation.parameters.UDE.grad.VJP_method, λ[j], H[j], θ, simulation, t₀, i)

                ### Update adjoint
                λ[j-1] .= λ[j] .+ Δt[j-1] .* (λ_∂f∂H .+ ∂ℓ∂H)

                ### Custom VJP for grad of loss function
                λ_∂f∂θ = VJP_λ_∂SIA∂θ(simulation.parameters.UDE.grad.VJP_method, λ[j-1], H[j], θ, dH_H, dH_λ[j], simulation, t₀, i)

                ### Update gradient
                # @assert ℓ ≈ loss_val "Loss in forward and reverse do not coincide: $(ℓ) != $(loss_val)"

                dLdθ .+= Δt[j-1] .* λ_∂f∂θ
            end

        elseif typeof(simulation.parameters.UDE.grad) <: ContinuousAdjoint

            # Adjoint setup
            # Construct continuous interpolator for solution of forward PDE
            # TODO: For now we do linear, but of course we can use something more sophisticated (although I don't think will make a huge difference for ice)
            # TODO: For an uniform grid, we don't need the Gridded, and actually this is more efficient based on docs
            # We construct H_itp with t_ref rather than t since t_ref can have small
            # numerical errors that make the interpolator to evaluate outside the interval
            # Notice this should not be an issue since t ≈ t_ref
            H_itp = interpolate((t_ref,), H, Gridded(Linear()))
            H_ref_itp = interpolate((t_ref,), H_ref, Gridded(Linear()))

            # Nodes and weights for numerical quadrature
            t_nodes, weights = GaussQuadrature(simulation.parameters.simulation.tspan..., simulation.parameters.UDE.grad.n_quadrature)

            ### Define the reverse ODE problem
            if (typeof(simulation.parameters.UDE.grad.VJP_method) <: DiscreteVJP) | (typeof(simulation.parameters.UDE.grad.VJP_method) <: EnzymeVJP) | (typeof(simulation.parameters.UDE.grad.VJP_method) <: ContinuousVJP)
                function f_adjoint_rev(dλ, λ, p, τ)
                    t = -τ
                    λ_∂f∂H, _ = VJP_λ_∂SIA∂H(simulation.parameters.UDE.grad.VJP_method, λ, H_itp(t), θ, simulation, t, i)
                    dλ .= λ_∂f∂H
                end
            else
                @error "VJP method $(simulation.parameters.UDE.grad.VJP_method) is not supported yet."
            end

            ### Definition of callback to introduce contrubution of loss function to adjoint
            t_ref_inv = .-reverse(t_ref)
            stop_condition(λ, t, integrator) = Sleipnir.stop_condition_tstops(λ, t, integrator, t_ref_inv)
            function effect!(integrator)
                t = - integrator.t
                ∂ℓ∂H = backward_loss(simulation.parameters.UDE.empirical_loss_function, H_itp(t), H_ref_itp(t); normalization=prod(N)*normalization)
                integrator.u .= integrator.u .+ simulation.parameters.simulation.step .* ∂ℓ∂H
            end
            cb_adjoint_loss = DiscreteCallback(stop_condition, effect!)

            # Final condition
            λ₁ = Enzyme.make_zero(H[end])
            # Include contribution of loss from last step since this is not accounted for in the discrete callback
            if simulation.parameters.simulation.tspan[2] ∈ t_ref
                t_final = simulation.parameters.simulation.tspan[2]
                λ₁ .+= simulation.parameters.simulation.step .* backward_loss(simulation.parameters.UDE.empirical_loss_function, H_itp(t_final), H_ref_itp(t_final); normalization=prod(N)*normalization)
                # λ₁ .-= only(backward_loss(simulation.parameters.UDE.empirical_loss_function, [H_itp(t_final)], [H_ref_itp(t_final)]; normalization=prod(N)*normalization))
            end
            # Define ODE Problem with time in reverse
            adjoint_PDE_rev = ODEProblem(f_adjoint_rev, λ₁, .-reverse(simulation.parameters.simulation.tspan))

            # Solve reverse adjoint PDE with dense output
            sol_rev = solve(adjoint_PDE_rev,
                            callback = cb_adjoint_loss,
                            # saveat=t_nodes_rev, # dont use this!
                            dense = true,
                            save_everystep = true,
                            tstops = t_ref_inv,
                            simulation.parameters.UDE.grad.solver,
                            dtmax = simulation.parameters.UDE.grad.dtmax,
                            reltol = simulation.parameters.UDE.grad.reltol,
                            abstol = simulation.parameters.UDE.grad.abstol)

            ### Numerical integration using quadrature to compute gradient
            if (typeof(simulation.parameters.UDE.grad.VJP_method) <: DiscreteVJP) | (typeof(simulation.parameters.UDE.grad.VJP_method) <: EnzymeVJP) | (typeof(simulation.parameters.UDE.grad.VJP_method) <: ContinuousVJP)
                for j in 1:length(t_nodes)
                    λ_sol = sol_rev(-t_nodes[j])
                    _H = H_itp(t_nodes[j])
                    λ_∂f∂θ = VJP_λ_∂SIA∂θ(simulation.parameters.UDE.grad.VJP_method, λ_sol, _H, θ, nothing, zero(λ_sol), simulation, t_nodes[j], i)
                    dLdθ .+= weights[j] .* λ_∂f∂θ
                end
            else
                @error "VJP method $(simulation.parameters.UDE.grad.VJP_method) is not supported yet."
            end

        elseif typeof(simulation.parameters.UDE.grad) <: DummyAdjoint
            if isnothing(simulation.parameters.UDE.grad.grad_function)
                dLdθ .+= maximum(abs.(θ)) .* rand(Float64, size(θ))
            else
                dLdθ .+ simulation.parameters.UDE.grad.grad_function(θ)
            end
        else
            @error "Adjoint method $(simulation.parameters.UDE.grad) is not supported yet."
        end

        # Return final evaluations of gradient
        push!(dLdθs_vector, dLdθ)
    end

    return loss_val, dLdθs_vector

end

# """
# Define SIA2D forward map for the adjoint mode

# This is currently just use for Enzyme
# """
# function SIA2D_adjoint!(_θ, _dH::Matrix{R}, _H::Matrix{R}, simulation::FunctionalInversion, smodel, t::R, batch_id::I) where {R <: Real, I <: Integer}
#     # make prediction with neural network
#     apply_UDE_parametrization_enzyme!(_θ, simulation, smodel, batch_id)

#     # dH is computed as follows
#     Huginn.SIA2D!(_dH, _H, simulation, t; batch_id=batch_id)

#     return nothing
# end


"""
Generate fake forward simulation
"""
function generate_glacier_prediction!(glaciers::Vector{G}, params::Sleipnir.Parameters, model::Sleipnir.Model, tstops::Vector{Float64}) where {G <: Sleipnir.AbstractGlacier}
    # Generate timespan from simulation

    t₀, t₁ = params.simulation.tspan
    # Δt = params.simulation.step

    @assert t₀ <= minimum(tstops)
    @assert t₁ >= maximum(tstops)
    # @assert Δt <= t₁-t₀

    # nSteps = length(tstops)
    # ts = t₀ .+ Δt .* collect(0:nSteps)

    # Update reference value of glacier 
    # glacier.A = A

    prediction = Huginn.Prediction(model, glaciers, params)

    # # Update model iceflow parameter 
    # prediction.model.iceflow.A = Ref(A)

    Huginn.run!(prediction)

    # println("Reference value of A used for synthetic data:")
    # @show prediction.model.iceflow.A
    # @show glacier.A

    # Lets create a very simple static glacier
    # Hs = [glacier.H₀ for _ in 1:nSteps]

    # Store the thickness data in the glacier
    store_thickness_data!(prediction, tstops)
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