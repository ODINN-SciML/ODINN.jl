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

    @assert !simulation.parameters.simulation.light "Forward solution needs to be stored in dense mode, no light, for gradient computation."

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
        @show losses
        @show norm.(dθs)
        @warn "Potential unstable gradient for glacier"
    end

    @assert typeof(θ) == typeof(sum(dθs))
    dθ = sum(dθs)
end

"""
Inverse by glacier
"""
function SIA2D_grad_batch!(θ, simulation::FunctionalInversion)

    # Run forward simulation to trigger Result
    loss = loss_iceflow_transient(θ, simulation)
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
        t_ref = only(simulation.glaciers[i].data).t
        H_ref = only(simulation.glaciers[i].data).H

        @assert t ≈ t_ref "Reference times of simulation and reference data do not coincide."
        @assert length(H) == length(H_ref)
        @assert size(H[begin]) == size(H_ref[begin])

        # Dimensions
        N = size(result.B)
        k = length(H)

        # Adjoint setup
        # Define empty object to store adjoint in reverse mode
        λ  = [zeros(N) for _ in 1:k]
        dLdθ = zero(θ)

        # TODO: We do simply forward Euler, but we can probably write ODE for dλ

        """
        Define SIA2D forward map for the adjoint mode
        """
        function SIA2D_adjoint(θ, H::Matrix{R}, t::R, batch_id::I) where {R <: Real, I <: Integer}
            # make prediction with neural network
            # A = apply_UDE_parametrization(θ, simulation, batch_id)
            A = apply_UDE_parametrization(θ, simulation, batch_id)
            simulation.model.iceflow[batch_id].A[] = A
            # println("A during grad calculation:")
            # @show A
            # dH is computed as follows
            Huginn.SIA2D(H, simulation, t; batch_id)
        end

        println("Value of A used during gradient")
        @show apply_UDE_parametrization(θ, simulation, i)

        # This time does not really matter since SIA2D does not depend explicetely on time,
        # but we make it explicit here in case we want to customize this in the future.
        t₀ = 2010.0e0 
        ∂f∂H_closure(H) = SIA2D_adjoint(θ, H, t₀, i)

        # I wrote the for loop without really caring much about indices, so they may be one index off
        for j in reverse(2:k)

            # Create pullback function to evaluate VJPs
            dH_H, ∂f∂H_pullback = Zygote.pullback(∂f∂H_closure, H[j])

            # Compute VJP with adjoint variable
            λ_∂f∂H, = ∂f∂H_pullback(λ[j])

            # Compute derivative of local contribution to loss function
            # TODO: Update this based on the actual value of the loss function as ∂(parameters.UDE.empirical_loss_function)/∂H
            β = 2.0
            # normalization = mean(H_ref[j][H_ref[j] .> 0.0])^β
            normalization = 1.0
            ∂ℓ∂H = 2 .* (H[j] .- H_ref[j]) ./ (prod(N) * normalization)
            ℓ += Δt[j-1] * simulation.parameters.UDE.empirical_loss_function(H[j], H_ref[j]) / normalization
            ### Compute adjoint
            # Update time-dependent adjoint
            λ[j-1] .= λ[j] .+ Δt[j-1] .* (λ_∂f∂H .+ ∂ℓ∂H)

            ### Compute loss function
            ∂f∂θ_closure(θ) = SIA2D_adjoint(θ, H[j], t₀, i)
            dH_λ, ∂f∂θ_pullback = Zygote.pullback(∂f∂θ_closure, θ)
            λ_∂f∂θ, = ∂f∂θ_pullback(λ[j-1])
            dLdθ .+= Δt[j-1] .* λ_∂f∂θ

            # Run simple test that both closures are computing the same primal
            @assert dH_H ≈ dH_λ "Result from forward pass needs to coincide for both closures when computing the pullback."
        end

        # Return final evaluation of gradient
        # dθ .= dLdθ
        push!(dLdθs_vector, dLdθ)
        # @show apply_UDE_parametrization(θ, simulation, i)
        # @show norm(dLdθ)
    end

    @assert ℓ ≈ loss "Loss in forward and reverse do not coincide: $(ℓ) != $(loss)"
    @show loss
    return loss, dLdθs_vector

end

"""
Copy of apply_UDE_parametrization! but without inplacement
"""
function apply_UDE_parametrization(θ, simulation::FunctionalInversion, batch_id::I) where {I <: Integer}
    # We load the ML model with the parameters
    model = simulation.model.machine_learning.architecture
    st = simulation.model.machine_learning.st
    smodel = StatefulLuxLayer{true}(model, θ.θ, st)

    # We generate the ML parametrization based on the target
    if simulation.parameters.UDE.target == "A"
        A = predict_A̅(smodel, [mean(simulation.glaciers[batch_id].climate.longterm_temps)])[1]
        return A
    end
end

"""
Use just to generate results, don't need to change this API.
"""
function apply_UDE_parametrization(θ, simulation::FunctionalInversion, T::F) where {F <: AbstractFloat}
    # We load the ML model with the parameters
    model = simulation.model.machine_learning.architecture
    st = simulation.model.machine_learning.st
    smodel = StatefulLuxLayer{true}(model, θ.θ, st)

    # We generate the ML parametrization based on the target
    if simulation.parameters.UDE.target == "A"
        A = predict_A̅(smodel, [T])[1]
        return A
    end
end


"""
Generate fake forward simulation
"""
# function generate_glacier_prediction!(glacier::AbstractGlacier, params::Parameters; A::Float64, tstops::Vector{Float64})
function generate_glacier_prediction!(glacier::AbstractGlacier, params::Sleipnir.Parameters, model::Sleipnir.Model; A::Float64, tstops::Vector{Float64})
    # Generate timespan from simulation

    t₀, t₁ = params.simulation.tspan
    # Δt = params.simulation.step

    @assert t₀ <= minimum(tstops)
    @assert t₁ >= maximum(tstops)
    # @assert Δt <= t₁-t₀

    # nSteps = length(tstops)
    # ts = t₀ .+ Δt .* collect(0:nSteps)

    # Update reference value of glacier 
    glacier.A = A

    prediction = Huginn.Prediction(model, [glacier], params)

    # Update model iceflow parameter topo
    # prediction.model.iceflow.A[] = A

    Huginn.run!(prediction)

    @info "Reference value of A used for synthetic data:"
    @show prediction.model.iceflow.A
    @show glacier.A

    ts = only(prediction.results).t
    Hs = only(prediction.results).H

    @assert ts ≈ tstops "Timestops of simulated PDE solution and UDE solution do not match."

    # Lets create a very simple static glacier
    # Hs = [glacier.H₀ for _ in 1:nSteps]

    if isnothing(glacier.data)
        glacier.data = [Sleipnir.ThicknessData(t=ts, H=Hs)]
    else
        push!(glacier.data, Sleipnir.ThicknessData(t=ts, H=Hs))
    end
end