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
    # @show losses
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
    @assert norm(sum(dθs)) > 0.0

    dθ .= sum(dθs)
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
        Δx = simulation.glaciers[i].Δx
        Δy = simulation.glaciers[i].Δy

        @assert t ≈ t_ref "Reference times of simulation and reference data do not coincide."
        @assert length(H) == length(H_ref)
        @assert size(H[begin]) == size(H_ref[begin])

        # Dimensions
        N = size(result.B)
        k = length(H)

        # Adjoint setup
        # Define empty object to store adjoint in reverse mode
        λ  = [Enzyme.make_zero(result.B) for _ in 1:k]
        dH_λ = [Enzyme.make_zero(H[1]) for _ in 1:k]
        dLdθ = Enzyme.make_zero(θ)

        # TODO: We do simply forward Euler, but we can probably write ODE for dλ

        # This time does not really matter since SIA2D does not depend explicetely on time,
        # but we make it explicit here in case we want to customize this in the future.
        t₀ = 2010.0e0 
        # TODO: Try with the closure, should be the same
        # ∂f∂H_closure(_dH, _H) = SIA2D_adjoint(θ, _dH, _H, simulation, t₀, i)
        # ∂f∂H_closure(_dH, _H) = SIA2D_adjoint(Enzyme.Const(θ), _dH, _H, Enzyme.Const(simulation), Enzyme.Const(t₀), Enzyme.Const(i))

        for j in reverse(2:k)

            # β = 2.0
            # normalization = std(H_ref[j][H_ref[j] .> 0.0])^β
            normalization = 1.0
            # Compute derivative of local contribution to loss function
            # TODO: Update this based on the actual value of the loss function as ∂(parameters.UDE.empirical_loss_function)/∂H
            ∂ℓ∂H = 2 .* (H[j] .- H_ref[j]) ./ (prod(N) * Δx * Δy * normalization)
            ℓ += Δt[j-1] * simulation.parameters.UDE.empirical_loss_function(H[j], H_ref[j]) / normalization

            if typeof(simulation.parameters.UDE.grad) <: ODINNZygoteAdjoint

                # Zygote adjoint implementation
                # Create pullback function to evaluate VJPs
                ∂f∂H_closure(_dH, _H) = SIA2D_adjoint(θ, _dH, _H, simulation, t₀, i)
                dH_H, ∂f∂H_pullback = Zygote.pullback(∂f∂H_closure, H[j])

                # Compute VJP with adjoint variable
                # Transpose operation
                # _, λ_∂f∂H = ∂f∂H_pullback(λ[j])
                λ_∂f∂H, = ∂f∂H_pullback(λ[j])

                ### Update adjoint
                λ[j-1] .= λ[j] .+ Δt[j-1] .* (λ_∂f∂H .- ∂ℓ∂H)

                ### Compute loss function
                ∂f∂θ_closure(_dH, _θ) = SIA2D_adjoint(_θ, _dH, H[j], simulation, t₀, i)
                dH_λ, ∂f∂θ_pullback = Zygote.pullback(∂f∂θ_closure, θ)
                # Compute loss with transpose of adjoint
                λ_∂f∂θ, = ∂f∂θ_pullback(λ[j-1])

                dLdθ .+= Δt[j-1] .* λ_∂f∂θ
                # Run simple test that both closures are computing the same primal
                # @assert dH_H ≈ dH_λ "Result from forward pass needs to coincide for both closures when computing the pullback."

            elseif typeof(simulation.parameters.UDE.grad) <: ODINNEnzymeAdjoint


                # Enzyme adjoint implementation
                dH_H = Enzyme.make_zero(H[j])
                λ_∂f∂H = Enzyme.make_zero(H[j])
                _simulation = Enzyme.make_zero(simulation)
                smodel = StatefulLuxLayer{true}(simulation.model.machine_learning.architecture, θ.θ, simulation.model.machine_learning.st)

                # TODO: Now initializing adjoint as ones, remove this next line
                # λ[j] = ones(size(λ[j])...)
                # λ[j] = randn(size(λ[j])...)

                Enzyme.autodiff(Reverse, SIA2D_adjoint, Const,
                                Enzyme.Const(θ),
                                Duplicated(dH_H, λ[j]),
                                Duplicated(H[j], λ_∂f∂H),
                                Enzyme.Duplicated(simulation, _simulation),
                                Enzyme.Const(smodel),
                                Enzyme.Const(t₀),
                                Enzyme.Const(i))
                # Enzyme.autodiff(Reverse, SIA2D_adjoint_enzyme, Const, Enzyme.Const(θ), Duplicated(dH_H, λ[j]), Duplicated(H[j], λ_∂f∂H), Enzyme.Duplicated(simulation, _simulation), Enzyme.Duplicated(simulation.model.iceflow, _iceflow_models), Enzyme.Duplicated(simulation.glaciers, _glaciers), Enzyme.Const(t₀), Enzyme.Const(i))
                @show maximum(abs.(λ_∂f∂H))



                ### Compute adjoint
                # Update time-dependent adjoint
                λ[j-1] .= λ[j] .+ Δt[j-1] .* (λ_∂f∂H .- ∂ℓ∂H)

                ### Compute loss function
                # ∂f∂θ_closure(_dH, _θ) = SIA2D_adjoint_enzyme(_θ, _dH, H[j], simulation, simulation.model.iceflow, simulation.gloaciers, t₀, i)
                # ∂f∂θ_closure(_dH, _θ) = SIA2D_adjoint(_θ, _dH, H[j], simulation, t₀, i)

                # Enzyme implementation
                # TODO: Check on indices of λ
                # TODO: Change the function definition to remove closure and passing Const()
                λ_∂f∂θ = Enzyme.make_zero(θ)
                _simulation = Enzyme.make_zero(simulation)
                # The target variable needs to be set to one in order to be differentiated in the buffer variable if it's zero
                # if _simulation.model.iceflow[i].A[] == 0.0
                _simulation.model.iceflow[i].A[] = 1.0
                # end
                _smodel = Enzyme.make_zero(smodel)
                _H = Enzyme.make_zero(H[j])

                @infiltrate

                # temp = [5.0]
                # _temp = Enzyme.make_zero(temp)
                # Enzyme.autodiff(Reverse, predict_A̅, Active,
                #                 Duplicated(smodel, _smodel), 
                #                 Duplicated(temp, _temp))

                # _smodel = Enzyme.make_zero(smodel)
                # _iceflow = Enzyme.make_zero(simulation.model.iceflow[1])
                # _iceflow.A[] = 1.0
                # Enzyme.autodiff(Reverse, NN_enzyme!, Const,
                #                 Duplicated(smodel, _smodel),
                #                 Duplicated(simulation.model.iceflow[1], _iceflow))

                Enzyme.autodiff(Reverse, apply_UDE_parametrization_enzyme!, Const, 
                                Duplicated(θ, λ_∂f∂θ), 
                                Duplicated(simulation, _simulation), 
                                Duplicated(smodel, _smodel), 
                                Const(i))

                Enzyme.autodiff(Reverse, apply_UDE_parametrization_enzyme, Active, 
                                Duplicated(θ, λ_∂f∂θ), 
                                Duplicated(simulation, _simulation), 
                                Duplicated(smodel, _smodel), 
                                Const(i))

                # @show maximum(abs.(λ_∂f∂θ))

                # Enzyme.autodiff(Reverse, SIA2D_adjoint_enzyme, Const, Duplicated(dH_λ, λ[j]), Duplicated(θ, λ_∂f∂θ), Duplicated(H[j], λ_∂f∂H), Enzyme.Duplicated(simulation, _simulation), Enzyme.Duplicated(simulation.model.iceflow, _iceflow_models), Enzyme.Duplicated(simulation.glaciers, _glaciers), Enzyme.Const(t₀), Enzyme.Const(i))
                Enzyme.autodiff(Reverse, SIA2D_adjoint, Const, 
                                Duplicated(θ, λ_∂f∂θ), 
                                Duplicated(dH_λ[j], λ[j]), 
                                Duplicated(H[j], _H), 
                                Duplicated(simulation, _simulation), 
                                Duplicated(smodel, _smodel), 
                                Const(t₀), 
                                Const(i))
                @show maximum(abs.(λ_∂f∂θ))

                # Run simple test that both closures are computing the same primal
                # @assert dH_H ≈ dH_λ[j] "Result from forward pass needs to coincide for both closures when computing the pullback."

            elseif typeof(simulation.parameters.UDE.grad) <: ODINNContinuousAdjoint

                # Custom adjoint
                λ_∂f∂H = VJP_λ_∂SIA∂H_continuous(λ[j], H[j], simulation, t₀; batch_id = i)
                # @show maximum(abs.(λ_∂f∂H))

                ### Update adjoint
                λ[j-1] .= λ[j] .+ Δt[j-1] .* (λ_∂f∂H .- ∂ℓ∂H)

                λ_∂f∂θ = VJP_λ_∂SIA∂θ_continuous(θ, λ[j-1], H[j], simulation, t₀; batch_id = i)

                # TODO: Sign of the gradient is correct, but magnitude is still a bit OrdinaryDiffEq
                # TODO: Check on the adjoint calculation again to see what is missing or wrong
                # TODO: This measn that the continuous manual adjoint works for few glaciers
                dLdθ .+= Δt[j-1] .* λ_∂f∂θ

            else
                @error "No AD method specificed."
            end

            # Run simple test that both closures are computing the same primal
            @assert dH_H ≈ dH_λ[j] "Result from forward pass needs to coincide for both closures when computing the pullback."
        end

        ### Finite diff check of gradient
        # @infiltrate
        # ϵ = 1e-5
        # loss_update = loss_iceflow_transient(θ .+ ϵ .* dLdθ, simulation)
        # δl = loss_update - loss
        # println("This ratio should be ≈ 1")
        # @show (δl / norm(dLdθ)^2) / ϵ

        # Return final evaluations of gradient
        # @show maximum(abs.(dLdθ))
        push!(dLdθs_vector, dLdθ)
    end

    @assert ℓ ≈ loss "Loss in forward and reverse do not coincide: $(ℓ) != $(loss)"
    return loss, dLdθs_vector

end

"""
Define SIA2D forward map for the adjoint mode
"""
function SIA2D_adjoint(_θ, _dH::Matrix{R}, _H::Matrix{R}, simulation::FunctionalInversion, smodel, t::R, batch_id::I) where {R <: Real, I <: Integer}
    # make prediction with neural network
    apply_UDE_parametrization_enzyme!(_θ, simulation, smodel, batch_id)

    # dH is computed as follows
    Huginn.SIA2D!(_dH, _H, simulation, t; batch_id=batch_id)

    return nothing
end


# """
# Copy of apply_UDE_parametrization! but without inplacement
# """
# function apply_UDE_parametrization(θ, simulation::FunctionalInversion, batch_id::I) where {I <: Integer}
#     # We load the ML model with the parameters
#     model = simulation.model.machine_learning.architecture
#     st = simulation.model.machine_learning.st
#     smodel = StatefulLuxLayer{true}(model, θ.θ, st)

#     # We generate the ML parametrization based on the target
#     if simulation.parameters.UDE.target == "A"
#         A = predict_A̅(smodel, [mean(simulation.glaciers[batch_id].climate.longterm_temps)])[1]
#         return A
#     end
# end

# """
# Use just to generate results, don't need to change this API.
# """
# function apply_UDE_parametrization(θ, simulation::FunctionalInversion, T::F) where {F <: AbstractFloat}
#     # We load the ML model with the parameters
#     model = simulation.model.machine_learning.architecture
#     st = simulation.model.machine_learning.st
#     smodel = StatefulLuxLayer{true}(model, θ.θ, st)

#     # We generate the ML parametrization based on the target
#     if simulation.parameters.UDE.target == "A"
#         A = predict_A̅(smodel, [T])[1]
#         return A
#     end
# end


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