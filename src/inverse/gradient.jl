export SIA2D_grad!
export generate_glacier_prediction!

"""
Inverse with batch
"""
function SIA2D_grad!(dθ, θ, glacier_data_loader::Tuple{Vector{I}, Vector{String}}, simulation::Union{FunctionalInversion, Nothing}=nothing) where {I <: Integer}

    # Run forward simulation in dense mode
    batch_ids = glacier_data_loader[1] # e.g., [2, 1]
    # Maybe this first forward run is not neccesary, since the adjoint will compute this same stuff.
    l = loss_iceflow(θ, batch_ids, simulation)

    # @assert length(glacier_ids) == 1 "Error when grabbing more than one glacier, id is lost somewhere"
    @assert !simulation.parameters.simulation.light "Forward solution needs to be stored in dense mode, no light, for gradient computation."

    # glacier_id = only(glacier_ids)
    dθ = zero(θ)

    # for glacier_id in glacier_ids
    # for batch_id in batch_ids
    #     # Find correct mapping between id and RGI in results inside train batch
    #     # glacier_results_id = Sleipnir.get_result_id_from_rgi(glacier_id, simulation)
    #     dθ .+= SIA2D_grad!(dθ, θ, batch_id, simulation)
    # end

    # @show length(simulation.model.iceflow)
    # @show simulation.model.iceflow[1].A
    # @show simulation.model.iceflow[2].A[]

    dθs = pmap(batch_id -> SIA2D_grad!(dθ, θ, batch_id, simulation), batch_ids)
    @show length(dθs)
    @show size(dθs)
    dθ = sum(dθs)
end

"""
Inverse by glacier
"""
function SIA2D_grad!(dθ, θ, batch_id::I, simulation::Union{FunctionalInversion, Nothing}=nothing) where {I <: Integer}

    # Extract relevant data
    glacier_results_id = Sleipnir.get_result_id_from_rgi(batch_id, simulation)
    result = simulation.results[glacier_results_id]

    # Results from forward simulation
    t = result.t
    Δt = diff(t)
    H = result.H

    # Reference data
    t_ref = only(simulation.glaciers[batch_id].data).t
    H_ref = only(simulation.glaciers[batch_id].data).H


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
        # Check if we can do this without the out-place function
        A = apply_UDE_parametrization(θ, simulation, batch_id)
        # Pass value of A
        # if rand() < 0.001
        #     @show A
        # end
        # @show A
        # @show batch_id
        # @show length(simulation.model.iceflow)
        simulation.model.iceflow[batch_id].A[] = A
        # dH is computed as follows
        Huginn.SIA2D(H, simulation, t; batch_id)
    end

    # This time does not really matter since SIA2D does not depend explicetely on time,
    # but we make it explicit here in case we want to customize this in the future.
    t₀ = 2010.0e0 
    ∂f∂H_closure(H) = SIA2D_adjoint(θ, H, t₀, batch_id)

    # I wrote the for loop without really caring much about indices, so they may be one index off
    for i in reverse(2:k)

        # Create pullback function to evaluate VJPs
        dH_H, ∂f∂H_pullback = Zygote.pullback(∂f∂H_closure, H[i])

        # Compute VJP with adjoint variable
        λ_∂f∂H, = ∂f∂H_pullback(λ[i])

        # Compute derivative of local contribution to loss function
        # TODO: Update this based on the actual value of the loss function as ∂(parameters.UDE.empirical_loss_function)/∂H
        β = 1.0
        ∂ℓ∂H = 2 .* (H[i] .- H_ref[i]) ./ (prod(N) * mean(H_ref[i])^β)

        ### Compute adjoint
        # Update time-dependent adjoint
        λ[i-1] .= λ[i] .+ Δt[i-1] .* (λ_∂f∂H .+ ∂ℓ∂H)

        ### Compute loss function
        ∂f∂θ_closure(θ) = SIA2D_adjoint(θ, H[i], t₀, batch_id)
        dH_λ, ∂f∂θ_pullback = Zygote.pullback(∂f∂θ_closure, θ)
        λ_∂f∂θ, = ∂f∂θ_pullback(λ[i-1])
        dLdθ .+= Δt[i-1] .* λ_∂f∂θ

        # Run simple test that both closures are computing the same primal
        @assert dH_H ≈ dH_λ "Result from forward pass needs to coincide for both closures when computing the pullback."
    end

    # Return final evaluation of gradient
    # dθ .= dLdθ
    return dLdθ
end


# function simulate_iceflow_adjoint(θ, simulation::SIM, batch_id::I) where {I <: Integer, SIM <: Simulation}

#     # I think the parametrization has been already applied


#     apply_UDE_parametrization!(θ, simulation, nothing, batch_id)

#     SIA2D_adjoint_closure(λ, H, t) = 1

# end

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

    nSteps = length(tstops)
    # ts = t₀ .+ Δt .* collect(0:nSteps)

    # Update reference value of glacier 
    glacier.A = A

    prediction = Huginn.Prediction(model, [glacier], params)
    Huginn.run!(prediction)

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