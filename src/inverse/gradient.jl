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

        println("Value of A used during gradient")
        @show apply_UDE_parametrization(θ, simulation, i)

        # This time does not really matter since SIA2D does not depend explicetely on time,
        # but we make it explicit here in case we want to customize this in the future.
        t₀ = 2010.0e0 
        # TODO: Try with the closure, should be the same
        # ∂f∂H_closure(_dH, _H) = SIA2D_adjoint(θ, _dH, _H, simulation, t₀, i)
        # ∂f∂H_closure(_dH, _H) = SIA2D_adjoint(Enzyme.Const(θ), _dH, _H, Enzyme.Const(simulation), Enzyme.Const(t₀), Enzyme.Const(i))

        # I wrote the for loop without really caring much about indices, so they may be one index off
        for j in reverse(2:k)

            # Zygote adjoint implementation
            # Create pullback function to evaluate VJPs
            # dH_H, ∂f∂H_pullback = Zygote._pullback(∂f∂H_closure, H[j])

            # Compute VJP with adjoint variable
            # Transpose operation
            # _, λ_∂f∂H = ∂f∂H_pullback(λ[j])
            # λ_∂f∂H, = ∂f∂H_pullback(λ[j])

            # Enzyme adjoint implementation
            dH_H = Enzyme.make_zero(H[j])
            λ_∂f∂H = Enzyme.make_zero(H[j])

            # TODO: Now initializing adjoint as ones, remove this next line
            λ[j] = ones(size(λ[j])...)

            Enzyme.autodiff(Reverse, SIA2D_adjoint, Const, Enzyme.Const(θ), Duplicated(dH_H, λ[j]), Duplicated(H[j], λ_∂f∂H), Enzyme.Const(simulation), Enzyme.Const(t₀), Enzyme.Const(i))
            @show maximum(abs.(λ_∂f∂H))

            dH_H = Enzyme.make_zero(H[j])
            λ_∂f∂H = Enzyme.make_zero(H[j])
            _H = zero(H[j])
            _H[1,:] .= 1.0
            _H[:,1] .= 1.0
            _H = abs.(randn(size(H[j])[1],size(H[j])[2]))
            Enzyme.autodiff(Reverse, SIA2D_adjoint, Const, Enzyme.Const(θ), Duplicated(dH_H, λ[j]), Duplicated(_H, λ_∂f∂H), Enzyme.Const(simulation), Enzyme.Const(t₀), Enzyme.Const(i))
            @show maximum(abs.(λ_∂f∂H))

            @infiltrate

            _H = zero(H[j])
            _H[1,:] .= 1.0
            _H[:,1] .= 1.0
            _H = abs.(randn(size(H[j])[1],size(H[j])[2]))

            λ  = ones(size(_H))
            λ[1,:] .= 1.0
            λ[:,1] .= 1.0
            t₀ = 2010.0

            dH_H = Enzyme.make_zero(λ)
            λ_∂f∂H = Enzyme.make_zero(_H)

            Enzyme.autodiff(Reverse, SIA2D_adjoint_test, Const, Duplicated(dH_H, λ), Duplicated(_H, λ_∂f∂H), Enzyme.Const(t₀))
            @show maximum(abs.(λ_∂f∂H))

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
            ∂f∂θ_closure(_dH, _θ) = SIA2D_adjoint(_θ, _dH, H[j], simulation, t₀, i)
            # Zygote adjoint implementation
            # TODO: change this to _pullback
            # dH_λ, ∂f∂θ_pullback = Zygote.pullback(∂f∂θ_closure, θ)
            # Compute loss with transpose of adjoint
            # λ_∂f∂θ, = ∂f∂θ_pullback(λ[j-1])

            # Enzyme implementation
            # TODO: Check on indices of λ
            # TODO: Change the function definition to remove closure and passing Const()
            dH_λ = Enzyme.make_zero(H[j])
            λ_∂f∂θ = Enzyme.make_zero(θ)
            Enzyme.autodiff(Reverse, ∂f∂θ_closure, Const, Duplicated(dH_λ, λ[j]), Duplicated(θ, λ_∂f∂θ))

            dLdθ .+= Δt[j-1] .* λ_∂f∂θ

            # Run simple test that both closures are computing the same primal
            @assert dH_H ≈ dH_λ "Result from forward pass needs to coincide for both closures when computing the pullback."
        end

        # Return final evaluations of gradient
        push!(dLdθs_vector, dLdθ)

    end

    @assert ℓ ≈ loss "Loss in forward and reverse do not coincide: $(ℓ) != $(loss)"
    return loss, dLdθs_vector

end

"""
Define SIA2D forward map for the adjoint mode
"""
function SIA2D_adjoint(_θ, _dH::Matrix{R}, _H::Matrix{R}, simulation::FunctionalInversion, t::R, batch_id::I) where {R <: Real, I <: Integer}
    # make prediction with neural network
    # A = apply_UDE_parametrization(_θ, simulation, batch_id)
    # simulation.model.iceflow[batch_id].A[] = A

    # dH is computed as follows
    _dH .= Huginn.SIA2D(_H, simulation, t; batch_id=batch_id)

    return nothing
end

function SIA2D_adjoint_test(_dH::Matrix{R}, _H::Matrix{R}, t::R) where {R <: Real}
    # make prediction with neural network
    # A = apply_UDE_parametrization(_θ, simulation, batch_id)
    # simulation.model.iceflow[batch_id].A[] = A

    # dH is computed as follows
    # _dH .= Huginn.SIA2D(_H, simulation, t)
    SIA2D!(_dH, _H, t)

    return nothing
end

@views diff_x(A) = (A[begin + 1:end, :] .- A[1:end - 1, :])
@views diff_y(A) = (A[:, begin + 1:end] .- A[:, 1:end - 1])
@views avg(A) = 0.25 .* ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )
@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )
@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )
@views inn(A) = A[2:end-1,2:end-1]
@views inn1(A) = A[1:end-1,1:end-1]

function SIA2D!(dH::Matrix{R}, H::Matrix{R}, t::R) where {R <:Real}
    
    S = abs.(randn(size(H)[1],size(H)[2]))
    Δx = 10.0
    Δy = 10.0
    n = 3
    ρ = 900.0
    g = 9.81
    A = 5e-18

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇Sx = avg_y(dSdx)
    ∇Sy = avg_x(dSdy)
    ∇S  = (∇Sx.^2 .+ ∇Sy.^2).^((n[] - 1)/2) 

    H̄ = avg(H)
    Γ = 2.0 * A * (ρ * g)^n / (n+2) # 1 / m^3 s 
    D = Γ .* H̄.^(n + 2) .* ∇S
    
    # Compute flux components
    @views dSdx_edges = diff_x(S[:,2:end - 1]) ./ Δx
    @views dSdy_edges = diff_y(S[2:end - 1,:]) ./ Δy

    # Cap surface elevaton differences with the upstream ice thickness to
    # impose boundary condition of the SIA equation
    # We need to do this with Tullio or something else that allow us to set indices.
    η₀ = 1.0
    dSdx_edges = @views @. min(dSdx_edges,  η₀ * H[2:end, 2:end-1]/Δx)
    dSdx_edges = @views @. max(dSdx_edges, -η₀ * H[1:end-1, 2:end-1]/Δx)
    dSdy_edges = @views @. min(dSdy_edges,  η₀ * H[2:end-1, 2:end]/Δy)
    dSdy_edges = @views @. max(dSdy_edges, -η₀ * H[2:end-1, 1:end-1]/Δy)

    Dx = avg_y(D)
    Dy = avg_x(D)
    Fx = .-Dx .* dSdx_edges
    Fy = .-Dy .* dSdy_edges 

    # #  Flux divergence
    Fxx = diff_x(Fx) / Δx
    Fyy = diff_y(Fy) / Δy

    # inn(dH) .= -(Fx[2:199,:] - Fx[1:198,:]) / Δx * 0.0001
    # inn(dH) .= -(diff_x(Fx)) / Δx * 0.0001
    inn(dH) .= .-(Fxx .+ Fyy) 
    @show maximum(dH)
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