export VJP_λ_∂SIA∂H_continuous
export VJP_λ_∂SIA∂θ_continuous
"""
Implementation of continuous adjoint to time-independent SIA2D equation.

Given λ and H, returns the VJP of λ^T * ∂(SIA2D)/∂H (H)
"""
function VJP_λ_∂SIA∂H_continuous(λ::Matrix{R}, H::Matrix{R}, simulation::SIM, t::R; batch_id::Union{Nothing, I} = nothing) where {R <: Real, I <: Integer, SIM <: Simulation}

    if isnothing(batch_id)
        SIA2D_model = simulation.model.iceflow
        glacier = simulation.glaciers[SIA2D_model.glacier_idx[]]
    else
        SIA2D_model = simulation.model.iceflow[batch_id] # We pick the right iceflow model for this glacier
        glacier = simulation.glaciers[batch_id]
    end

    params = simulation.parameters
    # Retrieve parameters
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    A = SIA2D_model.A
    n = SIA2D_model.n
    ρ = params.physical.ρ
    g = params.physical.g

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) ./ Δx
    dSdy = diff_y(S) ./ Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^(1/2)
    # This is used to compute the diffusivity:
    Γ = 2.0 * A[] * (ρ * g)^n[] / (n[]+2) # 1 / m^3 s
    D = Γ .* avg(H).^(n[] + 2) .* ∇S.^(n[] - 1)

    # Compute flux components
    @views dλdx_edges = diff_x(λ[:,2:end - 1]) ./ Δx
    @views dλdy_edges = diff_y(λ[2:end - 1,:]) ./ Δy

    Fx = .-avg_y(D) .* dλdx_edges
    Fy = .-avg_x(D) .* dλdy_edges

    Fxx = diff_x(Fx) / Δx
    Fyy = diff_y(Fy) / Δy

    # Flux divergence
    ∇D∇λ = .-(Fxx .+ Fyy)

    ### Contribution of ∂D/∂H
    ∂D∂H_dual = (n[] + 2) .* Γ .* avg(H).^(n[] + 1) .* ∇S.^(n[] - 1)
    ∂D∂H = avg(∂D∂H_dual)
    # Contibution of ∇⋅(|∇S|^(n-3) (∂xS + ∂yS))
    ∂D∂H .+= (n[] - 1) .* Γ .* inn(H).^(n[] + 2) .* avg(∇S).^(n[] - 3) .* (diff_x(dSdx)[:,2:end-1] ./ Δx .+ diff_y(dSdy)[2:end-1,:] ./ Δy)

    ∇λ∇S_x_edges = dSdx .* diff_x(λ) ./ Δx
    ∇λ∇S_y_edges = dSdy .* diff_y(λ) ./ Δy
    ∇λ∇S_x = avg_y(∇λ∇S_x_edges)
    ∇λ∇S_y = avg_x(∇λ∇S_y_edges)
    ∇λ∇S = ∇λ∇S_x .+ ∇λ∇S_y

    ∂D∂H_∇λ∇S = ∂D∂H .* avg(∇λ∇S)

    dλ = zero(λ)
    inn(dλ) .= ∇D∇λ .- ∂D∂H_∇λ∇S
    return dλ
end


function VJP_λ_∂SIA∂θ_continuous(θ, λ::Matrix{R}, H::Matrix{R}, simulation::SIM, t::R; batch_id::Union{Nothing, I} = nothing) where {R <: Real, I <: Integer, SIM <: Simulation}

    if isnothing(batch_id)
        SIA2D_model = simulation.model.iceflow
        glacier = simulation.glaciers[SIA2D_model.glacier_idx[]]
    else
        SIA2D_model = simulation.model.iceflow[batch_id] # We pick the right iceflow model for this glacier
        glacier = simulation.glaciers[batch_id]
    end

    params = simulation.parameters
    # Retrieve parameters
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    # A = SIA2D_model.A
    n = SIA2D_model.n
    ρ = params.physical.ρ
    g = params.physical.g

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) ./ Δx
    dSdy = diff_y(S) ./ Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^(1/2)
    Γ = 2.0 * (ρ * g)^n[] / (n[]+2)
    ∂D∂A = Γ .* avg(H).^(n[] + 2) .* ∇S.^(n[] - 1)

    ∇λ∇S_x_edges = dSdx .* diff_x(λ) ./ Δx
    ∇λ∇S_y_edges = dSdy .* diff_y(λ) ./ Δy
    ∇λ∇S_x = avg_y(∇λ∇S_x_edges)
    ∇λ∇S_y = avg_x(∇λ∇S_y_edges)
    ∇λ∇S = ∇λ∇S_x .+ ∇λ∇S_y

    ∇θ, = Zygote.gradient(_θ -> grad_apply_UDE_parametrization(_θ, simulation, 1), θ)
    # @infiltrate
    # if rand() < 0.03
    #     δA = 1e-20
    #     ϵ = δA / norm(∇θ)^2
    #     A_2 = grad_apply_UDE_parametrization(θ .+ ϵ .* ∇θ, simulation, 1)
    #     A_1 = grad_apply_UDE_parametrization(θ, simulation, 1)
    #     @show δA
    #     @show A_2 - A_1
    #     @assert isapprox(δA, A_2 - A_1, rtol=1e-2)
    # end

    # Integrate final result
    # if rand() < 0.05
    #     if ((Δx * Δy) * sum(∂D∂A .* ∇λ∇S)) > 0.0
    #         @info "A ------"
    #     else
    #         @info "A ++++++"
    #     end
    # end
    # @infiltrate
    return (Δx * Δy) * sum(∂D∂A .* ∇λ∇S) .* ∇θ
    # This fails!!! A keeps increasing!
    # return 1e19 .* ∇θ
end

# Repeated function
function grad_apply_UDE_parametrization(θ, simulation::SIM, batch_id::I) where {I <: Integer, SIM <: Simulation}
    # We load the ML model with the parameters
    model = simulation.model.machine_learning.architecture
    st = simulation.model.machine_learning.st
    smodel = StatefulLuxLayer{true}(model, θ.θ, st)

    # We generate the ML parametrization based on the target
    if simulation.parameters.UDE.target == "A"
        A = predict_A̅(smodel, [mean(simulation.glaciers[batch_id].climate.longterm_temps)])[1]
        # println("Value of A during internal gradient evaluation:")
        # @show A
        return A
    end
end