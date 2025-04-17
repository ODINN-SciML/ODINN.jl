export VJP_λ_∂SIA_discrete
export VJP_λ_∂SIA∂H_continuous, VJP_λ_∂SIA∂θ_continuous


#######################################
######     Discrete Adjoint      ######
#######################################

"""
    VJP_λ_∂SIA_discrete(
        ∂dH::Matrix{R},
        H::Matrix{R},
        simulation::SIM,
        t::R;
        batch_id::Union{Nothing, I} = nothing
    )

Compute an out-of-place adjoint step of the Shallow Ice Approximation PDE.
Given an output gradient, it backpropagates the gradient to the inputs H and A.
To some extent, this function is equivalent to VJP_λ_∂SIA∂H_continuous and
VJP_λ_∂SIA∂θ_continuous.

Arguments:
- `∂dH::Matrix{R}`: Output gradient to backpropagate.
- `H::Matrix{R}`: Ice thickness which corresponds to the input state of the SIA2D.
- `simulation::SIM`: Simulation parameters.
- `t::R`: Time value, not used as SIA2D is time independent.
- `batch_id::Union{Nothing, I}`: Batch index.

Returns:
- `∂H::Matrix{R}`: Input gradient wrt H.
- `∂A::F`: Input gradient wrt A.
"""
function VJP_λ_∂SIA_discrete(
# function SIA2D_discrete_adjoint(
    ∂dH::Matrix{R},
    H::Matrix{R},
    θ,
    simulation::SIM,
    t::R;
    batch_id::Union{Nothing, I} = nothing
) where {R <:Real, I <: Integer, SIM <: Simulation}

    if isnothing(batch_id)
        SIA2D_model = simulation.model.iceflow
        glacier = simulation.glaciers[SIA2D_model.glacier_idx[]]
    else
        SIA2D_model = simulation.model.iceflow[batch_id] # We pick the right iceflow model for this glacier
        glacier = simulation.glaciers[batch_id]
    end

    # Retrieve models and parameters
    params = simulation.parameters
    ml_model = simulation.model.machine_learning
    target = ml_model.target

    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy

    # First, enforce values to be positive
    map!(x -> ifelse(x > 0.0, x, 0.0), H, H)
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = Huginn.diff_x(S)/Δx
    dSdy = Huginn.diff_y(S)/Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)

    # Compute surface slope
    ∇S = (∇Sx.^2 .+ ∇Sy.^2).^(1/2)

    # Compute average ice thickness
    H̄ = Huginn.avg(H)

    # Compute diffusivity based on target objective
    D = target.D(
        H = H̄, ∇S = ∇S, θ = θ,
        ice_model = SIA2D_model, ml_model = ml_model,
        glacier = glacier, params = params
    )

    # Compute flux components
    @views dSdx_edges = Huginn.diff_x(S[:,2:end - 1]) / Δx
    @views dSdy_edges = Huginn.diff_y(S[2:end - 1,:]) / Δy

    # Cap surface elevaton differences with the upstream ice thickness to
    # imporse boundary condition of the SIA equation
    η₀ = params.physical.η₀
    dSdx_edges_clamp = clamp_borders_dx(dSdx_edges, H, η₀, Δx)
    dSdy_edges_clamp = clamp_borders_dy(dSdy_edges, H, η₀, Δy)

    Dx = Huginn.avg_y(D)
    Dy = Huginn.avg_x(D)

    ∂dH_inn = ∂dH[2:end-1,2:end-1]
    Fx_adjoint = diff_x_adjoint(-∂dH_inn, Δx)
    Fy_adjoint = diff_y_adjoint(-∂dH_inn, Δy)
    Dx_adjoint = avg_y_adjoint(-Fx_adjoint .* dSdx_edges_clamp)
    Dy_adjoint = avg_x_adjoint(-Fy_adjoint .* dSdy_edges_clamp)
    D_adjoint = Dx_adjoint + Dy_adjoint

    ### First term

    # Equals ∂D/∂H
    α = target.∂D∂H(
        H = H̄, ∇S = ∇S, θ = θ,
        ice_model = SIA2D_model, ml_model = ml_model,
        glacier = glacier, params = params
    )
    # Equals ∂D/∂(∇H)
    β = target.∂D∂∇H(
        H = H̄, ∇S = ∇S, θ = θ,
        ice_model = SIA2D_model, ml_model = ml_model,
        glacier = glacier, params = params
    )

    βx = β .* ∇Sx
    βy = β .* ∇Sy
    ∂D∂H_adj = avg_adjoint(α .* D_adjoint) + diff_x_adjoint(avg_y_adjoint(βx .* D_adjoint), Δx) + diff_y_adjoint(avg_x_adjoint(βy .* D_adjoint), Δy)

    ### Second term
    ∂Cx = - Fx_adjoint .* Dx
    ∂Cy = - Fy_adjoint .* Dy
    ∂dSx = zeros(Sleipnir.Float, size(dSdx_edges))
    ∂dSy = zeros(Sleipnir.Float, size(dSdy_edges))
    ∂Hlocx = zeros(Sleipnir.Float, size(H))
    ∂Hlocy = zeros(Sleipnir.Float, size(H))
    clamp_borders_dx_adjoint!(∂dSx, ∂Hlocx, ∂Cx, η₀, Δx, H, dSdx_edges)
    clamp_borders_dy_adjoint!(∂dSy, ∂Hlocy, ∂Cy, η₀, Δy, H, dSdy_edges)
    ∇adj∂dSx = zero(S); ∇adj∂dSx[:,2:end - 1] .= diff_x_adjoint(∂dSx, Δx)
    ∂C∂H_adj_x = ∇adj∂dSx + ∂Hlocx
    ∇adj∂dSy = zero(S); ∇adj∂dSy[2:end - 1,:] .= diff_y_adjoint(∂dSy, Δy)
    ∂C∂H_adj_y = ∇adj∂dSy + ∂Hlocy
    ∂C∂H_adj = ∂C∂H_adj_x + ∂C∂H_adj_y

    # Sum contributions of diffusivity and clipping
    ∂H = ∂D∂H_adj + ∂C∂H_adj
    ∂H .= ∂H.*(H.>0)

    # Gradient wrt θ
    ∂D∂θ = target.∂D∂θ(
        H = H̄, ∇S = ∇S, θ = θ,
        ice_model = SIA2D_model, ml_model = ml_model,
        glacier = glacier, params = params
    )
    # Evaluate numerical integral for loss
    @tullio ∂θ_v[k] := ∂D∂θ[i, j, k] * D_adjoint[i, j]
    # Construct component vector
    ∂θ = Vector2ComponentVector(∂θ_v, θ)

    return ∂H, ∂θ
end


#######################################
#####     Contiuous Adjoint      ######
#######################################

"""
    VJP_λ_∂SIA∂H_continuous(
        λ::Matrix{R},
        H::Matrix{R},
        simulation::SIM,
        t::R;
        batch_id::Union{Nothing, I} = nothing
    )
Implementation of the continuous adjoint of the SIA2D equation with respect to H.
Given λ and H, it returns the VJP of λ^T * ∂(SIA2D)/∂H (H).

Arguments:
- `λ::Matrix{R}`: Adjoint state, also called output gradient in reverse-mode AD.
- `H::Matrix{R}`: Ice thickness which corresponds to the input state of the SIA2D.
- `simulation::SIM`: Simulation parameters.
- `t::R`: Time value, not used as SIA2D is time independent.
- `batch_id::Union{Nothing, I}`: Batch index.

Returns:
- `dλ::Matrix{R}`: Jacobian vector product, also called input gradient in reverse-mode AD.
"""
function VJP_λ_∂SIA∂H_continuous(
    λ::Matrix{R},
    H::Matrix{R},
    θ,
    simulation::SIM,
    t::R;
    batch_id::Union{Nothing, I} = nothing
) where {R <: Real, I <: Integer, SIM <: Simulation}

    if isnothing(batch_id)
        SIA2D_model = simulation.model.iceflow
        glacier = simulation.glaciers[SIA2D_model.glacier_idx[]]
    else
        SIA2D_model = simulation.model.iceflow[batch_id] # We pick the right iceflow model for this glacier
        glacier = simulation.glaciers[batch_id]
    end

    # Retrieve models and parameters
    params = simulation.parameters
    ml_model = simulation.model.machine_learning
    target = ml_model.target

    # Retrieve parameters
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy

    # First, enforce values to be positive
    map!(x -> ifelse(x > 0.0, x, 0.0), H, H)
    # Update glacier surface altimetry
    S = B .+ H

    ### Computation of effective diffusivity
    dSdx = Huginn.diff_x(S) ./ Δx
    dSdy = Huginn.diff_y(S) ./ Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)

    # Compute surface slope
    ∇S = (∇Sx.^2 .+ ∇Sy.^2).^(1/2)

    # Compute average ice thickness
    H̄ = Huginn.avg(H)

    # Compute diffusivity based on target objective
    D = target.D(
        H = H̄, ∇S = ∇S, θ = θ,
        ice_model = SIA2D_model, ml_model = ml_model,
        glacier = glacier, params = params
    )

    # TODO: Clip dSdx and dSdx for conservation of mass condition

    ### Computation of partial derivatives of diffusivity
    ∂D∂H_dual = target.∂D∂H(
        H = H̄, ∇S = ∇S, θ = θ,
        ice_model = SIA2D_model, ml_model = ml_model,
        glacier = glacier, params = params
    )
    ∂D∂H = Huginn.avg(∂D∂H_dual)

    β = target.∂D∂∇H(
        H = H̄, ∇S = ∇S, θ = θ,
        ice_model = SIA2D_model, ml_model = ml_model,
        glacier = glacier, params = params
    )

    # Derivatives evaluated in dual grid
    ∂D∂∇H_x = β .* ∇Sx
    ∂D∂∇H_y = β .* ∇Sy

    ### Computation of term ∇⋅(D⋅∇λ)

    # Compute flux components
    @views dλdx_edges = Huginn.diff_x(λ[:,2:end - 1]) ./ Δx
    @views dλdy_edges = Huginn.diff_y(λ[2:end - 1,:]) ./ Δy

    Fx = .-Huginn.avg_y(D) .* dλdx_edges
    Fy = .-Huginn.avg_x(D) .* dλdy_edges

    Fxx = Huginn.diff_x(Fx) / Δx
    Fyy = Huginn.diff_y(Fy) / Δy

    ### Flux divergence
    ∇D∇λ = .-(Fxx .+ Fyy)

    ### Computation of term ∂D∂H x ⟨∇S, ∇λ⟩
    ∇λ∇S_x_edges = dSdx .* Huginn.diff_x(λ) ./ Δx
    ∇λ∇S_y_edges = dSdy .* Huginn.diff_y(λ) ./ Δy
    ∇λ∇S_x = Huginn.avg_y(∇λ∇S_x_edges)
    ∇λ∇S_y = Huginn.avg_x(∇λ∇S_y_edges)
    ∇λ∇S = ∇λ∇S_x .+ ∇λ∇S_y

    ∂D∂H_∇S_∇λ = ∂D∂H .* Huginn.avg(∇λ∇S)

    ### Computation of term ∇⋅(∂D∂∇H x ⟨∇S, ∇λ⟩)
    ∂D∂∇H_∇S_∇λ_x = ∇λ∇S .* ∂D∂∇H_x
    ∂D∂∇H_∇S_∇λ_y = ∇λ∇S .* ∂D∂∇H_y

    ∇_∂D∂∇H_∇S_∇λ_x = Huginn.diff_x(∂D∂∇H_∇S_∇λ_x) ./ Δx
    ∇_∂D∂∇H_∇S_∇λ_y = Huginn.diff_y(∂D∂∇H_∇S_∇λ_y) ./ Δy
    ∇_∂D∂∇H_∇S_∇λ = Huginn.avg_y(∇_∂D∂∇H_∇S_∇λ_x) .+ Huginn.avg_x(∇_∂D∂∇H_∇S_∇λ_y)

    ### Final computation of VJP
    dλ = zero(λ)
    Huginn.inn(dλ) .= .+ ∇D∇λ .- ∂D∂H_∇S_∇λ .+ ∇_∂D∂∇H_∇S_∇λ
    return dλ
end

"""
    VJP_λ_∂SIA∂θ_continuous(
        θ,
        λ::Matrix{R},
        H::Matrix{R},
        simulation::SIM,
        t::R;
        batch_id::Union{Nothing, I} = nothing
    )
Implementation of the continuous adjoint of the SIA2D equation with respect to θ.
Given λ, H and θ, it returns the VJP of λ^T * ∂(SIA2D)/∂θ (θ).

Arguments:
- `θ`: Vector of parameters
- `λ::Matrix{R}`: Adjoint state, also called output gradient in reverse-mode AD.
- `H::Matrix{R}`: Ice thickness which corresponds to the input state of the SIA2D.
- `simulation::SIM`: Simulation parameters.
- `t::R`: Time value, not used as SIA2D is time independent.
- `batch_id::Union{Nothing, I}`: Batch index.

Returns:
- `dλ::Matrix{R}`: Jacobian vector product, also called input gradient in reverse-mode AD.
"""
function VJP_λ_∂SIA∂θ_continuous(
    λ::Matrix{R},
    H::Matrix{R},
    θ,
    simulation::SIM,
    t::R;
    batch_id::Union{Nothing, I} = nothing
) where {R <: Real, I <: Integer, SIM <: Simulation}

    if isnothing(batch_id)
        SIA2D_model = simulation.model.iceflow
        glacier = simulation.glaciers[SIA2D_model.glacier_idx[]]
    else
        SIA2D_model = simulation.model.iceflow[batch_id] # We pick the right iceflow model for this glacier
        glacier = simulation.glaciers[batch_id]
    end

    # Retrieve models and parameters
    params = simulation.parameters
    ml_model = simulation.model.machine_learning
    target = ml_model.target

    # Retrieve parameters
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy

    # First, enforce values to be positive
    map!(x -> ifelse(x > 0.0, x, 0.0), H, H)
    # Update glacier surface altimetry
    S = B .+ H

    ### Computation of effective diffusivity
    dSdx = Huginn.diff_x(S) ./ Δx
    dSdy = Huginn.diff_y(S) ./ Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)

    @views dSdx_edges = Huginn.diff_x(S[:,2:end - 1]) ./ Δx
    @views dSdy_edges = Huginn.diff_y(S[2:end - 1,:]) ./ Δy

    # Compute surface slope
    ∇S = (∇Sx.^2 .+ ∇Sy.^2).^(1/2)

    # Compute average ice thickness
    H̄ = Huginn.avg(H)

    # Compute diffusivity based on target objective
    D = target.D(
        H = H̄, ∇S = ∇S, θ = θ,
        ice_model = SIA2D_model, ml_model = ml_model,
        glacier = glacier, params = params
    )

    # Gradient wrt θ
    ∂D∂θ = target.∂D∂θ(
        H = H̄, ∇S = ∇S, θ = θ,
        ice_model = SIA2D_model, ml_model = ml_model,
        glacier = glacier, params = params
    )

    # Compute the flux
    η₀ = params.physical.η₀
    dSdx_edges = clamp_borders_dx(dSdx_edges, H, η₀, Δx)
    dSdy_edges = clamp_borders_dy(dSdy_edges, H, η₀, Δy)

    # Compute flux components
    @tullio Fx[i, j, k] := (∂D∂θ[i, j, k] + ∂D∂θ[i, j + 1, k]) / 2 * dSdx_edges[i, j]
    @tullio Fy[i, j, k] := (∂D∂θ[i, j, k] + ∂D∂θ[i + 1, j, k]) / 2 * dSdy_edges[i, j]

    @tullio Fxx[i, j, k] := (Fx[i + 1, j, k] - Fx[i, j, k]) / Δx
    @tullio Fyy[i, j, k] := (Fy[i, j + 1, k] - Fy[i, j, k]) / Δy

    # Combine fluxes and pad contours
    @tullio ∇_∂D∂θ_∇S[i, j, k] := Fxx[pad(i-1,1,1), pad(j-1,1,1), k] + Fyy[pad(i-1,1,1), pad(j-1,1,1), k]

    # Evaluate numerical integral for loss
    @tullio ∂θ_v[k] := ∇_∂D∂θ_∇S[i, j, k] * λ[i, j]

    return ∂θ_v
end

# Repeated function
# function grad_apply_UDE_parametrization(θ, simulation::SIM, batch_id::I) where {I <: Integer, SIM <: Simulation}
#     # We load the ML model with the parameters
#     model = simulation.model.machine_learning.architecture
#     st = simulation.model.machine_learning.st
#     smodel = StatefulLuxLayer{true}(model, θ.θ, st)

#     # We generate the ML parametrization based on the target
#     if simulation.model.machine_learning.target.name == :A
#         min_NN = simulation.parameters.physical.minA
#         max_NN = simulation.parameters.physical.maxA
#         A = predict_A̅(smodel, [mean(simulation.glaciers[batch_id].climate.longterm_temps)], (min_NN, max_NN))[1]
#         # println("Value of A during internal gradient evaluation:")
#         # @show A
#         return A
#     end
# end

