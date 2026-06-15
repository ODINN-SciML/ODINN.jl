export VJP_λ_∂SIA∂H_discrete, VJP_λ_∂SIA∂θ_discrete
export VJP_λ_∂SIA∂H_continuous, VJP_λ_∂SIA∂θ_continuous

#######################################
######     Discrete Adjoint      ######
#######################################

"""
    VJP_λ_∂SIA∂H_discrete(
        λ::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R,
    ) where {R <:Real, SIM <: Simulation}

Implementation of the discrete VJP of the SIA2D equation with respect to H.
Given λ and H, it returns the VJP of λ^T * ∂(SIA2D)/∂H (H).

Arguments:

  - `λ::Matrix{R}`: Adjoint state, also called output gradient in reverse-mode AD.
  - `H::Matrix{R}`: Ice thickness which corresponds to the input state of the SIA2D.
  - `simulation::SIM`: Simulation parameters.
  - `t::R`: Time value, not used as SIA2D is time independent.

Returns:

  - `dλ::Matrix{R}`: Jacobian vector product, also called input gradient in reverse-mode AD.
"""
function VJP_λ_∂SIA∂H_discrete(
        λ::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R
) where {R <: Real, SIM <: Simulation}
    SIA2D_model = simulation.model.iceflow
    SIA2D_cache = simulation.cache.iceflow
    glacier_idx = SIA2D_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]

    # Retrieve models and parameters
    params = simulation.parameters
    target = simulation.model.trainable_components.target

    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy

    # First, enforce values to be positive
    H = map(x -> ifelse(x > 0.0, x, 0.0), H)
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = Huginn.diff_x(S)/Δx
    dSdy = Huginn.diff_y(S)/Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)

    # Compute surface slope
    ∇S = (∇Sx .^ 2 .+ ∇Sy .^ 2) .^ (1/2)

    # Compute average ice thickness
    H̄ = Huginn.avg(H)

    # Store temporary variables for use with the laws
    SIA2D_cache.∇S .= ∇S
    SIA2D_cache.H̄ .= H̄

    θ = isnothing(simulation.model.trainable_components) ? nothing :
        simulation.model.trainable_components.θ
    Huginn.apply_all_non_callback_laws!(
        SIA2D_model, SIA2D_cache, simulation, glacier_idx, t, θ)

    # Compute diffusivity based on target objective
    D = Diffusivity(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )

    # Compute flux components
    @views dSdx_edges = Huginn.diff_x(S[:, 2:(end - 1)]) / Δx
    @views dSdy_edges = Huginn.diff_y(S[2:(end - 1), :]) / Δy

    # Cap surface elevaton differences with the upstream ice thickness to
    # imporse boundary condition of the SIA equation
    η₀ = params.physical.η₀
    dSdx_edges_clamp = clamp_borders_dx(dSdx_edges, H, η₀, Δx)
    dSdy_edges_clamp = clamp_borders_dy(dSdy_edges, H, η₀, Δy)

    Dx = Huginn.avg_y(D)
    Dy = Huginn.avg_x(D)

    λ_inn = λ[2:(end - 1), 2:(end - 1)]
    Fx_adjoint = diff_x_adjoint(-λ_inn, Δx)
    Fy_adjoint = diff_y_adjoint(-λ_inn, Δy)
    Dx_adjoint = avg_y_adjoint(-Fx_adjoint .* dSdx_edges_clamp)
    Dy_adjoint = avg_x_adjoint(-Fy_adjoint .* dSdy_edges_clamp)
    D_adjoint = Dx_adjoint + Dy_adjoint

    ### First term

    # Equals ∂D/∂H
    α = ∂Diffusivity∂H(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )
    # Equals ∂D/∂(∇H)
    β = ∂Diffusivity∂∇H(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )

    βx = β .* ∇Sx
    βy = β .* ∇Sy
    ∂D∂H_adj = avg_adjoint(α .* D_adjoint) +
               diff_x_adjoint(avg_y_adjoint(βx .* D_adjoint), Δx) +
               diff_y_adjoint(avg_x_adjoint(βy .* D_adjoint), Δy)

    ### Second term
    ∂Cx = - Fx_adjoint .* Dx
    ∂Cy = - Fy_adjoint .* Dy
    ∂dSx = zeros(Sleipnir.Float, size(dSdx_edges))
    ∂dSy = zeros(Sleipnir.Float, size(dSdy_edges))
    ∂Hlocx = zeros(Sleipnir.Float, size(H))
    ∂Hlocy = zeros(Sleipnir.Float, size(H))
    clamp_borders_dx_adjoint!(∂dSx, ∂Hlocx, ∂Cx, η₀, Δx, H, dSdx_edges)
    clamp_borders_dy_adjoint!(∂dSy, ∂Hlocy, ∂Cy, η₀, Δy, H, dSdy_edges)
    ∇adj∂dSx = zero(S);
    ∇adj∂dSx[:, 2:(end - 1)] .= diff_x_adjoint(∂dSx, Δx)
    ∂C∂H_adj_x = ∇adj∂dSx + ∂Hlocx
    ∇adj∂dSy = zero(S);
    ∇adj∂dSy[2:(end - 1), :] .= diff_y_adjoint(∂dSy, Δy)
    ∂C∂H_adj_y = ∇adj∂dSy + ∂Hlocy
    ∂C∂H_adj = ∂C∂H_adj_x + ∂C∂H_adj_y

    # Sum contributions of diffusivity and clipping
    dλ = ∂D∂H_adj + ∂C∂H_adj
    dλ .= dλ .* (H .> 0)

    return dλ
end

"""
    VJP_λ_∂SIA∂θ_discrete(
        λ::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R,
    ) where {R <: Real, SIM <: Simulation}

Implementation of the discrete VJP of the SIA2D equation with respect to θ.
Given λ, H and θ, it returns the VJP of λ^T * ∂(SIA2D)/∂θ (θ).

Arguments:

  - `θ`: Vector of parameters
  - `λ::Matrix{R}`: Adjoint state, also called output gradient in reverse-mode AD.
  - `H::Matrix{R}`: Ice thickness which corresponds to the input state of the SIA2D.
  - `simulation::SIM`: Simulation parameters.
  - `t::R`: Time value, not used as SIA2D is time independent.

Returns:

  - `∂θ`: Jacobian vector product with respect to θ, also called input gradient in
    reverse-mode AD. It has the same type as θ.
"""
function VJP_λ_∂SIA∂θ_discrete(
        λ::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R
) where {R <: Real, SIM <: Simulation}
    SIA2D_model = simulation.model.iceflow
    SIA2D_cache = simulation.cache.iceflow
    glacier_idx = SIA2D_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]

    # Retrieve models and parameters
    params = simulation.parameters
    target = simulation.model.trainable_components.target

    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy

    # First, enforce values to be positive
    H = map(x -> ifelse(x > 0.0, x, 0.0), H)
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = Huginn.diff_x(S)/Δx
    dSdy = Huginn.diff_y(S)/Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)

    # Compute surface slope
    ∇S = (∇Sx .^ 2 .+ ∇Sy .^ 2) .^ (1/2)

    # Compute average ice thickness
    H̄ = Huginn.avg(H)

    # Store temporary variables for use with the laws
    SIA2D_cache.∇S .= ∇S
    SIA2D_cache.H̄ .= H̄

    θ = isnothing(simulation.model.trainable_components) ? nothing :
        simulation.model.trainable_components.θ
    Huginn.apply_all_non_callback_laws!(
        SIA2D_model, SIA2D_cache, simulation, glacier_idx, t, θ)

    # Compute flux components
    @views dSdx_edges = Huginn.diff_x(S[:, 2:(end - 1)]) / Δx
    @views dSdy_edges = Huginn.diff_y(S[2:(end - 1), :]) / Δy

    # Cap surface elevaton differences with the upstream ice thickness to
    # imporse boundary condition of the SIA equation
    η₀ = params.physical.η₀
    dSdx_edges_clamp = clamp_borders_dx(dSdx_edges, H, η₀, Δx)
    dSdy_edges_clamp = clamp_borders_dy(dSdy_edges, H, η₀, Δy)

    λ_inn = λ[2:(end - 1), 2:(end - 1)]
    Fx_adjoint = diff_x_adjoint(-λ_inn, Δx)
    Fy_adjoint = diff_y_adjoint(-λ_inn, Δy)
    Dx_adjoint = avg_y_adjoint(-Fx_adjoint .* dSdx_edges_clamp)
    Dy_adjoint = avg_x_adjoint(-Fy_adjoint .* dSdy_edges_clamp)
    D_adjoint = Dx_adjoint + Dy_adjoint

    # Gradient wrt θ
    ∂D∂θ = ∂Diffusivity∂θ(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )
    # Evaluate numerical integral for loss
    @tullio ∂θ_v[k] := ∂D∂θ[i, j, k] * D_adjoint[i, j]
    # Construct component vector
    ∂θ = Vector2ComponentVector(∂θ_v, θ)

    return ∂θ
end

function VJP_λ_∂V∂Vxy(
        ∂V::Matrix{R},
        Vx::Matrix{R},
        Vy::Matrix{R}
) where {R <: Real}
    V = (Vx .^ 2 .+ Vy .^ 2) .^ (1/2)
    ∂Vx = Vx .* ∂V ./ V
    ∂Vy = Vy .* ∂V ./ V
    return ∂Vx, ∂Vy
end

function VJP_λ_∂surface_V∂H_discrete(
        ∂Vx::Matrix{R},
        ∂Vy::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R
) where {R <: Real, SIM <: Simulation}
    SIA2D_model = simulation.model.iceflow
    SIA2D_cache = simulation.cache.iceflow
    glacier_idx = SIA2D_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]

    # Retrieve models and parameters
    params = simulation.parameters
    target = simulation.model.trainable_components.target

    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy

    # First, enforce values to be positive
    H = map(x -> ifelse(x > 0.0, x, 0.0), H)
    # Update glacier surface altimetry
    S = B .+ H

    dSdx = Huginn.diff_x(S) / Δx
    dSdy = Huginn.diff_y(S) / Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)

    # Compute surface slope
    ∇S = (∇Sx .^ 2 .+ ∇Sy .^ 2) .^ (1/2)

    # Compute average ice thickness
    H̄ = Huginn.avg(H)

    # Store temporary variables for use with the laws
    SIA2D_cache.∇S .= ∇S
    SIA2D_cache.H̄ .= H̄

    θ = isnothing(simulation.model.trainable_components) ? nothing :
        simulation.model.trainable_components.θ
    Huginn.apply_all_non_callback_laws!(
        SIA2D_model, SIA2D_cache, simulation, glacier_idx, t, θ)

    # Equals ∂Dꜛ/∂H
    α = ∂Velocityꜛ∂H(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )
    # Equals ∂Dꜛ/∂(∇H)
    β = ∂Velocityꜛ∂∇H(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )

    inn1∂Vx = inn1(∂Vx)
    inn1∂Vy = inn1(∂Vy)

    ∇S∂V = (∇Sx .* inn1∂Vx .+ ∇Sy .* inn1∂Vy)

    βx = β .* ∇Sx
    βy = β .* ∇Sy
    ∂D∂H = avg_adjoint(α .* ∇S∂V) .+ diff_x_adjoint(avg_y_adjoint(βx .* ∇S∂V), Δx) +
           diff_y_adjoint(avg_x_adjoint(βy .* ∇S∂V), Δy)

    Dꜛ = Velocityꜛ(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )
    ∂∇S∂H = diff_x_adjoint(avg_y_adjoint(Dꜛ .* inn1∂Vx), Δx) .+
            diff_y_adjoint(avg_x_adjoint(Dꜛ .* inn1∂Vy), Δy)
    ∂H = ∂D∂H .+ ∂∇S∂H

    return -∂H
end

function VJP_λ_∂surface_V∂θ_discrete(
        ∂Vx::Matrix{R},
        ∂Vy::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R
) where {R <: Real, SIM <: Simulation}
    SIA2D_model = simulation.model.iceflow
    SIA2D_cache = simulation.cache.iceflow
    glacier_idx = SIA2D_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]

    # Retrieve models and parameters
    params = simulation.parameters
    target = simulation.model.trainable_components.target

    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy

    # First, enforce values to be positive
    H = map(x -> ifelse(x > 0.0, x, 0.0), H)
    # Update glacier surface altimetry
    S = B .+ H

    dSdx = Huginn.diff_x(S) / Δx
    dSdy = Huginn.diff_y(S) / Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)

    # Compute surface slope
    ∇S = (∇Sx .^ 2 .+ ∇Sy .^ 2) .^ (1/2)

    # Compute average ice thickness
    H̄ = Huginn.avg(H)

    # Store temporary variables for use with the laws
    SIA2D_cache.∇S .= ∇S
    SIA2D_cache.H̄ .= H̄

    θ = isnothing(simulation.model.trainable_components) ? nothing :
        simulation.model.trainable_components.θ
    Huginn.apply_all_non_callback_laws!(
        SIA2D_model, SIA2D_cache, simulation, glacier_idx, t, θ)

    ∇S∂V = (∇Sx .* inn1(∂Vx) .+ ∇Sy .* inn1(∂Vy))

    # Gradient wrt θ
    ∂D∂θ = ∂Velocityꜛ∂θ(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )
    # Evaluate numerical integral for loss
    @tullio ∂θ_v[k] := ∂D∂θ[i, j, k] * ∇S∂V[i, j]
    # Construct component vector
    ∂θ = Vector2ComponentVector(∂θ_v, θ)

    return -∂θ
end

#######################################
#####     Contiuous Adjoint      ######
#######################################

"""
    VJP_λ_∂SIA∂H_continuous(
        λ::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R,
    ) where {R <: Real, SIM <: Simulation}

Implementation of the continuous VJP of the SIA2D equation with respect to H.
Given λ and H, it returns the VJP of λ^T * ∂(SIA2D)/∂H (H).

Arguments:

  - `λ::Matrix{R}`: Adjoint state, also called output gradient in reverse-mode AD.
  - `H::Matrix{R}`: Ice thickness which corresponds to the input state of the SIA2D.
  - `simulation::SIM`: Simulation parameters.
  - `t::R`: Time value, not used as SIA2D is time independent.

Returns:

  - `dλ::Matrix{R}`: Jacobian vector product, also called input gradient in reverse-mode AD.
"""
function VJP_λ_∂SIA∂H_continuous(
        λ::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R
) where {R <: Real, SIM <: Simulation}
    SIA2D_model = simulation.model.iceflow
    SIA2D_cache = simulation.cache.iceflow
    glacier_idx = SIA2D_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]

    # Retrieve models and parameters
    params = simulation.parameters
    target = simulation.model.trainable_components.target

    # Retrieve parameters
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy

    # First, enforce values to be positive
    H = map(x -> ifelse(x > 0.0, x, 0.0), H)
    # Update glacier surface altimetry
    S = B .+ H

    ### Computation of effective diffusivity
    dSdx = Huginn.diff_x(S) ./ Δx
    dSdy = Huginn.diff_y(S) ./ Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)

    # Compute surface slope
    ∇S = (∇Sx .^ 2 .+ ∇Sy .^ 2) .^ (1/2)

    # Compute average ice thickness
    H̄ = Huginn.avg(H)

    # Store temporary variables for use with the laws
    SIA2D_cache.∇S .= ∇S
    SIA2D_cache.H̄ .= H̄

    θ = isnothing(simulation.model.trainable_components) ? nothing :
        simulation.model.trainable_components.θ
    Huginn.apply_all_non_callback_laws!(
        SIA2D_model, SIA2D_cache, simulation, glacier_idx, t, θ)

    # Compute diffusivity based on target objective
    D = Diffusivity(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )

    # TODO: Clip dSdx and dSdx for conservation of mass condition

    ### Computation of partial derivatives of diffusivity
    ∂D∂H_dual = ∂Diffusivity∂H(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )
    ∂D∂H = Huginn.avg(∂D∂H_dual)

    β = ∂Diffusivity∂∇H(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
        glacier = glacier, params = params
    )

    # Derivatives evaluated in dual grid
    ∂D∂∇H_x = β .* ∇Sx
    ∂D∂∇H_y = β .* ∇Sy

    ### Computation of term ∇⋅(D⋅∇λ)

    # Compute flux components
    @views dλdx_edges = Huginn.diff_x(λ[:, 2:(end - 1)]) ./ Δx
    @views dλdy_edges = Huginn.diff_y(λ[2:(end - 1), :]) ./ Δy

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
        λ::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R,
    ) where {R <: Real, SIM <: Simulation}

Implementation of the continuous VJP of the SIA2D equation with respect to θ.
Given λ, H and θ, it returns the VJP of λ^T * ∂(SIA2D)/∂θ (θ).

Arguments:

  - `θ`: Vector of parameters
  - `λ::Matrix{R}`: Adjoint state, also called output gradient in reverse-mode AD.
  - `H::Matrix{R}`: Ice thickness which corresponds to the input state of the SIA2D.
  - `simulation::SIM`: Simulation parameters.
  - `t::R`: Time value, not used as SIA2D is time independent.

Returns:

  - `∂θ`: Jacobian vector product with respect to θ, also called input gradient in
    reverse-mode AD. It has the same type as θ.
"""
function VJP_λ_∂SIA∂θ_continuous(
        λ::Matrix{R},
        H::Matrix{R},
        θ,
        simulation::SIM,
        t::R
) where {R <: Real, SIM <: Simulation}
    SIA2D_model = simulation.model.iceflow
    SIA2D_cache = simulation.cache.iceflow
    glacier_idx = SIA2D_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]

    # Retrieve models and parameters
    params = simulation.parameters
    target = simulation.model.trainable_components.target

    # Retrieve parameters
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy

    # First, enforce values to be positive
    H = map(x -> ifelse(x > 0.0, x, 0.0), H)
    # Update glacier surface altimetry
    S = B .+ H

    ### Computation of effective diffusivity
    dSdx = Huginn.diff_x(S) ./ Δx
    dSdy = Huginn.diff_y(S) ./ Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)

    @views dSdx_edges = Huginn.diff_x(S[:, 2:(end - 1)]) ./ Δx
    @views dSdy_edges = Huginn.diff_y(S[2:(end - 1), :]) ./ Δy

    # Compute surface slope
    ∇S = (∇Sx .^ 2 .+ ∇Sy .^ 2) .^ (1/2)

    # Compute average ice thickness
    H̄ = Huginn.avg(H)

    # Store temporary variables for use with the laws
    SIA2D_cache.∇S .= ∇S
    SIA2D_cache.H̄ .= H̄

    θ = isnothing(simulation.model.trainable_components) ? nothing :
        simulation.model.trainable_components.θ
    Huginn.apply_all_non_callback_laws!(
        SIA2D_model, SIA2D_cache, simulation, glacier_idx, t, θ)

    # Gradient wrt θ
    ∂D∂θ = ∂Diffusivity∂θ(
        target;
        H̄ = H̄, ∇S = ∇S, θ = θ,
        simulation = simulation, glacier_idx = glacier_idx, t = t,
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
    @tullio ∇_∂D∂θ_∇S[i, j, k] := Fxx[pad(i-1, 1, 1), pad(j-1, 1, 1), k] +
                                  Fyy[pad(i-1, 1, 1), pad(j-1, 1, 1), k]

    # Evaluate numerical integral for loss
    @tullio ∂θ_v[k] := ∇_∂D∂θ_∇S[i, j, k] * λ[i, j]

    ∂θ = Vector2ComponentVector(∂θ_v, θ)

    return ∂θ
end
