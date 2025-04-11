# export VJP_λ_∂SIA∂H_continuous, VJP_λ_∂SIA∂θ_continuous


#######################################
######     Discrete Adjoint      ######
#######################################

### Utils for discrete adjoint

function diff_x_adjoint(I, Δx)
    O = zeros(Sleipnir.Float, (size(I,1)+1,size(I,2)))
    O[begin+1:end,:] += I
    O[1:end-1,:] -= I
    return O / Δx
end

function diff_y_adjoint(I, Δy)
    O = zeros(Sleipnir.Float, (size(I,1),size(I,2)+1))
    O[:,begin+1:end] += I
    O[:,1:end - 1] -= I
    return O / Δy
end

function clamp_borders_dx(dS, H, η₀, Δx)
    return max.(min.(dS,  η₀ * H[2:end, 2:end-1]/Δx), -η₀ * H[1:end-1, 2:end-1]/Δx)
end

function clamp_borders_dx_adjoint!(∂dS, ∂H, ∂C, η₀, Δx, H, dS)
    # Note: this implementation doesn't hold if H is negative
    ∂dS .= ∂C .* ((dS .< η₀ * H[2:end, 2:end-1]/Δx) .& (dS .> -η₀ * H[1:end-1, 2:end-1]/Δx))
    ∂H[1:end-1, 2:end-1] .= - (η₀ * ∂C / Δx) .* (dS .< -η₀ * H[1:end-1, 2:end-1]/Δx)
    ∂H[2:end, 2:end-1] += (η₀ * ∂C / Δx) .* (dS .> η₀ * H[2:end, 2:end-1]/Δx)
end

function clamp_borders_dy(dS, H, η₀, Δy)
    return max.(min.(dS,  η₀ * H[2:end-1, 2:end]/Δy), -η₀ * H[2:end-1, 1:end-1]/Δy)
end

function clamp_borders_dy_adjoint!(∂dS, ∂H, ∂C, η₀, Δy, H, dS)
    # Note: this implementation doesn't hold if H is negative
    ∂dS .= ∂C .* ((dS .< η₀ * H[2:end-1, 2:end]/Δy) .& (dS .> -η₀ * H[2:end-1, 1:end-1]/Δy))
    ∂H[2:end-1, 1:end-1] .= - (η₀ * ∂C / Δy) .* (dS .< -η₀ * H[2:end-1, 1:end-1]/Δy)
    ∂H[2:end-1, 2:end] += (η₀ * ∂C / Δy) .* (dS .> η₀ * H[2:end-1, 2:end]/Δy)
end

function avg_adjoint(I)
    O = zeros(Sleipnir.Float, (size(I,1)+1,size(I,2)+1))
    O[1:end-1,1:end-1] += I
    O[2:end,1:end-1] += I
    O[1:end-1,2:end] += I
    O[2:end,2:end] += I
    return 0.25*O
end

function avg_x_adjoint(I)
    O = zeros(Sleipnir.Float, (size(I, 1)+1, size(I, 2)))
    O[1:end-1,:] += I
    O[2:end,:] += I
    return 0.5*O
end

function avg_y_adjoint(I)
    O = zeros(Sleipnir.Float, (size(I, 1), size(I, 2)+1))
    O[:,1:end-1] += I
    O[:,2:end] += I
    return 0.5*O
end


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

    # Retrieve parameters
    params = simulation.parameters
    A = SIA2D_model.A
    n = SIA2D_model.n
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    Γ = SIA2D_model.Γ
    ρ = simulation.parameters.physical.ρ
    g = simulation.parameters.physical.g

    # Retrieve target
    target = params.UDE.target

    # First, enforce values to be positive
    map!(x -> ifelse(x>0.0,x,0.0), H, H)
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = Huginn.diff_x(S)/Δx
    dSdy = Huginn.diff_y(S)/Δy
    ∇Sx = Huginn.avg_y(dSdx)
    ∇Sy = Huginn.avg_x(dSdy)
    ∇S = (∇Sx.^2 .+ ∇Sy.^2).^((n[] - 1)/2)

    H̄ = Huginn.avg(H)
    Γ = 2.0 * A[] * (ρ * g)^n[] / (n[]+2) # 1 / m^3 s
    # @assert Γ == 2.0 * A[] * (ρ * g)^n[] / (n[]+2)
    D = Γ .* H̄.^(n[] + 2) .* ∇S

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
    # α = Γ .* (n[]+2) .* H̄.^(n[]+1) .* ∇S
    α = target.∂D∂H(
        H = H̄, ∇S = ∇S, θ = θ,
        model = SIA2D_model, glacier = glacier, params = params
    )
    # Equals ∂D/∂(∇H)
    # β = Γ .* (n[]-1) .* H̄.^(n[]+2) .* (∇Sx.^2 .+ ∇Sy.^2).^((n[] - 3)/2)
    β = target.∂D∂∇H(
        H = H̄, ∇S = ∇S, θ = θ,
        model = SIA2D_model, glacier = glacier, params = params
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
    ∇θ, = Zygote.gradient(_θ -> grad_apply_UDE_parametrization(_θ, simulation, batch_id), θ)
    ∂D∂θ = target.∂D∂θ(
        H = H̄, ∇S = ∇S, θ = θ, ∇θ = ∇θ,
        model = SIA2D_model, glacier = glacier, params = params
    )
    # Evaluate numerical integral for loss
    @tullio ∂θ_v[k] := ∂D∂θ[i, j, k] * D_adjoint[i, j]
    # Construct component vector
    ∂θ = Vector2ComponentVector(∂θ_v, θ)

    # D_adjoint = ∇ ⋅ (∇S ⋅ ∇λ)
    # fac = 2.0 * (ρ * g)^n[] / (n[]+2)
    # # Why this does not compute as with βx and βy?
    # ∂A_spatial = fac .* Huginn.avg(H).^(n[] + 2) .* ∇S .* D_adjoint
    # ∂A = sum(∂A_spatial)
    # ∇θ, = Zygote.gradient(_θ -> grad_apply_UDE_parametrization(_θ, simulation, batch_id), θ)
    # ∂θ = ∂A * ∇θ
    # ∂θ = target.∂D∂θ(avg(H), ∇S, params, ∇θ)

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

    ### Computation of effective diffusivity
    dSdx = Huginn.diff_x(S) ./ Δx
    dSdy = Huginn.diff_y(S) ./ Δy
    ∇S = (Huginn.avg_y(dSdx).^2 .+ Huginn.avg_x(dSdy).^2).^(1/2)
    # This is used to compute the diffusivity:
    Γ = 2.0 * A[] * (ρ * g)^n[] / (n[]+2) # 1 / m^3 s
    D = Γ .* Huginn.avg(H).^(n[] + 2) .* ∇S.^(n[] - 1)

    ### Computation of partial derivatives of diffusivity
    ∂D∂H_dual = (n[] + 2) .* Γ .* Huginn.avg(H).^(n[] + 1) .* ∇S.^(n[] - 1)
    ∂D∂H = Huginn.avg(∂D∂H_dual)

    # Derivatives evaluated in dual grid
    ∂D∂∇H_x = (n[] - 1) .* Γ .* Huginn.avg(H).^(n[] + 2) .* ∇S.^(n[] - 3) .* Huginn.avg_y(dSdx)
    ∂D∂∇H_y = (n[] - 1) .* Γ .* Huginn.avg(H).^(n[] + 2) .* ∇S.^(n[] - 3) .* Huginn.avg_x(dSdy)

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
    θ,
    λ::Matrix{R},
    H::Matrix{R},
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
    dSdx = Huginn.diff_x(S) ./ Δx
    dSdy = Huginn.diff_y(S) ./ Δy
    ∇S = (Huginn.avg_y(dSdx).^2 .+ Huginn.avg_x(dSdy).^2).^(1/2)
    Γ = 2.0 * (ρ * g)^n[] / (n[]+2)

    ### Computation of partial derivatives of diffusivity
    ∂D∂A = Γ .* Huginn.avg(H).^(n[] + 2) .* ∇S.^(n[] - 1)
    ∇θ, = Zygote.gradient(_θ -> grad_apply_UDE_parametrization(_θ, simulation, 1), θ)
    # ∂D∂θ = ∂D∂A .* ∇θ

    ∇λ∇S_x_edges = dSdx .* Huginn.diff_x(λ) ./ Δx
    ∇λ∇S_y_edges = dSdy .* Huginn.diff_y(λ) ./ Δy
    ∇λ∇S_x = Huginn.avg_y(∇λ∇S_x_edges)
    ∇λ∇S_y = Huginn.avg_x(∇λ∇S_y_edges)
    ∇λ∇S = ∇λ∇S_x .+ ∇λ∇S_y

    return - sum(∂D∂A .* ∇λ∇S) .* ∇θ
end

# Repeated function
function grad_apply_UDE_parametrization(θ, simulation::SIM, batch_id::I) where {I <: Integer, SIM <: Simulation}
    # We load the ML model with the parameters
    model = simulation.model.machine_learning.architecture
    st = simulation.model.machine_learning.st
    smodel = StatefulLuxLayer{true}(model, θ.θ, st)

    # We generate the ML parametrization based on the target
    if simulation.parameters.UDE.target.name == "A"
        min_NN = simulation.parameters.physical.minA
        max_NN = simulation.parameters.physical.maxA
        A = predict_A̅(smodel, [mean(simulation.glaciers[batch_id].climate.longterm_temps)], (min_NN, max_NN))[1]
        # println("Value of A during internal gradient evaluation:")
        # @show A
        return A
    end
end
