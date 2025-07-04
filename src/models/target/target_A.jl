export SIA2D_A_target

"""
    SIA2D_A_target <: AbstractSIA2DTarget

Struct to define inversion where only the creep coefficient `A` is learnt.
"""

@kwdef struct SIA2D_A_target <: AbstractSIA2DTarget
end

targetType(::SIA2D_A_target) = :A

### Target functions

function Diffusivity(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; n, C, A) = iceflow_cache
    (; ρ, g) = params.physical
    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)
    return (C .* (ρ * g).^n .+ A .* Γ_no_A .* H̄) .* H̄.^(n .+ 1) .* ∇S.^(n .- 1)
end

function ∂Diffusivity∂H(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; n, C) = iceflow_cache
    (; ρ, g) = params.physical
    return ( C .* (n .+ 1) .* (ρ * g).^n .+ Γ(iceflow_model, iceflow_cache, params) .* H̄ .* (n .+ 2) ) .* H̄.^n .* ∇S.^(n .- 1)
end

function ∂Diffusivity∂∇H(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; n, C) = iceflow_cache
    (; ρ, g) = params.physical
    return ( C .* (ρ * g).^n .+ Γ(iceflow_model, iceflow_cache, params) .* H̄ ) .* (n .- 1) .* H̄.^(n .+ 1) .* ∇S.^(n .- 3)
end

function ∂Diffusivity∂θ(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n
    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H̄.^(n .+ 2) .* ∇S.^(n .- 1)

    if is_callback_law(iceflow_model.A)
        @assert "The A law cannot be a callback law as it needs to be differentiated in ∂Diffusivity∂θ. To support A as a callback law, you need to update the structure of the adjoint code computation."
    end

    ∇θ, = Zygote.gradient(_θ -> apply_law!(
        iceflow_model.A, iceflow_cache.A, simulation, glacier_idx, t, _θ),
        θ)
    ∇θ_cv = ComponentVector2Vector(∇θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, ∇θ_cv)
end
