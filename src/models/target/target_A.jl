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
    (; n, A) = iceflow_cache
    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)
    return (
            S(iceflow_model, iceflow_cache, params) .+ A.value .* Γ_no_A .* H̄
        ) .* H̄.^(n.value .+ 1) .* ∇S.^(n.value .- 1)
end

function ∂Diffusivity∂H(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n
    return (
            (n.value .+ 1) .* S(iceflow_model, iceflow_cache, params) .+ Γ(iceflow_model, iceflow_cache, params) .* H̄ .* (n.value .+ 2)
        ) .* H̄.^n.value .* ∇S.^(n.value .- 1)
end

function ∂Diffusivity∂∇H(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n
    return (
            S(iceflow_model, iceflow_cache, params) .+ Γ(iceflow_model, iceflow_cache, params) .* H̄
        ) .* (n.value .- 1) .* H̄.^(n.value .+ 1) .* ∇S.^(n.value .- 3)
end

function ∂Diffusivity∂θ(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n
    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H̄.^(n.value .+ 2) .* ∇S.^(n.value .- 1)

    # if is_callback_law(iceflow_model.A)
    #     @assert "The A law cannot be a callback law as it needs to be differentiated in ∂Diffusivity∂θ. To support A as a callback law, you need to update the structure of the adjoint code computation."
    # end

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    # Spare computations in the case where the f_VJP_θ function of A does nothing
    skipInputs = isa(simulation.model.iceflow.A, Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, CustomVJP}) &&
        isa(simulation.model.iceflow.A.f_VJP_θ.f, typeof(Sleipnir.emptyVJPWithInputs))
    inputs = skipInputs ? nothing : generate_inputs(iceflow_model.A.f.inputs, simulation, glacier_idx, t)
    ∂law∂θ!(backend, iceflow_model.A, iceflow_cache.A, iceflow_cache.A_prep_vjps, inputs, θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, iceflow_cache.A.vjp_θ)
end

function Diffusivityꜛ(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; n, A) = iceflow_cache
    Γꜛ_no_A = Γꜛ(iceflow_model, iceflow_cache, params; include_A = false)
    return (
            Sꜛ(iceflow_model, iceflow_cache, params) .+ A.value .* Γꜛ_no_A
        ) .* H̄.^(n.value .+ 1) .* ∇S.^(n.value .- 1)
end

function ∂Diffusivityꜛ∂H(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n
    return (
            Sꜛ(iceflow_model, iceflow_cache, params) .+ Γꜛ(iceflow_model, iceflow_cache, params)
        ) .* (n.value .+ 1) .* H̄.^n.value .* ∇S.^(n.value .- 1)
end

function ∂Diffusivityꜛ∂∇H(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n
    return (
            Sꜛ(iceflow_model, iceflow_cache, params) .+ Γꜛ(iceflow_model, iceflow_cache, params)
        ) .* (n.value .- 1) .* H̄.^(n.value .+ 1) .* ∇S.^(n.value .- 3)
end

function ∂Diffusivityꜛ∂θ(
    target::SIA2D_A_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n
    Γꜛ_no_A = Γꜛ(iceflow_model, iceflow_cache, params; include_A = false)
    ∂A_spatial = Γꜛ_no_A .* H̄.^(n.value .+ 1) .* ∇S.^(n.value .- 1)

    # if is_callback_law(iceflow_model.A)
    #     @assert "The A law cannot be a callback law as it needs to be differentiated in ∂Diffusivityꜛ∂θ. To support A as a callback law, you need to update the structure of the adjoint code computation."
    # end

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    # Spare computations in the case where the f_VJP_θ function of A does nothing
    skipInputs = isa(simulation.model.iceflow.A, Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, CustomVJP}) &&
        isa(simulation.model.iceflow.A.f_VJP_θ.f, typeof(Sleipnir.emptyVJPWithInputs))
    inputs = skipInputs ? nothing : generate_inputs(iceflow_model.A.f.inputs, simulation, glacier_idx, t)
    ∂law∂θ!(backend, iceflow_model.A, iceflow_cache.A, iceflow_cache.A_prep_vjps, inputs, θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, iceflow_cache.A.vjp_θ)
end
