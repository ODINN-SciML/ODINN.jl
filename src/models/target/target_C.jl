export SIA2D_C_target

"""
    SIA2D_C_target <: AbstractSIA2DTarget

Struct to define inversion where only the sliding coefficient `C` is learnt.
"""

@kwdef struct SIA2D_C_target <: AbstractSIA2DTarget
end

targetType(::SIA2D_C_target) = :C

### Target functions

function Diffusivity(
        target::SIA2D_C_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; A, n, p, q) = iceflow_cache
    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)
    return (
        S(iceflow_model, iceflow_cache, params) .* H̄ .^ (p.value .- q.value .+ 1) .*
        ∇S .^ (p.value .- 1)
        +
        A.value .* Γ_no_A .* H̄ .^ (n.value .+ 2) .* ∇S .^ (n.value .- 1)
    )
end

function ∂Diffusivity∂H(
        target::SIA2D_C_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; A, n, p, q) = iceflow_cache
    return (
        (p.value .- q.value .+ 1) .* S(iceflow_model, iceflow_cache, params) .*
        H̄ .^ (p.value .- q.value) .* ∇S .^ (p.value .- 1)
        +
        A.value .* Γ(iceflow_model, iceflow_cache, params; include_A = false) .*
        (n.value .+ 2) .* H̄ .^ (n.value .+ 1) .* ∇S .^ (n.value .- 1)
    )
end

function ∂Diffusivity∂∇H(
        target::SIA2D_C_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; A, n, p, q) = iceflow_cache
    return (
        S(iceflow_model, iceflow_cache, params) .* (p.value .- 1) .*
        H̄ .^ (p.value .- q.value .+ 1) .* ∇S .^ (p.value .- 3)
        +
        A.value .* Γ(iceflow_model, iceflow_cache, params; include_A = false) .*
        (n.value .- 1) .* H̄ .^ (n.value .+ 2) .* ∇S .^ (n.value .- 3)
    )
end

function ∂Diffusivity∂θ(
        target::SIA2D_C_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; ρ, g) = params.physical
    (; A, n, p, q) = iceflow_cache
    ∂C_spatial = (ρ * g) .^ (p.value .- q.value) .* H̄ .^ (p.value .+ 1 .- q.value) .* ∇S .^ (p.value .- 1)

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    skipInputs = isa(
        simulation.model.iceflow.C, Law{
            <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, CustomVJP}) &&
                 isa(simulation.model.iceflow.C.f_VJP_θ.f, typeof(Sleipnir.emptyVJPWithInputs))
    inputs = skipInputs ? nothing :
             generate_inputs(iceflow_model.C.f.inputs, simulation, glacier_idx, t)
    ∂law∂θ!(iceflow_model.C, iceflow_cache.C, iceflow_cache.C_prep_vjps, backend, inputs, θ)

    if isa(iceflow_cache.C, Union{ScalarCache, ScalarCacheGlacierId})
        return cartesian_tensor(∂C_spatial, iceflow_cache.C.vjp_θ)
    else
        return sparse_cartesian_tensor(∂C_spatial, iceflow_cache.C.vjp_θ)
    end
end

function Velocityꜛ(
        target::SIA2D_C_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; A, n, p, q) = iceflow_cache
    Γꜛ_no_A = Γꜛ(iceflow_model, iceflow_cache, params; include_A = false)
    return (
        S(iceflow_model, iceflow_cache, params) .* (p.value .- q.value .+ 2) *
        H̄ .^ (p.value .- q.value .+ 1) .* ∇S .^ (n.value .- 1)
        +
        A.value .* Γꜛ_no_A .* H̄ .^ (n.value .+ 1) .* ∇S .^ (n.value .- 1)
    )
end

function ∂Velocityꜛ∂H(
        target::SIA2D_C_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; A, n, p, q) = iceflow_cache
    return (
        S(iceflow_model, iceflow_cache, params) .* (p.value .- q.value .+ 2) *
        H̄ .^ (p.value .- q.value) .* ∇S .^ (n.value .- 1)
        +
        A.value .* Γꜛ(iceflow_model, iceflow_cache, params; include_A = false) .*
        (n.value .+ 1) .* H̄ .^ n.value .* ∇S .^ (n.value .- 1)
    )
end

function ∂Velocityꜛ∂∇H(
        target::SIA2D_C_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; A, n, p, q) = iceflow_cache
    return (
        S(iceflow_model, iceflow_cache, params) .* (p.value .- q.value .+ 2) .*
        (p.value .- 1) * H̄ .^ (p.value .- q.value .+ 1) .* ∇S .^ (n.value .- 3)
        +
        A.value .* Γꜛ(iceflow_model, iceflow_cache, params; include_A = false) .*
        (n.value .- 1) .* H̄ .^ (n.value .+ 1) .* ∇S .^ (n.value .- 3)
    )
end

function ∂Velocityꜛ∂θ(
        target::SIA2D_C_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; ρ, g) = params.physical
    (; A, n, p, q) = iceflow_cache
    ∂C_spatial = (ρ * g) .^ (p.value .- q.value) .* (p.value .- q.value .+ 2) .* H̄ .^ (p.value .- q.value .+ 1) .* ∇S .^ (n.value .- 1)

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    skipInputs = isa(
        simulation.model.iceflow.C, Law{
            <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, CustomVJP}) &&
                 isa(simulation.model.iceflow.C.f_VJP_θ.f, typeof(Sleipnir.emptyVJPWithInputs))
    inputs = skipInputs ? nothing :
             generate_inputs(iceflow_model.C.f.inputs, simulation, glacier_idx, t)
    ∂law∂θ!(iceflow_model.C, iceflow_cache.C, iceflow_cache.C_prep_vjps, backend, inputs, θ)

    if isa(iceflow_cache.C, Union{ScalarCache, ScalarCacheGlacierId})
        return cartesian_tensor(∂C_spatial, iceflow_cache.C.vjp_θ)
    else
        return sparse_cartesian_tensor(∂C_spatial, iceflow_cache.C.vjp_θ)
    end
end
