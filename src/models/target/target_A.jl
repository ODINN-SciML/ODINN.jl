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
        target::SIA2D_A_target;
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
        target::SIA2D_A_target;
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
        target::SIA2D_A_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n
    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H̄ .^ (n.value .+ 2) .* ∇S .^ (n.value .- 1)

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    # Spare computations in the case where the f_VJP_θ function of A does nothing
    skipInputs = isa(
        simulation.model.iceflow.A, Law{
            <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, CustomVJP}) &&
                 isa(simulation.model.iceflow.A.f_VJP_θ.f, typeof(Sleipnir.emptyVJPWithInputs))
    inputs = skipInputs ? nothing :
             generate_inputs(iceflow_model.A.f.inputs, simulation, glacier_idx, t)
    ∂law∂θ!(iceflow_model.A, iceflow_cache.A, iceflow_cache.A_prep_vjps, backend, inputs, θ)

    # Create a tensor with both elements
    if isa(iceflow_cache.A, Union{ScalarCache, ScalarCacheGlacierId})
        # Glacier wide VJP
        return cartesian_tensor(∂A_spatial, iceflow_cache.A.vjp_θ)
    else
        # Spatially distributed VJP
        return sparse_cartesian_tensor(∂A_spatial, iceflow_cache.A.vjp_θ)
    end
end

function Velocityꜛ(
        target::SIA2D_A_target;
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
        target::SIA2D_A_target;
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
        target::SIA2D_A_target;
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
        target::SIA2D_A_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n
    Γꜛ_no_A = Γꜛ(iceflow_model, iceflow_cache, params; include_A = false)
    ∂A_spatial = Γꜛ_no_A .* H̄ .^ (n.value .+ 1) .* ∇S .^ (n.value .- 1)

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    # Spare computations in the case where the f_VJP_θ function of A does nothing
    skipInputs = isa(
        simulation.model.iceflow.A, Law{
            <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, CustomVJP}) &&
                 isa(simulation.model.iceflow.A.f_VJP_θ.f, typeof(Sleipnir.emptyVJPWithInputs))
    inputs = skipInputs ? nothing :
             generate_inputs(iceflow_model.A.f.inputs, simulation, glacier_idx, t)
    ∂law∂θ!(iceflow_model.A, iceflow_cache.A, iceflow_cache.A_prep_vjps, backend, inputs, θ)

    # Create a tensor with both elements
    if isa(iceflow_cache.A, Union{ScalarCache, ScalarCacheGlacierId})
        # Glacier wide VJP
        return cartesian_tensor(∂A_spatial, iceflow_cache.A.vjp_θ)
    else
        # Spatially distributed VJP
        return sparse_cartesian_tensor(∂A_spatial, iceflow_cache.A.vjp_θ)
    end
end
