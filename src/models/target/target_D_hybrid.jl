export SIA2D_D_hybrid_target

"""
    SIA2D_D_hybrid_target{Fin, Fout} <: AbstractSIA2DTarget

Struct to define inversion where elements of the diffusivity D in the SIA equations are
replaced by a generic regressor. For this example, we consider the inversion of the form

    D(H̄, ∇S, θ) = ( C * (ρ * g)^n + 2 * H̄ * NeuralNet(T, H̄, ∇S; θ) / (n + 2) * (ρg)^n ) H̄^{n+1} |∇S|^{n-1}
"""

@kwdef struct SIA2D_D_hybrid_target <: AbstractSIA2DTarget
    interpolation::Symbol = :Linear
    n_interp_half::Int = 75
end

targetType(::SIA2D_D_hybrid_target) = :D_hybrid

# For this simple case, the target coincides with D, but not always.
# TODO: D should be cap to its maximum physical value. This can be done with one extra
# function and one extra differentiation.
function Diffusivity(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    return compute_D(
        target;
        H̄, ∇S, θ, iceflow_model, iceflow_cache, glacier, params
    )
end

function ∂Diffusivity∂H(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow

    # Allow different n for power in inversion of diffusivity
    # TODO: n is also inside Γ, so probably we want to grab this one too
    (; n, p, q, Y) = iceflow_cache
    n_H = iceflow_model.n_H_is_provided ? iceflow_cache.n_H : n.value
    n_∇S = iceflow_model.n_∇S_is_provided ? iceflow_cache.n_∇S : n.value

    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)
    ∂D∂H_no_NN = (
        (p.value .- q.value .+ 1) .* S(iceflow_model, iceflow_cache, params) .*
        H̄ .^ (p.value .- q.value) .* ∇S .^ (p.value .- 1)
        +
        (n_H .+ 2) .* Y.value .* Γ_no_A .* H̄ .^ (n_H .+ 1) .* ∇S .^ (n_∇S .- 1)
    )

    # Derivative of the output of the NN with respect to input layer
    # TODO: Change this to be done with AD or have this as an extra parameter.
    # This is done already in SphereUDE.jl with Lux
    δH = 1e-4 .* ones(size(H̄))
    # We don't use apply_law! because we want to evaluate with custom inputs
    temp = get_input(iAvgScalarTemp(), simulation, glacier_idx, t)
    iceflow_model.Y.f.f(iceflow_cache.Y, (; T = temp, H̄ = H̄+δH), θ)
    a = compute_D(
        target, iceflow_cache.Y.value;
        H̄ = H̄, ∇S, θ, iceflow_model, iceflow_cache, glacier, params
    )
    iceflow_model.Y.f.f(iceflow_cache.Y, (; T = temp, H̄ = H̄), θ)
    b = compute_D(
        target, iceflow_cache.Y.value;
        H̄ = H̄, ∇S, θ, iceflow_model, iceflow_cache, glacier, params
    )
    ∂D∂H_NN = (a .- b) ./ δH

    return ∂D∂H_no_NN + ∂D∂H_NN
end

function ∂Diffusivity∂∇H(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow

    (; n, p, q, Y) = iceflow_cache
    n_H = iceflow_model.n_H_is_provided ? iceflow_cache.n_H : n.value
    n_∇S = iceflow_model.n_∇S_is_provided ? iceflow_cache.n_∇S : n.value

    ∂D∂∇S_no_NN = (
        S(iceflow_model, iceflow_cache, params) .* (p.value .- 1) .*
        H̄ .^ (p.value .- q.value .+ 1) .* ∇S .^ (p.value .- 3)
        +
        Γ(iceflow_model, iceflow_cache, params; include_A = false) .* Y.value .*
        (n_∇S .- 1) .* H̄ .^ (n_H .+ 2) .* ∇S .^ (n_∇S .- 3)
    )

    return ∂D∂∇S_no_NN
end

function ∂Diffusivity∂θ(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n

    if is_callback_law(iceflow_model.Y)
        @assert "The Y law cannot be a callback law as it needs to be differentiated in ∂Diffusivity∂θ. To support Y as a callback law, you need to update the structure of the adjoint code computation."
    end

    # Extract relevant parameters specific from the target
    interpolation = target.interpolation
    n_interp_half = target.n_interp_half

    n_H = iceflow_model.n_H_is_provided ? iceflow_cache.n_H : n.value
    n_∇S = iceflow_model.n_∇S_is_provided ? iceflow_cache.n_∇S : n.value

    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H̄ .^ (n_H .+ 2) .* ∇S .^ (n_∇S .- 1)

    temp = get_input(iAvgScalarTemp(), simulation, glacier_idx, t)

    ∂D∂θ = zeros(size(H̄)..., only(size(θ)))

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    if interpolation == :None
        """
        Computes derivative at each pixel using the exact numerical value of H̄ at each
        point in the glacier. Slower but more precise.
        """
        for i in axes(H̄, 1), j in axes(H̄, 2)

            ∂law∂θ!(iceflow_model.Y, iceflow_cache.Y,
                iceflow_cache.Y_prep_vjps, backend, (; T = temp, H̄ = H̄[i, j]), θ)
            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * iceflow_cache.Y.vjp_θ
        end
    elseif interpolation == :Linear
        """
        Interpolation of the gradient as function of values of H̄.
        Introduces interpolation errors but it is faster and probably sufficient depending
        the decired level of precision for the gradients.
        We construct an interpolator with quantiles and equal-spaced points.
        """
        H_interp = create_interpolation(H̄; n_interp_half = n_interp_half)

        # Compute exact gradient in certain values of H̄
        grads = []
        # TODO: Check if all these gradints cannot be computed at once withing Lux
        for h in H_interp
            ∂law∂θ!(iceflow_model.Y, iceflow_cache.Y,
                iceflow_cache.Y_prep_vjps, backend, (; T = temp, H̄ = h), θ)
            push!(grads, deepcopy(iceflow_cache.Y.vjp_θ)) # Copy cache otherwise it points to the same place in memory
        end
        # Create interpolation for gradient
        grad_itp = interpolate((H_interp,), grads, Gridded(Linear()))

        # Compute spatial distributed gradient
        for i in axes(H̄, 1), j in axes(H̄, 2)

            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * grad_itp(H̄[i, j])
        end
    else
        throw("Method to spatially compute gradient with respect to H̄ not specified.")
    end

    return ∂D∂θ
end

function compute_D(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, iceflow_model, iceflow_cache, glacier, params
)
    # Use the value of Y in the cache

    (; n, p, q, Y) = iceflow_cache
    n_H = iceflow_model.n_H_is_provided ? iceflow_cache.n_H : n.value
    n_∇S = iceflow_model.n_∇S_is_provided ? iceflow_cache.n_∇S : n.value

    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)

    D = (
        S(iceflow_model, iceflow_cache, params) .* H̄ .^ (p.value .- q.value .+ 1) .*
        ∇S .^ (p.value .- 1)
        +
        Y.value .* Γ_no_A .* H̄ .^ (n_H .+ 2) .* ∇S .^ (n_∇S .- 1)
    )
    return D
end

function compute_D(
        target::SIA2D_D_hybrid_target, Y;
        H̄, ∇S, θ, iceflow_model, iceflow_cache, glacier, params
)
    # Use the value of Y provided as an argument

    (; n, p, q) = iceflow_cache
    n_H = iceflow_model.n_H_is_provided ? iceflow_cache.n_H : n.value
    n_∇S = iceflow_model.n_∇S_is_provided ? iceflow_cache.n_∇S : n.value

    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)

    D = (
        S(iceflow_model, iceflow_cache, params) .* H̄ .^ (p.value .- q.value .+ 1) .*
        ∇S .^ (p.value .- 1)
        +
        Y .* Γ_no_A .* H̄ .^ (n_H .+ 2) .* ∇S .^ (n_∇S .- 1)
    )
    return D
end

function Velocityꜛ(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    return compute_Velocityꜛ(
        target;
        H̄, ∇S, θ, iceflow_model, iceflow_cache, glacier, params
    )
end

function ∂Velocityꜛ∂H(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    (; ρ, g) = params.physical

    (; C, n, p, q, Y) = iceflow_cache
    n_H = isnothing(iceflow_model.n_H) ? n.value : iceflow_model.n_H
    n_∇S = isnothing(iceflow_model.n_∇S) ? n.value : iceflow_model.n_∇S

    Γ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)
    S_surf = C.value .* (ρ * g) .^ (p.value - q.value)
    ∂D∂H_no_NN = (
        (p.value .- q.value .+ 1) .* S_surf .* H̄ .^ (p.value .- q.value) .*
        ∇S .^ (p.value .- 1)
        +
        (n_H .+ 1) .* Y.value .* Γ_no_A .* H̄ .^ n_H .* ∇S .^ (n_∇S .- 1)
    )

    δH = 1e-4 .* ones(size(H̄))
    # We don't use apply_law! because we want to evaluate with custom inputs
    temp = get_input(iAvgScalarTemp(), simulation, glacier_idx, t)
    iceflow_model.Y.f.f(iceflow_cache.Y, (; T = temp, H̄ = H̄+δH), θ)
    a = compute_D(
        target, iceflow_cache.Y.value;
        H̄ = H̄, ∇S, θ, iceflow_model, iceflow_cache, glacier, params
    )
    iceflow_model.Y.f.f(iceflow_cache.Y, (; T = temp, H̄ = H̄), θ)
    b = compute_D(
        target, iceflow_cache.Y.value;
        H̄ = H̄, ∇S, θ, iceflow_model, iceflow_cache, glacier, params
    )
    ∂D∂H_NN = (a .- b) ./ δH

    return ∂D∂H_no_NN + ∂D∂H_NN
end

function ∂Velocityꜛ∂∇H(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow

    (; n, p, q, Y) = iceflow_cache
    n_H = isnothing(iceflow_model.n_H) ? n.value : iceflow_model.n_H
    n_∇S = isnothing(iceflow_model.n_∇S) ? n.value : iceflow_model.n_∇S

    ∂D∂∇S_no_NN = (
        (p.value .- 1) .* S(iceflow_model, iceflow_cache, params) .*
        H̄ .^ (p.value .- q.value .+ 1) .* ∇S .^ (p.value .- 3)
        +
        Γꜛ(iceflow_model, iceflow_cache, params; include_A = false) .* Y.value .*
        (n_∇S .- 1) .* H̄ .^ (n_H .+ 2) .* ∇S .^ (n_∇S .- 3)
    )

    return ∂D∂∇S_no_NN
end

function ∂Velocityꜛ∂θ(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
)
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    n = iceflow_cache.n

    if is_callback_law(iceflow_model.Y)
        @assert "The Y law cannot be a callback law as it needs to be differentiated in ∂Diffusivity∂θ. To support Y as a callback law, you need to update the structure of the adjoint code computation."
    end

    # Extract relevant parameters specific from the target
    interpolation = target.interpolation
    n_interp_half = target.n_interp_half

    n_H = isnothing(iceflow_model.n_H) ? n.value : iceflow_model.n_H
    n_∇S = isnothing(iceflow_model.n_∇S) ? n.value : iceflow_model.n_∇S

    Γ_no_A = Γꜛ(iceflow_model, iceflow_cache, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H̄ .^ (n_H .+ 1) .* ∇S .^ (n_∇S .- 1)

    temp = get_input(iAvgScalarTemp(), simulation, glacier_idx, t)

    ∂D∂θ = zeros(size(H̄)..., only(size(θ)))

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    if interpolation == :None
        """
        Computes derivative at each pixel using the exact numerical value of H̄ at each
        point in the glacier. Slower but more precise.
        """
        for i in axes(H̄, 1), j in axes(H̄, 2)

            ∂law∂θ!(iceflow_model.Y, iceflow_cache.Y,
                iceflow_cache.Y_prep_vjps, backend, (; T = temp, H̄ = H̄[i, j]), θ)
            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * iceflow_cache.Y.vjp_θ
        end
    elseif interpolation == :Linear
        """
        Interpolation of the gradient as function of values of H̄.
        Introduces interpolation errors but it is faster and probably sufficient depending
        the decired level of precision for the gradients.
        We construct an interpolator with quantiles and equal-spaced points.
        """
        H_interp = create_interpolation(H̄; n_interp_half = n_interp_half)

        # Compute exact gradient in certain values of H̄
        grads = []
        # TODO: Check if all these gradints cannot be computed at once withing Lux
        for h in H_interp
            ∂law∂θ!(iceflow_model.Y, iceflow_cache.Y,
                iceflow_cache.Y_prep_vjps, backend, (; T = temp, H̄ = h), θ)
            push!(grads, deepcopy(iceflow_cache.Y.vjp_θ)) # Copy cache otherwise it points to the same place in memory
        end
        # Create interpolation for gradient
        grad_itp = interpolate((H_interp,), grads, Gridded(Linear()))

        # Compute spatial distributed gradient
        for i in axes(H̄, 1), j in axes(H̄, 2)

            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * grad_itp(H̄[i, j])
        end
    else
        @error "Method to spatially compute gradient with respect to H̄ not specified."
    end

    return ∂D∂θ
end

function compute_Velocityꜛ(
        target::SIA2D_D_hybrid_target;
        H̄, ∇S, θ, iceflow_model, iceflow_cache, glacier, params
)
    # Use the value of Y in the cache

    (; n, p, q, Y) = iceflow_cache
    n_H = isnothing(iceflow_model.n_H) ? n.value : iceflow_model.n_H
    n_∇S = isnothing(iceflow_model.n_∇S) ? n.value : iceflow_model.n_∇S

    Γꜛ_no_A = Γ(iceflow_model, iceflow_cache, params; include_A = false)

    D = (
        S(iceflow_model, iceflow_cache, params) .* H̄ .^ (p.value .- q.value .+ 1) .*
        ∇S .^ (p.value .- 1)
        +
        Y.value .* Γꜛ_no_A .* H̄ .^ (n_H .+ 1) .* ∇S .^ (n_∇S .- 1)
    )
    return D
end
