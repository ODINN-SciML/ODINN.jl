export SIA2D_D_target

"""
    SIA2D_D_target(; interpolation=:None, n_interp_half=20,
                     prescale=nothing, postscale=nothing)

Inversion of general diffusivity as a function of physical parameters.

D(H, ∇S, θ) = H * NN(H, ∇S; θ)

So now we are learning the velocity field given by D * ∇S. This inversion is similar to
learnign the velocity field assuming that this is parallel to the gradient in surface ∇S.

# Arguments
- `interpolation::Symbol = :None`: Specifies the interpolation method. Options include `:Linear`, `:None`.
- `n_interp_half::Int = 20`: Half-width of the interpolation stencil. Determines resolution of interpolation.
- `prescale::Union{Fin, Nothing} = nothing`: Optional prescaling function or factor applied before parametrization. Must be of type `Fin` or `nothing`.
- `postscale::Union{Fout, Nothing} = nothing`: Optional postscaling function or factor applied after parametrization. Must be of type `Fout` or `nothing`.

# Type Parameters
- `Fin`: Type of the prescale function or operator.
- `Fout`: Type of the postscale function or operator.

# Supertype
- `AbstractSIA2DTarget`: Inherits from the abstract target type for 2D SIA modeling.

# Returns
- An instance of `SIA2D_D_target` configured with optional scaling and interpolation parameters.
"""
@kwdef struct SIA2D_D_target <: AbstractSIA2DTarget
    interpolation::Symbol = :None
    n_interp_half::Int = 20
    prescale_provided::Bool = false
    postscale_provided::Bool = false
end

targetType(::SIA2D_D_target) = :D

"""
    Diffusivity(target::SIA2D_D_target; H, ∇S, θ, iceflow_model, glacier, params)

Compute the effective diffusivity field for a 2D shallow ice model using the diagnostic `target` and 
a predicted velocity matrix `U`.

This function uses a learned or specified model to estimate the velocity matrix `U`, then
calculates the diffusivity as either `H .* U` (if dimensions match) or the averaged `H` times `U`
if dimensions differ by one grid cell (staggered grid). Errors if dimensions are incompatible.

# Arguments
- `target::SIA2D_D_target`: Diagnostic target object defining interpolation and scaling rules.

# Keyword Arguments
- `H`: Ice thickness.
- `∇S`: Ice surface slope.
- `θ`: Parameters of the model.
- `iceflow_model`: Iceflow model used for simulation.
- `glacier`: Glacier data.
- `params`: Model parameters.

# Returns
- A matrix of diffusivity values with the same shape as `H` or staggered by one cell, depending on `U`.

# Throws
- An error if the dimensions of `U` and `H` are not compatible for diffusivity calculation.

# Notes
Supports both grid-matched and staggered configurations by averaging `H` where necessary.
"""
function Diffusivity(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    U = iceflow_cache.U
    if size(U) == size(H̄)
        return H̄ .* U
    elseif (size(U) .+ 1) == size(H̄)
        return Huginn.avg(H̄) .* U
    else
        throw("Not matching dimensions between U (∇S) and H̄. size(U)=$(size(U)) but size(H̄)=$(size(H̄))")
    end
end

function ∂Diffusivity∂H(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow

    # Neural network has already been evaluated in VJPs
    ∂D∂H_no_NN = iceflow_cache.U
    ∂H∂H = map(h -> h > 0.0 ? 1.0 : 0.0, H̄)
    ∂D∂H_no_NN .= ∂H∂H .* ∂D∂H_no_NN

    # Derivative of the output of the NN with respect to input layer
    δH = 1e-4 .* ones(size(H̄))
    iceflow_model.U.f.f(iceflow_cache.∂U∂H, (; H̄=H̄+δH, ∇S=∇S), θ)
    a = iceflow_cache.∂U∂H .* (H̄+δH)
    iceflow_model.U.f.f(iceflow_cache.∂U∂H, (; H̄=H̄, ∇S=∇S), θ)
    b = iceflow_cache.∂U∂H .* H̄
    ∂D∂H_NN = (a .- b) ./ δH

    return ∂D∂H_no_NN + ∂D∂H_NN
end

function ∂Diffusivity∂∇H(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow

    # For now we ignore the derivative in surface slope
    δ∇H = 1e-6 .* ones(size(∇S))
    iceflow_model.U.f.f(iceflow_cache.∂U∂H, (; H̄=H̄, ∇S=∇S+δ∇H), θ)
    a = iceflow_cache.∂U∂H .* H̄
    iceflow_model.U.f.f(iceflow_cache.∂U∂H, (; H̄=H̄, ∇S=∇S), θ)
    b = iceflow_cache.∂U∂H .* H̄
    ∂D∂∇S = (a .- b) ./ δ∇H

    return ∂D∂∇S
end

function ∂Diffusivity∂θ(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow

    if is_callback_law(iceflow_model.U)
        @assert "The U law cannot be a callback law as it needs to be differentiated in ∂Diffusivity∂θ. To support U as a callback law, you need to update the structure of the adjoint code computation."
    end

    # Extract relevant parameters specific from the target
    interpolation = target.interpolation
    n_interp_half = target.n_interp_half

    # ∂spatial = ones(size(H̄)...)
    ∂spatial = map(h -> h > 0.0 ? 1.0 : 0.0, H̄)

    ∂D∂θ = zeros(size(H̄)..., only(size(θ)))
    @assert size(H̄) == size(∇S)

    if interpolation == :None
        """
        Computes derivative at each pixel using the exact numerical value of H̄ at each
        point in the glacier. Slower but more precise.
        """
        for i in axes(H̄, 1), j in axes(H̄, 2)
            if H̄[i, j] == 0.0
                continue
            end
            ∇θ_point, = Zygote.gradient(_θ -> iceflow_model.U.f.f(
                iceflow_cache.∂U∂θ,
                (; H̄=H̄[i, j], ∇S=∇S[i, j]), _θ
                )*H̄[i, j], θ)
            ∂D∂θ[i, j, :] .= ∂spatial[i, j] * ComponentVector2Vector(∇θ_point)
        end

    elseif interpolation == :Linear
        """
        Interpolation of the gradient as function of values of H̄.
        Introduces interpolation errors but it is faster and probably sufficient depending
        the decired level of precision for the gradients.
        We construct an interpolator with quantiles and equal-spaced points.
        """

        # Interpolation for H̄
        H_interp = create_interpolation(H̄; n_interp_half = n_interp_half)
        # Interpolation for ∇S
        ∇S_interp = create_interpolation(∇S; n_interp_half = n_interp_half)

        if sum(H̄ .> 0.0) < 2.0 * length(H_interp) * length(∇S_interp)
            @warn "The total number of AD evaluations using interpolations is comparable to the total number of AD operations required to compute the derivative purely with AD with no interpolation. Recomendation is to switch to interpolation = :None"
        end

        # Compute exact gradient in certain values of H̄ and ∇S
        grads = [zeros(only(size(θ))) for i = 1:length(H_interp), j = 1:length(∇S_interp)]

        # TODO: Check if all these gradints cannot be computed at once withing Lux
        for (i, h) in enumerate(H_interp), (j, ∇s) in enumerate(∇S_interp)
            ∇θ_point, = Zygote.gradient(_θ -> iceflow_model.U.f.f(
                iceflow_cache.∂U∂θ,
                (; H̄=h, ∇S=∇s), _θ
                )*h, θ)
            grads[i, j] .= ComponentVector2Vector(∇θ_point)
        end
        # Create interpolation for gradient
        grad_itp = interpolate((H_interp, ∇S_interp), grads, Gridded(Linear()))

        # Compute spatial distributed gradient
        for i in axes(H̄, 1), j in axes(H̄, 2)
            ∂D∂θ[i, j, :] .= ∂spatial[i, j] * grad_itp(H̄[i, j], ∇S[i, j])
        end
    else
        throw("Method to spatially compute gradient with respect to H̄ not specified.")
    end

    return ∂D∂θ
end
