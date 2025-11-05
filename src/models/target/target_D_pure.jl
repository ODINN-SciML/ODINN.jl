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
    n_interp_half::Int = 100
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
    iceflow_cache = simulation.cache.iceflow
    iceflow_model = simulation.model.iceflow

    # TODO: this can be replace by interpolation too!
    U = iceflow_model.U.f.f(iceflow_cache.U, (; H̄ = H̄, ∇S = ∇S), θ)

    # Include extra dependency in H for U law:
    if size(U) == size(H̄)
        D = H̄ .* U
    elseif (size(U) .+ 1) == size(H̄)
        D = Huginn.avg(H̄) .* U
    else
        throw("Not matching dimensions between U (∇S) and H̄. size(U)=$(size(U)) but size(H̄)=$(size(H̄))")
    end
    return D
end

function ∂Diffusivity∂H(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )

    ∂H∂H = map(h -> h > 0.0 ? 1.0 : 0.0, H̄)

    # Derivative of the output of the NN with respect to input layer
    δH = 1e-4 .* ones(size(H̄))

    # TODO: This can also be replace by interpolation
    D₊ = Diffusivity(target; H̄ = H̄ + δH, ∇S = ∇S, θ, simulation, glacier_idx, t, glacier, params)
    D₋ = Diffusivity(target; H̄ = H̄ - δH, ∇S = ∇S, θ, simulation, glacier_idx, t, glacier, params)

    # Compute central difference derivative
    ∂D∂H_NN = (D₊ .- D₋) ./ (2.0 .* δH)

    return ∂H∂H .* ∂D∂H_NN
end

function ∂Diffusivity∂∇H(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    # For now we ignore the derivative in surface slope
    δ∇H = 1e-6 .* ones(size(∇S))

    # TODO: This can also be replaced by interpolation
    D₊ = Diffusivity(target; H̄ = H̄, ∇S = ∇S + δ∇H, θ, simulation, glacier_idx, t, glacier, params)
    D₋ = Diffusivity(target; H̄ = H̄, ∇S = ∇S - δ∇H, θ, simulation, glacier_idx, t, glacier, params)

    # Compute central difference derivative
    ∂D∂∇S = (D₊ .- D₋) ./ (2.0 .* δ∇H)

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

    ∂spatial = map(h -> h > 0.0 ? 1.0 : 0.0, H̄)

    ∂D∂θ = zeros(size(H̄)..., only(size(θ)))
    @assert size(H̄) == size(∇S)

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    if interpolation == :None
        """
        Computes derivative at each pixel using the exact numerical value of H̄ at each
        point in the glacier. Slower but more precise.
        """
        for i in axes(H̄, 1), j in axes(H̄, 2)
            if H̄[i, j] == 0.0
                continue
            end
            ∂law∂θ!(backend, iceflow_model.U, iceflow_cache.U, iceflow_cache.U_prep_vjps, (; H̄=H̄[i, j], ∇S=∇S[i, j]), θ)
            ∂D∂θ[i, j, :] .= ∂spatial[i, j] * iceflow_cache.U.vjp_θ[i, j, :] * H̄[i, j]
        end

    elseif interpolation == :Linear
        """
        Interpolation of the gradient as function of values of H̄.
        Introduces interpolation errors but it is faster and probably sufficient depending
        the desired level of precision for the gradients.
        We construct an interpolator with quantiles and equal-spaced points.
        """
        # Unpack gradient interpolation
        grad_itp = iceflow_cache.U.interp_θ
        # Compute spatial distributed gradient
        for i in axes(H̄, 1), j in axes(H̄, 2)
            # Include extra contribution of ice thickness H
            ∂D∂θ[i, j, :] .= ∂spatial[i, j] * grad_itp(H̄[i, j], ∇S[i, j]) * H̄[i, j]
        end
    else
        throw("Method to spatially compute gradient with respect to H̄ not specified.")
    end

    return ∂D∂θ
end

"""

Function to evaluate derivatives of ice surface velocity in D inversion.

TODO: This functions right now just make a call to the regular functions used for the
calculation of the adjoint. This is not correct, but we keep it as this for now until
we figure out how to do this in the case of the D inversion.
"""
function Diffusivityꜛ(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    return Diffusivity(
        target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
end

function ∂Diffusivityꜛ∂H(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    return ∂Diffusivity∂H(
        target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
end

function ∂Diffusivityꜛ∂∇H(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    return ∂Diffusivity∂∇H(
        target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
end

function ∂Diffusivityꜛ∂θ(
    target::SIA2D_D_target;
    H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
    return ∂Diffusivity∂θ(
        target;
        H̄, ∇S, θ, simulation, glacier_idx, t, glacier, params
    )
end
