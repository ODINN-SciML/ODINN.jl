export SIA2D_D_target

"""
    SIA2D_D_target(; interpolation=:None, n_interp_half=20,
                     prescale=nothing, postscale=nothing)

Inversion of general diffusivity as a function of physical parameters.

D(H, ∇S, θ) = H * NN(H, ∇S; θ)

So now we are learning the velocoty field given by D * ∇S. This inversion is similar to
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
@kwdef struct SIA2D_D_target{Fin, Fout} <: AbstractSIA2DTarget
    interpolation::Symbol
    n_interp_half::Int
    prescale::Union{Fin, Nothing}
    postscale::Union{Fout, Nothing}
end

"""
    SIA2D_D_target(; interpolation=:None, n_interp_half=20, 
                     prescale=nothing, postscale=nothing)

Construct a `SIA2D_D_target` instance with specified interpolation and optional prescale/postscale transformations.

This constructor infers the type parameters `Fin` and `Fout` from the provided `prescale` and `postscale` functions
(or `nothing`), allowing for flexible creation of diagnostic targets for use in 2D SIA models.

# Arguments
- `interpolation::Symbol = :None`: Specifies the interpolation method. Options include `:Linear`, `:None`.
- `n_interp_half::Int = 20`: Half-width of the interpolation stencil. Determines resolution of interpolation.
- `prescale::Union{Fin, Nothing} = nothing`: Optional prescaling function or factor applied before parametrization. Must be of type `Fin` or `nothing`.
- `postscale::Union{Fout, Nothing} = nothing`: Optional postscaling function or factor applied after parametrization. Must be of type `Fout` or `nothing`.

# Returns
- A `SIA2D_D_target{Fin, Fout}` instance, where `Fin` and `Fout` are the types of the provided `prescale` and `postscale` functions, respectively.
"""
function SIA2D_D_target(;
    interpolation::Symbol = :None,
    n_interp_half::Int = 20,
    prescale::Union{Fin, Nothing} = nothing,
    postscale::Union{Fout, Nothing} = nothing
) where {Fin <: Function, Fout <: Function}

    fin = typeof(prescale)
    fout = typeof(postscale)

    return SIA2D_D_target{fin, fout}(
        interpolation, n_interp_half,
        prescale, postscale
    )
end

"""
    Diffusivity(target::SIA2D_D_target; H, ∇S, θ, iceflow_model, ml_model, glacier, params)

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
- `ml_model`: Machine learning model used for simulation.
- `glacier`: Glacier data.
- `params`: Model parameters.

# Returns
- A matrix of diffusivity values with the same shape as `H` or staggered by one cell, depending on `U`.

# Throws
- An error if the dimensions of `U` and `H` are not compatible for diffusivity calculation.

# Notes
Uses `predict_U_matrix` internally to obtain velocity-like terms. Supports both grid-matched 
and staggered configurations by averaging `H` where necessary.
"""
function Diffusivity(
    target::SIA2D_D_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    U = predict_U_matrix(
        target;
        H = H, ∇S = ∇S, θ = θ,
        iceflow_model = iceflow_model, ml_model = ml_model,
        glacier = glacier, params = params
        )
    if size(U) == size(H)
        return H .* U
    elseif (size(U) .+ 1) == size(H)
        return Huginn.avg(H) .* U
    else
        @error "Not matching dimensions between U (∇S) and H."
    end
end

"""
    Diffusivity_scalar(target::SIA2D_D_target; h, ∇s, θ, iceflow_model, ml_model, glacier, params)

Scalar version if Diffusivity().
"""
function Diffusivity_scalar(
    target::SIA2D_D_target;
    h, ∇s, θ, iceflow_model, ml_model, glacier, params
    )
    u = predict_U_scalar(
        target;
        h = h, ∇s = ∇s, θ = θ,
        iceflow_model = iceflow_model, ml_model = ml_model,
        glacier = glacier, params = params
        )
    return h * u
end

function ∂Diffusivity∂H(
    target::SIA2D_D_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )

    ### Compute value of neural network
    ∂D∂H_no_NN = predict_U_matrix(
        target;
        H = H, ∇S = ∇S, θ = θ,
        iceflow_model = iceflow_model, ml_model = ml_model,
        glacier = glacier, params = params
        )
    ∂D∂H_no_NN .= (H .> 0.0) .* ∂D∂H_no_NN

    # Derivative of the output of the NN with respect to input layer
    δH = 1e-4 .* ones(size(H))
    ∂D∂H_NN = (
        Diffusivity(
            target;
            H = H + δH, ∇S = ∇S, θ = θ,
            iceflow_model = iceflow_model, ml_model = ml_model, glacier = glacier, params = params
        )
        .-
        Diffusivity(
            target;
            H = H, ∇S = ∇S, θ = θ,
            iceflow_model = iceflow_model, ml_model = ml_model, glacier = glacier, params = params
        )
    ) ./ δH

    return ∂D∂H_no_NN + ∂D∂H_NN
end

function ∂Diffusivity∂∇H(
    target::SIA2D_D_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )

    # For now we ignore the derivative in surface slope
    δ∇H = 1e-6 .* ones(size(∇S))
    ∂D∂∇S = (
        Diffusivity(
            target;
            H = H, ∇S = ∇S + δ∇H, θ = θ,
            iceflow_model = iceflow_model, ml_model = ml_model, glacier = glacier, params = params
        )
        .-
        Diffusivity(
            target;
            H = H, ∇S = ∇S, θ = θ,
            iceflow_model = iceflow_model, ml_model = ml_model, glacier = glacier, params = params
        )
    ) ./ δ∇H

    return ∂D∂∇S
end

function ∂Diffusivity∂θ(
    target::SIA2D_D_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )

    # Extract relevant parameters specific from the target
    interpolation = target.interpolation
    n_interp_half = target.n_interp_half

    # ∂spatial = ones(size(H)...)
    ∂spatial = map(h -> h > 0.0 ? 1.0 : 0.0, H)

    ∂D∂θ = zeros(size(H)..., only(size(θ)))
    @assert size(H) == size(∇S)

    if interpolation == :None
        """
        Computes derivative at each pixel using the exact numerical value of H at each 
        point in the glacier. Slower but more precise.
        """
        for i in axes(H, 1), j in axes(H, 2)
            if H[i, j] == 0.0
                continue
            end
            ∇θ_point, = Zygote.gradient(_θ -> Diffusivity_scalar(
                target;
                h = H[i, j], ∇s = ∇S[i, j], θ = _θ,
                iceflow_model = iceflow_model, ml_model = ml_model,
                glacier = glacier, params = params
                ), θ)
            ∂D∂θ[i, j, :] .= ∂spatial[i, j] * ComponentVector2Vector(∇θ_point)
        end

    elseif interpolation == :Linear
        """
        Interpolation of the gradient as function of values of H.
        Introduces interpolation errors but it is faster and probably sufficient depending
        the decired level of precision for the gradients.
        We construct an interpolator with quantiles and equal-spaced points.
        """

        # Interpolation for H
        H_interp = create_interpolation(H; n_interp_half = n_interp_half)
        # Interpolation for ∇S
        ∇S_interp = create_interpolation(∇S; n_interp_half = n_interp_half)

        if sum(H .> 0.0) < 2.0 * length(H_interp) * length(∇S_interp)
            @warn "The total number of AD evaluations using interpolations is comparable to the total number of AD operations required to compute the derivative purely with AD with no interpolation. Recomendation is to switch to interpolation = :None"
        end

        # Compute exact gradient in certain values of H and ∇S
        grads = [zeros(only(size(θ))) for i = 1:length(H_interp), j = 1:length(∇S_interp)]

        # TODO: Check if all these gradints cannot be computed at once withing Lux
        for (i, h) in enumerate(H_interp), (j, ∇s) in enumerate(∇S_interp)
            ∇θ_point, = Zygote.gradient(_θ -> Diffusivity_scalar(
                target;
                h = h, ∇s = ∇s, θ = _θ,
                iceflow_model = iceflow_model, ml_model = ml_model, glacier = glacier, params = params
                ), θ)
            grads[i, j] .= ComponentVector2Vector(∇θ_point)
        end
        # Create interpolation for gradient
        grad_itp = interpolate((H_interp, ∇S_interp), grads, Gridded(Linear()))

        # Compute spatial distributed gradient
        for i in axes(H, 1), j in axes(H, 2)
            ∂D∂θ[i, j, :] .= ∂spatial[i, j] * grad_itp(H[i, j], ∇S[i, j])
        end
    else
        @error "Method to spatially compute gradient with respect to H not specified."
    end

    return ∂D∂θ
end

"""
This returns the results without H, which corresponds to U
"""

function predict_U_matrix(
    target::SIA2D_D_target;
    H::Matrix{F}, ∇S::Union{Nothing, Matrix{F}}, θ,
    iceflow_model, ml_model, glacier, params
) where {F <: AbstractFloat}

    if isnothing(∇S)
        # TODO: Move all this code to function
        S = glacier.B .+ H
        dSdx = Huginn.diff_x(S) / glacier.Δx
        dSdy = Huginn.diff_y(S) / glacier.Δy
        ∇Sx = Huginn.avg_y(dSdx)
        ∇Sy = Huginn.avg_x(dSdy)
        # Compute slope in dual grid
        ∇S = (∇Sx.^2 .+ ∇Sy.^2).^(1/2)
        # Compute H in dual grid
        H = Huginn.avg(H)
    end

    @assert size(H) == size(∇S)
    res = zero(H)

    for i in axes(res, 1), j in axes(res, 2)
         res[i, j] = predict_U_scalar(
            target;
            h = H[i, j], ∇s = ∇S[i, j], θ = θ,
            iceflow_model = iceflow_model, ml_model = ml_model,
            glacier = glacier, params = params
        )
    end
    return res
end

"""

"""
function predict_U_scalar(
    target::SIA2D_D_target;
    h::F, ∇s::F, θ,
    iceflow_model, ml_model, glacier, params
) where {F <: AbstractFloat}

    nn_model = ml_model.architecture
    st = ml_model.st
    smodel = StatefulLuxLayer{true}(nn_model, θ.θ, st)

    # Pre and post scalling functions of the model
    prescale = isnothing(target.prescale) ? _ml_model_prescale : target.prescale
    postscale = isnothing(target.postscale) ? _ml_model_postscale : target.postscale

    U_pred = only(
        postscale(
            target,
            smodel(prescale(target, [h, ∇s], params)),
            params
        )
    )
    return U_pred
end

"""

"""
function apply_parametrization!(
    target::SIA2D_D_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    D = Diffusivity(
        target;
        H = H, ∇S = ∇S, θ = θ,
        iceflow_model = iceflow_model, ml_model = ml_model,
        glacier = glacier, params = params
        )
    iceflow_model.D_is_provided = true
    iceflow_model.D = D
    return nothing
end


function _ml_model_prescale(
    target::SIA2D_D_target,
    X::Vector,
    params
)
    return [
        normalize(X[1]; lims = (0.0, 300.0)),
        normalize(X[2]; lims = (0.0, 0.5))
        ]
end

function _ml_model_postscale(
    target::SIA2D_D_target,
    Y::Vector,
    params
)
    # max_NN = isnothing(target.max_NN) ? params.physical.maxA : target.max_NN
    # This shouuld correspond to maximum of Umax * dSdx
    max_NN = 50.0
    return max_NN .* exp.((Y .- 1.0) ./ Y)
end