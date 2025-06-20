export SIA2D_D_hybrid_target

"""
    SIA2D_D_hybrid_target{Fin, Fout} <: AbstractSIA2DTarget

Struct to define inversion where elements of the diffusivity D in the SIA equations are
replaced by a generic regressor. For this example, we consider the inversion of the form

    D(H, ∇S, θ) = ( C * (ρ * g)^n + 2 * H * NeuralNet(T, H, ∇S; θ) / (n + 2) * (ρg)^n ) H^{n+1} |∇S|^{n-1}
"""

@kwdef struct SIA2D_D_hybrid_target <: AbstractSIA2DTarget
    interpolation::Symbol = :Linear
    n_interp_half::Int = 75
    n_H::Union{Float64, Nothing} = nothing
    n_∇S::Union{Float64, Nothing} = nothing
    min_NN::Union{Float64, Nothing} = nothing
    max_NN::Union{Float64, Nothing} = nothing
    prescale_provided::Bool = false
    postscale_provided::Bool = false
end

# For this simple case, the target coincides with D, but not always.
# TODO: D should be cap to its maximum physical value. This can be done with one extra
# function and one extra differentiation.
function Diffusivity(
    target::SIA2D_D_hybrid_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return apply_parametrization(
        target;
        H, ∇S, θ, iceflow_model, ml_model, glacier, params
        )
end

function ∂Diffusivity∂H(
    target::SIA2D_D_hybrid_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )

    # Allow different n for power in inversion of diffusivity
    # TODO: n is also inside Γ, so probably we want to grab this one too
    n = iceflow_model.n
    C = iceflow_model.C
    ρ = params.physical.ρ
    g = params.physical.g
    n_H = isnothing(target.n_H) ? n[] : target.n_H
    n_∇S = isnothing(target.n_∇S) ? n[] : target.n_∇S

    Γ_no_A = Γ(iceflow_model, params; include_A = false)
    A = apply_parametrization_A(target; H, ∇S, θ, iceflow_model, ml_model, glacier, params)
    ∂D∂H_no_NN = ( (n_H + 1) .* C .* (ρ * g).^n .+ (n_H + 2) .* A .* Γ_no_A .* H ) .* H.^n_H .* ∇S.^(n_∇S - 1)

    # Derivative of the output of the NN with respect to input layer
    # TODO: Change this to be done with AD or have this as an extra parameter.
    # This is done already in SphereUDE.jl with Lux
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
    target::SIA2D_D_hybrid_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )

    n = iceflow_model.n
    C = iceflow_model.C
    ρ = params.physical.ρ
    g = params.physical.g
    n_H = isnothing(target.n_H) ? n[] : target.n_H
    n_∇S = isnothing(target.n_∇S) ? n[] : target.n_∇S

    A = apply_parametrization_A(target; H, ∇S, θ, iceflow_model, ml_model, glacier, params)
    ∂D∂∇S_no_NN = (C .* (ρ * g).^n .+ Γ(iceflow_model, params; include_A = false) .* A .* H) .* (n_∇S - 1) .* H.^(n_H + 1) .* ∇S.^(n_∇S - 3)

    return ∂D∂∇S_no_NN
end

function ∂Diffusivity∂θ(
    target::SIA2D_D_hybrid_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )

    # Extract relevant parameters specific from the target
    interpolation = target.interpolation
    n_interp_half = target.n_interp_half

    n = iceflow_model.n
    n_H = isnothing(target.n_H) ? n[] : target.n_H
    n_∇S = isnothing(target.n_∇S) ? n[] : target.n_∇S

    Γ_no_A = Γ(iceflow_model, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H.^(n_H + 2) .* ∇S.^(n_∇S - 1)

    temp = mean(glacier.climate.longterm_temps)

    ∂D∂θ = zeros(size(H)..., only(size(θ)))

    if interpolation == :None
        """
        Computes derivative at each pixel using the exact numerical value of H at each 
        point in the glacier. Slower but more precise.
        """
        for i in axes(H, 1), j in axes(H, 2)
            ∇θ_point, = Zygote.gradient(_θ -> predict_A(
                target,
                _θ, temp, H[i,j];
                ml_model = ml_model, params = params
                ), θ)
            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * ComponentVector2Vector(∇θ_point)
        end
    elseif interpolation == :Linear
        """
        Interpolation of the gradient as function of values of H.
        Introduces interpolation errors but it is faster and probably sufficient depending
        the decired level of precision for the gradients.
        We construct an interpolator with quantiles and equal-spaced points.
        """
        H_interp = create_interpolation(H; n_interp_half = n_interp_half)

        # Compute exact gradient in certain values of H
        grads = []
        # TODO: Check if all these gradints cannot be computed at once withing Lux
        for h in H_interp
            ∇θ_point, = Zygote.gradient(_θ -> predict_A(
                target,
                _θ, temp, h;
                ml_model = ml_model, params = params
                ), θ)
            push!(grads, ComponentVector2Vector(∇θ_point))
        end
        # Create interpolation for gradient
        grad_itp = interpolate((H_interp,), grads, Gridded(Linear()))

        # Compute spatial distributed gradient
        for i in axes(H, 1), j in axes(H, 2)
            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * grad_itp(H[i, j])
        end
    else
        @error "Method to spatially compute gradient with respect to H not specified."
    end

    return ∂D∂θ
end

function apply_parametrization(
    target::SIA2D_D_hybrid_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )

    n = iceflow_model.n
    C = iceflow_model.C
    ρ = params.physical.ρ
    g = params.physical.g
    n_H = isnothing(target.n_H) ? n[] : target.n_H
    n_∇S = isnothing(target.n_∇S) ? n[] : target.n_∇S

    Γ_no_A = Γ(iceflow_model, params; include_A = false)

    # Compute ∇S in case is not provided.
    # In this case, the matrix H will have a larger size, so we overwrite it.
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

    # # Predict value of A based on Temp and H
    A = apply_parametrization_A(target; H, ∇S, θ, iceflow_model, ml_model, glacier, params)
    D = (C .* (ρ * g).^n .+ A .* Γ_no_A .* H) .* H.^(n_H + 1) .* ∇S.^(n_∇S - 1)

    return D
end

function apply_parametrization!(
    target::SIA2D_D_hybrid_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    D = apply_parametrization(
        target;
        H, ∇S, θ, iceflow_model, ml_model, glacier, params
        )
    iceflow_model.D_is_provided = true
    iceflow_model.D = D
    return nothing
end

### Auxiliary functions

function apply_parametrization_A(
    target::SIA2D_D_hybrid_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    T_mean = mean(glacier.climate.longterm_temps)
    A_space = predict_A(
        target,
        θ, T_mean, H;
        ml_model = ml_model, params = params
    )
    return A_space
end

function predict_A(
    target::SIA2D_D_hybrid_target,
    θ,
    temp::F,
    H::Matrix{F};
    ml_model,
    params
) where {F <: AbstractFloat}
    return map(h -> predict_A(
        target,
        θ, temp, h;
        ml_model = ml_model, params = params
        ), H)
end

function predict_A(
    target::SIA2D_D_hybrid_target,
    θ,
    temp::F,
    h::F;
    ml_model,
    params
) where {F <: AbstractFloat}

    # We load the ML model with the parameters
    nn_model = ml_model.architecture
    st = ml_model.st
    smodel = StatefulLuxLayer{true}(nn_model, θ.θ, st)

    # Pre and post scalling functions of the model
    prescale = target.prescale_provided ? identity : X -> _ml_model_prescale(target, X, params)
    postscale = target.postscale_provided ? identity : Y -> _ml_model_postscale(target, Y, params)

    A_pred = only(
        postscale(
            smodel(prescale([temp, h])),
        )
    )
    return A_pred
end

function _ml_model_prescale(
    target::SIA2D_D_hybrid_target,
    X::Vector,
    params
)
    return [
        normalize(X[1]; lims = (-25.0, 0.0)),
        normalize(X[2]; lims = (0.0, 500.0))
        ]
end

function _ml_model_postscale(
    target::SIA2D_D_hybrid_target,
    Y::Vector,
    params
)
    max_NN = isnothing(target.max_NN) ? params.physical.maxA : target.max_NN
    return max_NN .* exp.((Y .- 1.0) ./ Y)
end
