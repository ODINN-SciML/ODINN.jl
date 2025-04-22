### Target to invert D as a function of H and Temp

"""
    build_target_D()

Inversion of the form

    D(H, ∇S, θ) = 2 / (n + 2) * (ρg)^n H^{n+2} |∇S|^{n-1} * NeuralNet(T, H; θ)
    D(H, ∇S, θ) = 2 / (n + 2) * (ρg)^n H^{n+2} |∇S|^{n-1} * exp( NeuralNet(T, H; θ))
    log D = log D_physics + NN
"""
function build_target_D(;
    interpolation::Bool = true
)
    return SIA2D_target(
        :D,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> D_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂H_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂∇H_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂θ_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params, interpolation),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> apply_parametrization_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> apply_parametrization_target_D!(; H, ∇S, θ, ice_model, ml_model, glacier, params),
    )
end

# For this simple case, the target coincides with D, but not always.
function D_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    return apply_parametrization_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
end

function ∂D∂H_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)
    A = _apply_parametrization_A_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    ∂D∂H_no_NN = (n[] + 2) .* A .* Γ_no_A .* H.^(n[] + 1) .* ∇S.^(n[] - 1)

    # Derivative of the output of the NN with respect to input layer
    # TODO: Change this to be done with AD or have this as an extra parameter.
    # This is done already in SphereUDE.jl with Lux
    δH = 1e-4 .* ones(size(H))
    ∂D∂H_NN = (
        D_target_D(;
            H = H + δH, ∇S = ∇S, θ = θ,
            ice_model = ice_model, ml_model = ml_model, glacier = glacier, params = params
        )
        .-
        D_target_D(;
            H = H, ∇S = ∇S, θ = θ,
            ice_model = ice_model, ml_model = ml_model, glacier = glacier, params = params
        )
    ) ./ δH
    return ∂D∂H_no_NN + ∂D∂H_NN
end

function ∂D∂∇H_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    A = _apply_parametrization_A_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    return Γ(ice_model, params; include_A = false) .* A .* (ice_model.n[] - 1) .* H.^(ice_model.n[] + 2) .* ∇S.^(ice_model.n[] - 3)
end

function ∂D∂θ_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params, interpolation)
    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)

    temp = mean(glacier.climate.longterm_temps)

    ∂D∂θ = zeros(size(H)..., only(size(θ)))

    # TODO: move this to be some parameter
    # mode = "full"
    # mode = "interp"
    # if mode == "full"

    if !interpolation
        # Computes derivative at each pixel. Slower but more precise.
        for i in axes(H, 1), j in axes(H, 2)
            ∇θ_point, = Zygote.gradient(_θ -> predict_A_target_D(_θ, temp, H[i,j]; ml_model = ml_model, params = params), θ)
            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * ComponentVector2Vector(∇θ_point)
        end
    else
    # elseif mode == "interp"
        # Interpolation of the gradient as function of values of H.
        # Introduces interpolation errors but it is faster and probably sufficient depending
        # the decired level of precision for the gradients.
        n_interp_half = 75
        # We construct an interpolator with quantiles and equal-spaced points
        H_interp_unif = LinRange(0.0, maximum(H), n_interp_half) |> collect
        H_interp_quantiles = quantile!(H[H .> 0.0], LinRange(0.0, 1.0, n_interp_half))
        H_interp = vcat(H_interp_unif, H_interp_quantiles)
        H_interp = unique(H_interp)
        H_interp = sort(H_interp)

        # Compute exact gradient in certain values of H
        grads = []
        # TODO: Check if all these gradints cannot be computed at once withing Lux
        for h in H_interp
            ∇θ_point, = Zygote.gradient(_θ -> predict_A_target_D(_θ, temp, h; ml_model = ml_model, params = params), θ)
            push!(grads, ComponentVector2Vector(∇θ_point))
        end
        # Create interpolation for gradient
        grad_itp = interpolate((H_interp,), grads, Gridded(Linear()))

        # Compute spatial distributed gradient
        for i in axes(H, 1), j in axes(H, 2)
            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * grad_itp(H[i, j])
        end
    # else
    #     @error "Method to spatially compute gradient with respect to H not specified."
    end

    return ∂D∂θ
end

function apply_parametrization_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)

    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)

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
    A = _apply_parametrization_A_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)

    # Diffusivity is always evaluated in dual grid.
    return A .* Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)
end

function apply_parametrization_target_D!(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    D = apply_parametrization_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    ice_model.D_is_provided = true
    ice_model.D = D
    return nothing
end

function _apply_parametrization_A_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    T_mean = mean(glacier.climate.longterm_temps)
    A_space = predict_A_target_D(
        θ, T_mean, H;
        ml_model = ml_model, params = params
    )
    return A_space
end

function predict_A_target_D(
    θ,
    temp::F,
    H::Matrix{F};
    ml_model,
    params
) where {F <: AbstractFloat}
    return map(h -> predict_A_target_D(θ, temp, h; ml_model = ml_model, params = params), H)
end

function predict_A_target_D(
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

    min_NN = params.physical.minA
    max_NN = params.physical.maxA

    # Neural network prediction
    A_pred = only(
        normalize_A(
            smodel([
                normalize_T(temp; lims = (-25.0, 0.0)),
                normalize_H(h; lims = (0.0, 500.0))
            ]),
            (min_NN, max_NN))
        )

    if rand() < 0.00000002
        println("Value of A used inside Target: $(A_pred).")
    end
    return A_pred
end

### Normalization functions to ensure the scales of input are comparable to each other

function normalize_T(T; lims)
    return (T .- lims[1]) ./ (lims[2] - lims[1]) .- 0.5
end

function normalize_H(H; lims)
    return (H .- lims[1]) ./ (lims[2] - lims[1]) .- 0.5
end

### General Utils

function cartesian_tensor(A, v)
    B = zeros(size(A)..., only(size(v)))
    for i in axes(A, 1), j in axes(A, 2), k in axes(v,1)
        B[i, j, k] = A[i, j] * v[k]
    end
    return B
end

function Vector2ComponentVector(v::Vector, cv_template::ComponentVector)
    cv = zero(cv_template)
    for i in 1:length(v)
        cv[i] = v[i]
    end
    return cv
end

function ComponentVector2Vector(cv::ComponentVector)
    return collect(cv)
end
