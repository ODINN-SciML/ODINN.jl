export AbstractTarget, AbstractSIA2DTarget
export SIA2D_target
export ComponentVector2Vector, Vector2ComponentVector
export predict_A̅

abstract type AbstractTarget end
abstract type AbstractSIA2DTarget <: AbstractTarget end

struct SIA2D_target <: AbstractSIA2DTarget
    name::Symbol
    D::Function
    ∂D∂H::Function
    ∂D∂∇H::Function
    ∂D∂θ::Function
    apply_parametrization::Function
    apply_parametrization!::Function
end

function SIA2D_target(;
    name::Symbol = :A,
)
    if name == :foo
        build_target_foo()
    elseif name == :A
        build_target_A()
    elseif name == :D
        build_target_D()
    else
        @error "Target method named $(name) not implemented."
    end
end

### Dummy target for testing

function build_target_foo()
    return SIA2D_target(
        :foo,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> nothing
    )
end

### Targrt to inverse creep coefficient A as a function of other quantities

function build_target_A()
    return SIA2D_target(
        :A,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> D_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂H_target_A(; H, ∇S, ice_model, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂∇H_target_A(; H, ∇S, ice_model, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂θ_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> apply_parametrization_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> apply_parametrization_target_A!(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    )
end

function Γ(model, params; include_A::Bool = true)
    n = model.n
    ρ = params.physical.ρ
    g = params.physical.g
    if include_A
        A = model.A
        return 2.0 * A[] * (ρ * g)^n[] / (n[]+2)
    else
        return 2.0 * (ρ * g)^n[] / (n[]+2)
    end
end

function D_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)
    # A = grad_apply_UDE_parametrization(H, ∇S, θ, ice_model, ml_model, params, glacier)
    A = apply_parametrization_target_A(;
        H = H, ∇S = ∇S, θ = θ,
        ice_model = ice_model, ml_model = ml_model, glacier = glacier, params = params
    )
    return A .* Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)
end

function ∂D∂H_target_A(; H, ∇S, ice_model, params)
    return Γ(ice_model, params) .* (ice_model.n[] + 2) .* H.^(ice_model.n[] + 1) .* ∇S.^(ice_model.n[] - 1)
end

function ∂D∂∇H_target_A(; H, ∇S, ice_model, params)
    return Γ(ice_model, params) .* (ice_model.n[] - 1) .* H.^(ice_model.n[] + 2) .* ∇S.^(ice_model.n[] - 3)
end

function ∂D∂θ_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)

    # Unfortunatelly, we need to vectorize ∇θ to do the inner product
    ∇θ, = Zygote.gradient(_θ -> apply_parametrization_target_A(;
        H = H, ∇S = ∇S, θ = _θ,
        ice_model = ice_model, ml_model = ml_model, glacier = glacier, params = params),
        θ)
    ∇θ_cv = ComponentVector2Vector(∇θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, ∇θ_cv)
end

function apply_parametrization_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params)
# function apply_parametrization_target_A(H, ∇S, θ, ice_model, ml_model, params, glacier) where {I <: Integer, SIM <: Simulation}
    # We load the ML model with the parameters
    nn_model = ml_model.architecture
    st = ml_model.st
    smodel = StatefulLuxLayer{true}(nn_model, θ.θ, st)
    min_NN = params.physical.minA
    max_NN = params.physical.maxA
    A = predict_A̅(smodel, [mean(glacier.climate.longterm_temps)], (min_NN, max_NN))[1]
    return A
end

function apply_parametrization_target_A!(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    A = apply_parametrization_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    ice_model.A[] = A
    ice_model.D = nothing
    return nothing
end

"""
    predict_A̅(U, temp, lims::Tuple{F, F}) where {F <: AbstractFloat}

Predicts the value of A with a neural network based on the long-term air temperature
and on the bounds value to normalize the output of the neural network.

# Arguments
- `U`: Neural network.
- `temp`: Temperature to be fed as an input of the neural network.
- `lims::Tuple{F, F}`: Bounds to use for the affine transformation of the neural
    network output.
"""
function predict_A̅(U, temp, lims::Tuple{F, F}) where {F <: AbstractFloat}
    return only(normalize_A(U(temp), lims))
end

"""
    normalize_A(x, lims::Tuple{F, F}) where {F <: AbstractFloat}

Normalize a variable by using an affine transformation defined by some lower and
upper bounds (m, M). The returned value is m+(M-m)*x.

# Arguments
- `x`: Input value.
- `lims::Tuple{F, F}`: Lower and upper bounds to use in the affine transformation.

# Returns
- The input variable scaled by the affine transformation.
"""
function normalize_A(x, lims::Tuple{F, F}) where {F <: AbstractFloat}
    minA_out = lims[1]
    maxA_out = lims[2]
    return minA_out .+ (maxA_out - minA_out) .* x
end

### Target to invert D as a function of H and Temp

"""
    build_target_D()

Inversion of the form

    D(H, ∇S, θ) = 2 / (n + 2) * (ρg)^n H^{n+2} |∇S|^{n-1} * NeuralNet(T, H; θ)
    D(H, ∇S, θ) = 2 / (n + 2) * (ρg)^n H^{n+2} |∇S|^{n-1} * exp( NeuralNet(T, H; θ))
    log D = log D_physics + NN
"""
function build_target_D()
    return SIA2D_target(
        :D,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> D_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂H_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂∇H_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂θ_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params),
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

function ∂D∂θ_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)

    nn_model = ml_model.architecture
    st = ml_model.st
    # smodel = StatefulLuxLayer{true}(nn_model, θ.θ, st)
    min_NN = params.physical.minA
    max_NN = params.physical.maxA
    lims = (min_NN, max_NN)
    temp = mean(glacier.climate.longterm_temps)

    ∂D∂θ = zeros(size(H)..., only(size(θ)))

    # TODO: move this to be some parameter
    mode = "full"
    # mode = "interp"

    if mode == "full"
        # Computes derivative at each pixel. Slower but more precise.
        for i in axes(H, 1), j in axes(H, 2)
            ∇θ_point, = Zygote.gradient(_θ -> only(normalize_A(StatefulLuxLayer{true}(
                nn_model,
                _θ.θ,
                st)([
                    normalize_T(temp; lims = (-25.0, 0.0))
                    normalize_H(H[i, j]; lims = (0.0, 500.0))
                ]), lims)) , θ)
            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * ComponentVector2Vector(∇θ_point)
        end
    elseif mode == "interp"
        # Interpolation of the gradient as function of values of H.
        # Introduces interpolation errors but it is faster and probably sufficient depending
        # the decired level of precision for the gradients.
        n_interp_half = 75
        # We construct an interpolator with quantiles and equal-spaced points
        H_interp_unif = LinRange(0.0, maximum(H), n_interp_half) |> collect
        H_interp_quantiles = quantile!(H, LinRange(0.0, 1.0, n_interp_half))
        H_interp = sort(vcat[H_interp_unif, H_interp_quantiles])

        # Compute exact gradient in certain values of H
        grads = []
        # TODO: Check if all these gradints cannot be computed at once withing Lux
        for h in H_interp
            # ∇θ_point, = Zygote.gradient(_θ -> only(normalize_A(StatefulLuxLayer{true}(nn_model, _θ.θ, st)([temp, h]), lims)) , θ)
            ∇θ_point, = Zygote.gradient(_θ -> only(normalize_A(StatefulLuxLayer{true}(
                nn_model,
                _θ.θ,
                st)([
                    normalize_T(temp; lims = (-25.0, 0.0))
                    normalize_H(h; lims = (0.0, 500.0))
                ]), lims)) , θ)
            push!(grads, ComponentVector2Vector(∇θ_point))
        end
        # Create interpolation for gradient
        grad_itp = interpolate((H_interp,), grads, Gridded(Linear()))

        # Compute spatial distributed gradient
        for i in axes(H, 1), j in axes(H, 2)
            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * grad_itp(H[i,j])
        end
    else
        @error "Method to spatially compute gradient with respect to H not specified."
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
    ice_model.D = D
    return nothing
end

function _apply_parametrization_A_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    # We load the ML model with the parameters
    nn_model = ml_model.architecture
    st = ml_model.st
    smodel = StatefulLuxLayer{true}(nn_model, θ.θ, st)
    min_NN = params.physical.minA
    max_NN = params.physical.maxA
    # Predict value of A based on Temp and H
    T_mean = mean(glacier.climate.longterm_temps)
    A_space = predict_A_target_D(smodel, T_mean, H, (min_NN, max_NN))
    return A_space
end

function predict_A_target_D(U, temp::F, H::Matrix{F}, lims::Tuple{F, F}) where {F <: AbstractFloat}
    return map(h -> only(
        normalize_A(
            U([
                normalize_T(temp; lims = (-25.0, 0.0)),
                normalize_H(h; lims = (0.0, 500.0))
            ]),
            lims)
        ),
        H)
end

# Normalization functions to ensure the scales of input are comparable to each other
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
