export AbstractTarget, AbstractSIA2DTarget
export SIA2D_target
export ComponentVector2Vector, Vector2ComponentVector

abstract type AbstractTarget end
abstract type AbstractSIA2DTarget <: AbstractTarget end

struct SIA2D_target <: AbstractSIA2DTarget
    name::Symbol
    D::Function
    ∂D∂H::Function
    ∂D∂∇H::Function
    ∂D∂θ::Function
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
        (; H, ∇S, θ, model, glacier, params) -> 1.0,
        (; H, ∇S, θ, model, glacier, params) -> 1.0,
        (; H, ∇S, θ, model, glacier, params) -> 1.0,
        (; H, ∇S, θ, model, glacier, params) -> 1.0
    )
end

### Targrt to inverse creep coefficient A as a function of other quantities

function build_target_A()
    return SIA2D_target(
        :A,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> D_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂H_target_A(; H, ∇S, ice_model, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂∇H_target_A(; H, ∇S, ice_model, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂θ_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params)
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
    A = grad_apply_UDE_parametrization(H, ∇S, θ, ice_model, ml_model, params, glacier)
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
    ∇θ, = Zygote.gradient(_θ -> grad_apply_UDE_parametrization(H, ∇S, _θ, ice_model, ml_model, params, glacier), θ)
    ∇θ_cv = ComponentVector2Vector(∇θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, ∇θ_cv)
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
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂θ_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    )
end

function A_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    A_space = grad_apply_UDE_parametrization(H, ∇S, θ, ice_model, ml_model, params, glacier)
    return A_space
end

function D_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)
    A_space = A_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    return A_space .* Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)
end

function ∂D∂H_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)
    A_space = A_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    ∂D∂H_no_NN = (n[] + 2) .* A_space .* Γ_no_A .* H.^(n[] + 1) .* ∇S.^(n[] - 1)

    # Compute finite difference derivative for the forward diff needed to compute
    # Derivative of the output of the NN with respect to input layer
    # TODO: Change this to be done with AD or have this as an extra parameter.
    δH = ones(size(H))
    ∇H_NN = (
        grad_apply_UDE_parametrization(H + δH, ∇S, θ, ice_model, ml_model, params, glacier)
        .- grad_apply_UDE_parametrization(H, ∇S, θ, ice_model, ml_model, params, glacier)
    ) ./ δH
    ∂D∂H_NN = ∇H_NN .* Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)
    return ∂D∂H_no_NN + ∂D∂H_NN
end

function ∂D∂∇H_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    A_space = A_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    return Γ(ice_model, params; include_A = false) .* A_space .* (ice_model.n[] - 1) .* H.^(ice_model.n[] + 2) .* ∇S.^(ice_model.n[] - 3)
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

    # mode = "full"
    mode = "interp"

    if mode == "full"
        # TODO: This is too slow and unefficient
        # This step will make all much slower unless we do some trick, like interpolating some
        # values of H instead of numerically differentiating each single H in the glacier.
        for i in axes(H, 1), j in axes(H, 2)
            ∇θ_point, = Zygote.gradient(_θ -> only(normalize_A(StatefulLuxLayer{true}(nn_model, _θ.θ, st)([temp, H[i,j]]), lims)) , θ)
            ∂D∂θ[i, j, :] .= ∂A_spatial[i, j] * ComponentVector2Vector(∇θ_point)
        end
    elseif mode == "interp"
        n_interp = 100
        H_interp = LinRange(0.0, maximum(H), n_interp) |> collect
        # Compute exact gradient in certain values of H
        grads = []
        # TODO: Check if all these gradints cannot be computed at once withing Lux
        for h in H_interp
            ∇θ_point, = Zygote.gradient(_θ -> only(normalize_A(StatefulLuxLayer{true}(nn_model, _θ.θ, st)([temp, h]), lims)) , θ)
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

### This should not be here!

function grad_apply_UDE_parametrization(H, ∇S, θ, ice_model, ml_model, params, glacier) where {I <: Integer, SIM <: Simulation}
    # We load the ML model with the parameters
    nn_model = ml_model.architecture
    st = ml_model.st
    smodel = StatefulLuxLayer{true}(nn_model, θ.θ, st)

    # We generate the ML parametrization based on the target
    if ml_model.target.name == :A
        min_NN = params.physical.minA
        max_NN = params.physical.maxA
        A = predict_A̅(smodel, [mean(glacier.climate.longterm_temps)], (min_NN, max_NN))[1]
        return A
    elseif ml_model.target.name == :D
        min_NN = params.physical.minA
        max_NN = params.physical.maxA
        # Predict value of A based on Temp and H
        T_mean = mean(glacier.climate.longterm_temps)
        A_space = predict_D(smodel, T_mean, H, (min_NN, max_NN))
        return A_space
    else
        @error "UDE parametrization not defined for Gradient calculation."
    end
end

function predict_D(U, temp::F, H::Matrix{F}, lims::Tuple{F, F}) where {F <: AbstractFloat}
    return map(h -> only(normalize_A(U([temp, h]), lims)), H)
    # return only(normalize_A(U(temp), lims))
end