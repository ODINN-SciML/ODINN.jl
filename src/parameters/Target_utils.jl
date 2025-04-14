export AbstractTarget, AbstractSIA2DTarget
export SIA2D_target
export ComponentVector2Vector, Vector2ComponentVector

abstract type AbstractTarget end
abstract type AbstractSIA2DTarget <: AbstractTarget end

struct SIA2D_target <: AbstractSIA2DTarget
    name::String
    D::Function
    ∂D∂H::Function
    ∂D∂∇H::Function
    ∂D∂θ::Function
end

function SIA2D_target(;
    name::String = "A",
)
    if name == "Foo"
        build_target_foo()
    elseif name == "A"
        build_target_A()
    elseif name == "D"
        build_target_D()
    else
        @error "Target method named $(name) not implemented."
    end
end

### Dummy target for testing

function build_target_foo()
    return SIA2D_target(
        "Foo",
        (; H, ∇S, θ, model, glacier, params) -> 1.0,
        (; H, ∇S, θ, model, glacier, params) -> 1.0,
        (; H, ∇S, θ, model, glacier, params) -> 1.0,
        (; H, ∇S, θ, model, glacier, params) -> 1.0
    )
end

### Targrt to inverse creep coefficient A as a function of other quantities

function build_target_A()
    return SIA2D_target(
        "A",
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
    A = grad_apply_UDE_parametrization(θ, ml_model, params, glacier)
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
    ∇θ, = Zygote.gradient(_θ -> grad_apply_UDE_parametrization(_θ, ml_model, params, glacier), θ)
    ∇θ_cv = ComponentVector2Vector(∇θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, ∇θ_cv)
end

### Target to invert D as a function of H and Temp

function build_target_D()
    return SIA2D_target(
        "D",
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> D_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂H_target_D(; H, ∇S, ice_model, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂∇H_target_D(; H, ∇S, ice_model, params),
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> ∂D∂θ_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    )
end

function D_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)
    A = grad_apply_UDE_parametrization(θ, ml_model, params, glacier)
    return A .* Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)
end

function ∂D∂H_target_D(; H, ∇S, ice_model, params)
    return Γ(ice_model, params) .* (ice_model.n[] + 2) .* H.^(ice_model.n[] + 1) .* ∇S.^(ice_model.n[] - 1)
end

function ∂D∂∇H_target_D(; H, ∇S, ice_model, params)
    return Γ(ice_model, params) .* (ice_model.n[] - 1) .* H.^(ice_model.n[] + 2) .* ∇S.^(ice_model.n[] - 3)
end

function ∂D∂θ_target_D(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    n = ice_model.n
    Γ_no_A = Γ(ice_model, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)

    # Unfortunatelly, we need to vectorize ∇θ to do the inner product
    ∇θ, = Zygote.gradient(_θ -> grad_apply_UDE_parametrization(_θ, ml_model, params, glacier), θ)
    ∇θ_cv = ComponentVector2Vector(∇θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, ∇θ_cv)
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

function grad_apply_UDE_parametrization(θ, ml_model, params, glacier) where {I <: Integer, SIM <: Simulation}
    # We load the ML model with the parameters
    nn_model = ml_model.architecture
    st = ml_model.st
    smodel = StatefulLuxLayer{true}(nn_model, θ.θ, st)

    # We generate the ML parametrization based on the target
    if params.UDE.target.name == "A"
        min_NN = params.physical.minA
        max_NN = params.physical.maxA
        A = predict_A̅(smodel, [mean(glacier.climate.longterm_temps)], (min_NN, max_NN))[1]
        return A
    end
end