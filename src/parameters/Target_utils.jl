export AbstractTarget, AbstractSIA2DTarget
export SIA2D_target
export ComponentVector2Vector, Vector2ComponentVector

abstract type AbstractTarget end
abstract type AbstractSIA2DTarget <: AbstractTarget end

struct SIA2D_target <: AbstractSIA2DTarget
    name::String
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
        (; H, ∇S, θ, model, glacier, params) -> 1.0
    )
end

### Targrt to inverse creep coefficient A as a function of other quantities

function build_target_A()
    return SIA2D_target(
        "A",
        (; H, ∇S, θ, model, glacier, params) -> Γ(model, params) .* (model.n[] + 2) .* H.^(model.n[] + 1) .* ∇S,
        (; H, ∇S, θ, model, glacier, params) -> Γ(model, params) .* (model.n[] - 1) .* H.^(model.n[] + 2) .* ∇S.^(model.n[] - 3),
        # (; H, ∇S, θ, model, glacier, params) -> ∂D∂θ_target_A(; H, ∇S, θ, model, glacier, params)
        (; H, ∇S, θ, ∇θ, model, glacier, params) -> ∂D∂θ_target_A(; H, ∇S, θ, ∇θ, model, glacier, params)
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

function ∂D∂θ_target_A(; H, ∇S, θ, ∇θ, model, glacier, params)
    n = model.n
    ρ = params.physical.ρ
    g = params.physical.g

    Γ_no_A = Γ(model, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H.^(n[] + 2) .* ∇S

    # TODO: Find a better solution for this
    # Unfortunatelly, we need to vectoriza ∇θ to do the iterproduct
    ∇θ_cv = ComponentVector2Vector(∇θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, ∇θ_cv)
end

### Target to invert D as a function of H, ∇S, ...


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