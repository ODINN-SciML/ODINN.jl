export SIA2D_A_target

"""
    function SIA2D_target(;
       name::Symbol = :A,
    )

Target to inverse creep coefficient A as a function of other quantities
Constructor of the SIA target. All the relevant functions defined inside Target are
constructed automatically by just providing the keyword `name` for the inversion.

# Arguments
- `name::Symbol`: Identifying name for the model inversion.
"""

@kwdef struct SIA2D_A_target <: AbstractSIA2DTarget
end

### Target functions

function Diffusivity(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    n = iceflow_model.n
    Γ_no_A = Γ(iceflow_model, params; include_A = false)
    A = apply_parametrization(
        target;
        H = H, ∇S = ∇S, θ = θ,
        iceflow_model = iceflow_model, ml_model = ml_model, glacier = glacier, params = params
    )
    return A .* Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)
end

function ∂Diffusivity∂H(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return Γ(iceflow_model, params) .* (iceflow_model.n[] + 2) .* H.^(iceflow_model.n[] + 1) .* ∇S.^(iceflow_model.n[] - 1)
end

function ∂Diffusivity∂∇H(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return Γ(iceflow_model, params) .* (iceflow_model.n[] - 1) .* H.^(iceflow_model.n[] + 2) .* ∇S.^(iceflow_model.n[] - 3)
end

function ∂Diffusivity∂θ(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    n = iceflow_model.n
    Γ_no_A = Γ(iceflow_model, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H.^(n[] + 2) .* ∇S.^(n[] - 1)

    # Unfortunatelly, we need to vectorize ∇θ to do the inner product
    ∇θ, = Zygote.gradient(_θ -> apply_parametrization(
        target;
        H = H, ∇S = ∇S, θ = _θ,
        iceflow_model = iceflow_model, ml_model = ml_model, glacier = glacier, params = params),
        θ)
    ∇θ_cv = ComponentVector2Vector(∇θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, ∇θ_cv)
end

function Diffusivityꜛ(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
)
    n = iceflow_model.n
    Γ_no_A = Γꜛ(iceflow_model, params; include_A = false)
    A = apply_parametrization(
        target;
        H = H, ∇S = ∇S, θ = θ,
        iceflow_model = iceflow_model, ml_model = ml_model, glacier = glacier, params = params
    )
    return (A .* Γ_no_A) .* H.^(n[] + 1) .* ∇S.^(n[] - 1)
end

function ∂Diffusivityꜛ∂H(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return Γꜛ(iceflow_model, params) .* (iceflow_model.n[] + 1) .* H.^(iceflow_model.n[]) .* ∇S.^(iceflow_model.n[] - 1)
end

function ∂Diffusivityꜛ∂∇H(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return Γꜛ(iceflow_model, params) .* (iceflow_model.n[] - 1) .* H.^(iceflow_model.n[] + 1) .* ∇S.^(iceflow_model.n[] - 3)
end

function ∂Diffusivityꜛ∂θ(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    n = iceflow_model.n
    Γ_no_A = Γꜛ(iceflow_model, params; include_A = false)
    ∂A_spatial = Γ_no_A .* H.^(n[] + 1) .* ∇S.^(n[] - 1)

    # Unfortunatelly, we need to vectorize ∇θ to do the inner product
    ∇θ, = Zygote.gradient(_θ -> apply_parametrization(
        target;
        H = H, ∇S = ∇S, θ = _θ,
        iceflow_model = iceflow_model, ml_model = ml_model, glacier = glacier, params = params),
        θ)
    ∇θ_cv = ComponentVector2Vector(∇θ)

    # Create a tensor with both elements
    return cartesian_tensor(∂A_spatial, ∇θ_cv)
end

function apply_parametrization(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    # We load the ML model with the parameters
    nn_model = ml_model.architecture
    st = ml_model.st
    smodel = StatefulLuxLayer{true}(nn_model, θ.θ, st)
    min_NN = params.physical.minA
    max_NN = params.physical.maxA
    A = predict_A̅(smodel, [mean(glacier.climate.longterm_temps)], (min_NN, max_NN))[1]
    return A
end

function apply_parametrization!(
    target::SIA2D_A_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    A = apply_parametrization(target; H, ∇S, θ, iceflow_model, ml_model, glacier, params)
    iceflow_model.A[] = A
    iceflow_model.D = nothing
    iceflow_model.D_is_provided = false
    return nothing
end

### Auxiliary functions

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
function Γꜛ(model, params; include_A::Bool = true)
    n = model.n
    ρ = params.physical.ρ
    g = params.physical.g
    if include_A
        A = model.A
        return 2.0 * A[] * (ρ * g)^n[] / (n[]+1)
    else
        return 2.0 * (ρ * g)^n[] / (n[]+1)
    end
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
function predict_A̅(
    U, temp, lims::Tuple{F, F}
    ) where {F <: AbstractFloat}
    return only(scale(U(temp), lims))
end
