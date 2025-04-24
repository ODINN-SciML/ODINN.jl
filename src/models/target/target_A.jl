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
    ice_model.D_is_provided = false
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
    return only(scale(U(temp), lims))
end
