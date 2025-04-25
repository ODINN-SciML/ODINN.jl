export ComponentVector2Vector, Vector2ComponentVector
export predict_A̅

### Dummy target for testing

# function build_target_foo()
#     fD = (; H, ∇S, θ, iceflow_model, ml_model, glacier, params) -> 1.0
#     f∂D∂H = (; H, ∇S, θ, iceflow_model, ml_model, glacier, params) -> 1.0
#     f∂D∂∇H = (; H, ∇S, θ, iceflow_model, ml_model, glacier, params) -> 1.0
#     f∂D∂θ = (; H, ∇S, θ, iceflow_model, ml_model, glacier, params) -> 1.0
#     fP = (; H, ∇S, θ, iceflow_model, ml_model, glacier, params) -> 1.0
#     fP! = (; H, ∇S, θ, iceflow_model, ml_model, glacier, params) -> nothing

#     return SIA2D_target{
#         typeof(fD), typeof(f∂D∂H), typeof(f∂D∂∇H), typeof(f∂D∂θ), typeof(fP), typeof(fP!)
#         }(
#         :foo, fD, f∂D∂H, f∂D∂∇H, f∂D∂θ, fP, fP!
#     )
# end
"""
Create foo target for testing
"""
@kwdef struct SIA2D_foo_target <: AbstractSIA2DTarget
end

function Diffusivity(
    Target::SIA2D_foo_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return 1.0
end

function ∂Diffusivity∂H(
    Target::SIA2D_foo_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return 1.0
end

function ∂Diffusivity∂∇H(
    Target::SIA2D_foo_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return 1.0
end

function ∂Diffusivity∂θ(
    Target::SIA2D_foo_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return 1.0
end

function apply_parametrization(
    Target::SIA2D_foo_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return 1.0
end

function apply_parametrization!(
    Target::SIA2D_foo_target;
    H, ∇S, θ, iceflow_model, ml_model, glacier, params
    )
    return nothing
end

### Normalization function to ensure the scales of input are comparable to each other

# function normalize_T(T; lims)
#     return (T .- lims[1]) ./ (lims[2] - lims[1]) .- 0.5
# end

# function normalize_H(H; lims)
#     return (H .- lims[1]) ./ (lims[2] - lims[1]) .- 0.5
# end

"""
    scale(x, lims::Tuple{F, F}) where {F <: AbstractFloat}

Scale a variable by using an affine transformation defined by some lower and
upper bounds (m, M). The returned value is m+(M-m)*x.

# Arguments
- `X`: Input value.
- `lims::Tuple{F, F}`: Lower and upper bounds to use in the affine transformation.

# Returns
- The input variable scaled by the affine transformation.
"""

function scale(X, lims::Tuple{F, F}) where {F <: AbstractFloat}
    min_value = lims[1]
    max_value = lims[2]
    return min_value .+ (max_value - min_value) .* X
end

"""
    normalize(X; lims::Tuple{F, F}; method = :shift) where {F <: AbstractFloat}

Normalize a variable by using an affine transformation defined by some input lower and
upper bounds (m, M) and transforming to O(1) scale.

# Arguments
- `X`: Input value.
- `lims::Tuple{F, F}`: Lower and upper bounds to use in the affine transformation.
- `method::Symbol`: Method to scale data.

# Returns
- The input variable scaled by the affine transformation.
"""
function normalize(
    X;
    lims::Tuple{F, F},
    method::Symbol = :shift
) where {F <: AbstractFloat}

    if method == :shift
        return (X .- lims[1]) ./ (lims[2] - lims[1]) .- 0.5
    else
        throw("Normalization method not implemented.")
    end
end

"""
Normalization of D to cap at a maximum physical value
"""
function cap_D(D; maxD = 1.0)
    return maxD .* tanh(D ./ maxD)
end

function ∂cap_D(D; maxD = 1.0)
    return sech.(D ./ maxD)^2
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