export ComponentVector2Vector, Vector2ComponentVector

function Γ(model, model_cache, params; include_A::Bool = true)
    n = model_cache.n
    (; ρ, g) = params.physical
    if include_A
        A = model_cache.A
        return 2.0 .* A .* (ρ * g).^n ./ (n.+2)
    else
        return 2.0 .* (ρ * g).^n ./ (n.+2)
    end
end

function S(model, model_cache, params)
    (; C, n) = model_cache
    (; ρ, g) = params.physical
    return C .* (ρ * g).^n
end

function Γꜛ(model, model_cache, params; include_A::Bool = true)
    n = model_cache.n
    (; ρ, g) = params.physical
    if include_A
        A = model_cache.A
        return 2.0 .* A .* (ρ * g).^n ./ (n.+1)
    else
        return 2.0 .* (ρ * g).^n ./ (n.+1)
    end
end

function Sꜛ(model, model_cache, params)
    (; C, n) = model_cache
    (; ρ, g) = params.physical
    return (n.+2) .* C .* (ρ * g).^n
end

"""
    _ml_model_prescale(
        X::Vector,
        prescale_bounds::Vector{Tuple{F, F}},
    ) where {F <: AbstractFloat}

Scales each element of the input vector `X` using the corresponding bounds from `prescale_bounds`.
For each index `i`, `X[i]` is normalized based on the interval specified in `prescale_bounds[i]`
using the `normalize` function.
This function is typically used to ensure that the scales of the inputs of a neural network are
comparable to each other.

# Arguments
- `X::Vector`: A vector of input values to be normalized.
- `prescale_bounds::Vector{Tuple{F, F}}`: A vector of tuples specifying the lower and upper bounds
    for normalization of each corresponding element in `X`.

# Returns
- A vector where each element is the normalized value of the corresponding input, using the
    specified bounds.

# Notes
- The length of `X` and `prescale_bounds` must be equal.
"""
function _ml_model_prescale(
    X::Vector,
    prescale_bounds::Vector{Tuple{F, F}},
) where {F <: AbstractFloat}
    @assert length(X)==length(prescale_bounds)
    return [
        normalize(X[i]; lims=prescale_bounds[i])
        for i in 1:length(X)
    ]
end

"""
    _ml_model_postscale(
        Y::Vector,
        max_NN,
    )

Applies an exponential transformation to each element in `Y`, then rescales the
result by multiplying with `max_NN`.
For each element, the transformation is: `max_NN * exp((Y - 1.0) / Y)`

# Arguments
- `Y::Vector`: Values to be post-processed.
- `max_NN`: Scalar representing the maximum value for rescaling.

# Returns
- The rescaled values after applying the exponential transformation.
"""
function _ml_model_postscale(
    Y::Vector,
    max_NN,
)
    return max_NN .* exp.((Y .- 1.0) ./ Y)
end

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

"""
    Vector2ComponentVector(v::Vector, cv_template::ComponentVector)

Transform a vector `v` to a `ComponentVector` that has the same structure as `cv_template`.
This function creates a new `ComponentVector` and copies the values of `v` explicitly.
The arguments `v` and `cv_template` must be of the same length.

Arguments:
- `v::Vector`: Vector whose values are copied.
- `cv_template::ComponentVector`: ComponentVector whose structure is used to create a new `ComponentVector`.
"""
function Vector2ComponentVector(v::Vector, cv_template::ComponentVector)
    cv = zero(cv_template)
    vec(cv) .= vec(v)
    return cv
end

"""
    ComponentVector2Vector(cv::ComponentVector)

Transform a `ComponentVector` into a `Vector` of same length.
This function creates a new `Vector` and does not mutate the original `ComponentVector`.

Arguments:
- `cv::ComponentVector`: Input `ComponentVector`.
"""
function ComponentVector2Vector(cv::ComponentVector)
    return collect(cv)
end

"""
    function create_interpolation(A::Matrix; n_interp_half::Int)

Function to create an intepolation for AD computation combining uniform and quantiles.
"""
function create_interpolation(A::Matrix; n_interp_half::Int)
    A_interp_unif = LinRange(0.0, maximum(A), n_interp_half) |> collect
    A_interp_quantiles = quantile!(A[A .> 0.0], LinRange(0.0, 1.0, n_interp_half))
    A_interp = vcat(A_interp_unif, A_interp_quantiles)
    A_interp = unique(A_interp)
    A_interp = sort(A_interp)
    return A_interp
end