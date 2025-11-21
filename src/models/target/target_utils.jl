export ComponentVector2Vector, Vector2ComponentVector

function Γ(model, model_cache, params; include_A::Bool = true)
    n = model_cache.n.value
    (; ρ, g) = params.physical
    if include_A
        A = model_cache.A.value
        return 2.0 .* A .* (ρ * g).^n ./ (n.+2)
    else
        return 2.0 .* (ρ * g).^n ./ (n.+2)
    end
end

function S(model, model_cache, params)
    (; C, p, q) = model_cache
    (; ρ, g) = params.physical
    return C.value .* (ρ * g).^(p.value - q.value)
end

function Γꜛ(model, model_cache, params; include_A::Bool = true)
    n = model_cache.n.value
    (; ρ, g) = params.physical
    if include_A
        A = model_cache.A.value
        return 2.0 .* A .* (ρ * g).^n ./ (n.+1)
    else
        return 2.0 .* (ρ * g).^n ./ (n.+1)
    end
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
    lims::Union{Tuple{F, F}, Tuple{Vector{F}, Vector{F}}},
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

function cartesian_tensor(A::Matrix{F}, v::Vector{F}) where {F <: AbstractFloat}
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
    create_interpolation(
        A::Vector;
        n_interp_half::Int,
        dilation_factor = 1.0,
        minA_unif::Union{F, Nothing} = nothing,
        minA_quantile::Union{F, Nothing} = nothing,
        maxA_unif::Union{F, Nothing} = nothing,
        maxA_quantile::Union{F, Nothing} = nothing
        ) where {F <: AbstractFloat}

Construct a one-dimensional interpolation grid from the data in `A`, combining
uniformly spaced and quantile-based sampling points.

This hybrid interpolation grid provides both coverage of the entire range
of values and higher resolution in regions where `A` has dense data,
making it useful for interpolation or machine learning applications
that need balanced sampling.

# Arguments
- `A::Vector`: Input data vector (typically containing positive values).
- `n_interp_half::Int`: Number of points used for both the uniform and quantile-based
  subsets of the interpolation grid.
- `dilation_factor::Real = 1.0`: Optional multiplier applied to `maximum(A)` to slightly
  extend the grid beyond the data range (useful to avoid extrapolation issues).
- `minA_unif::Union{F, Nothing} = nothing`: Minimum value used for the uniform interpolation
- `minA_quantile::Union{F, Nothing} = nothing`: Maximum value used for the uniform interpolation
- `maxA_unif::Union{F, Nothing} = nothing`: Minimum value used for the quantile interpolation
- `maxA_quantile::Union{F, Nothing} = nothing`: Maximum value used for the quantile interpolation

# Returns
A sorted, unique vector of interpolation nodes combining:
- `n_interp_half` uniformly spaced values between `0` and `dilation_factor * maximum(A)`
- `n_interp_half` quantile-based values computed from the positive entries of `A`
"""
function create_interpolation(
    A::Vector;
    n_interp_half::Int,
    dilation_factor = 1.0,
    minA_unif::Union{F, Nothing} = nothing,
    minA_quantile::Union{F, Nothing} = nothing,
    maxA_unif::Union{F, Nothing} = nothing,
    maxA_quantile::Union{F, Nothing} = nothing
    ) where {F <: AbstractFloat}

    # Assign ranges for both uniform and quantile interpolation
    minA_unif = isnothing(minA_unif) ? 0.0 : minA_unif
    minA_quantile = isnothing(minA_quantile) ? 0.0 : minA_quantile
    maxA_unif = isnothing(maxA_unif) ? dilation_factor * maximum(A) : maxA_unif
    maxA_quantile = isnothing(maxA_quantile) ? maximum(A) : maxA_quantile

    @assert (minA_unif < maxA_unif) && (minA_quantile < maxA_quantile) "There are not enough different values of A to create a proper interpolation."

    # Construct uniform interpolation
    A_interp_unif = LinRange(minA_unif, maxA_unif, n_interp_half) |> collect

    # Construct quantile interpolation
    quantile_range = LinRange(0.0, 1.0, n_interp_half + 2)[begin + 1: end - 1]
    A_interp_quantiles = quantile!(A[(minA_quantile .< A) .&& (A .< maxA_quantile)], quantile_range)
    A_interp = unique(vcat(A_interp_unif, A_interp_quantiles))

    A_interp = unique(A_interp)
    # Order values in increasing order
    A_interp = sort(A_interp)

    # If some some reason some values between uniform interpolation and quantile are
    # repeated, we add more values to the interpolation to have the total desired number
    # of knots in the interpolation.
    # In theory, this should never happen, but just in case we include this fix.
    if length(A_interp) < 2 * n_interp_half
        n_left = 2 * n_interp_half - length(A_interp)
        rand_idx = sample(1:(length(A_interp) - 1), n_left, replace = false)
        # The middle point between two elements of A_interp is by definition never included
        # in the original interpolation array
        A_left = [(A_interp[i] + A_interp[i + 1]) / 2 for i in rand_idx]
        A_interp = vcat(A_interp, A_left)
        A_interp = sort(A_interp)
    end
    # This assert is a bit redundant based on the previous code, but we keep it for safety.
    @assert length(A_interp) == 2 * n_interp_half "The number of interpolation points is different than twice `n_interp_half`"

    return A_interp
end

"""
    create_interpolation(A::Matrix; n_interp_half::Int) -> Vector{Float64}

Construct a one-dimensional interpolation grid from the elements of a matrix `A` by
flattening it and delegating to [`create_interpolation(::Vector)`](@ref). This is a 
convenience method that allows users to pass a 2D array to the function `create_interpolation(A::Vector)`
directly without manually reshaping it.
"""
function create_interpolation(
    A::Matrix;
    n_interp_half::Int,
    dilation_factor = 1.0,
    minA_unif::Union{F, Nothing} = nothing,
    minA_quantile::Union{F, Nothing} = nothing,
    maxA_unif::Union{F, Nothing} = nothing,
    maxA_quantile::Union{F, Nothing} = nothing,
    ) where {F <: AbstractFloat}
    create_interpolation(
        vec(A);
        n_interp_half = n_interp_half,
        dilation_factor = dilation_factor,
        minA_unif = minA_unif,
        minA_quantile = minA_quantile,
        maxA_unif = maxA_unif,
        maxA_quantile = maxA_quantile,
        )
end
