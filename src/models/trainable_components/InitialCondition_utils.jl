
"""
    evaluate_H₀(
        θ::ComponentArray,
        glacier::Glacier2D,
        filter::Symbol,
        glacier_id::Integer,
    )

Evaluate the initial ice thickness `H₀` for a given glacier, optionally applying a smooth thresholding function.

# Arguments

  - `θ::ComponentArray`: A `ComponentArray` containing glacier parameters.

  - `glacier::Glacier2D`: Glacier for which to evaluate `H₀`.

  - `filter::Symbol`: Specifies the smoothing function to apply to the raw initial condition:

      + `:identity`: applies the identity function (no change).
      + `:softplus`: applies the softplus function `log(1 + exp(x))` to ensure positivity.
      + `:Zang1980`: applies the `σ_zang` function (Zang 1980) as a smooth positivity threshold.

  - `glacier_id::Integer`: Index of the glacier in order to retrieve the parameters of the IC in θ.

# Returns

  - A numeric value or array representing the filtered initial ice thickness for the specified glacier.
"""
function evaluate_H₀(
        θ::ComponentArray,
        glacier::Glacier2D,
        filter::Symbol,
        glacier_id::Integer
)
    glacier_id_symbol = Symbol("$(glacier_id)")
    H₀ = deepcopy(θ.IC[glacier_id_symbol])
    H₀ = @match filter begin
        :identity => H₀
        :softplus => log.(1 .+ exp.(H₀))
        :Zang1980 => σ_zang.(H₀)
    end
    # Apply mask. Written as a product rather than a masked write because
    # `InitialThicknessRegularization` calls this inside the loss, so it has to be
    # differentiable, and Zygote refuses the `copyto!` that indexing with the mask creates.
    # `mask` is true on the no-ice cells, so the complement is what survives.
    return H₀ .* .!glacier.mask
end

"""
    evaluate_∂H₀(
        θ::ComponentArray,
        glacier::Glacier2D,
        filter::Symbol,
        glacier_id::Integer,
    )

Evaluate the derivative of the initial ice thickness `H₀` for a given glacier, optionally applying a smooth thresholding function.

# Arguments

  - `θ::ComponentArray`: A `ComponentArray` containing glacier parameters.

  - `glacier::Glacier2D`: Glacier for which to evaluate `∂H₀`.

  - `filter::Symbol`: Specifies the smoothing function to apply to the raw initial condition:

      + `:identity`: applies the identity function (no change).
      + `:softplus` — applies the softplus function `log(1 + exp(x))` to ensure positivity.
      + `:Zang1980` — applies the `σ_zang` function (Zang 1980) as a smooth positivity threshold.

  - `glacier_id::Integer`: Index of the glacier in order to retrieve the parameters of the IC in θ.

# Returns

  - A numeric value or array representing the filtered initial ice thickness for the specified glacier.
"""
function evaluate_∂H₀(
        θ::ComponentArray,
        glacier::Glacier2D,
        filter::Symbol,
        glacier_id::Integer
)
    glacier_id_symbol = Symbol("$(glacier_id)")
    ∂H₀ = deepcopy(θ.IC[glacier_id_symbol])
    ∂H₀ = @match filter begin
        :identity => 1.0
        :softplus => 1 ./ (1 .+ exp.(-∂H₀))
        :Zang1980 => ∂σ_zang.(∂H₀)
    end
    # Apply mask
    ∂H₀[glacier.mask] .= 0.0
    return ∂H₀
end

"""
    σ_zang(x; β = 2.0)

Smooth thresholding function for enforcing non-negativity and zero values for negative
values following I. Zang, "A smoothing-out technique for min—max optimization" (1980).

# Arguments

  - `x::Real`: Input value to be thresholded.
  - `β::Real`: (optional) Parameter controlling the transition zone width. Default is 2.0.
"""
function σ_zang(x; β = 2.0)
    # Every branch must return the same type. The original returned the literal `0.0` below
    # the threshold and `x` unchanged above it, which agree only when `x` is already a
    # Float64. Under Zygote's broadcast the elements are `ForwardDiff.Dual`, so the two
    # branches give `Float64` and `Dual`, `promote_typejoin` widens the result to
    # `Matrix{Real}`, and that abstract state reaches the solver through `evaluate_H₀` and
    # `define_iceflow_prob`.
    #
    # It needs elements on *both* sides of the threshold at once, so it stays hidden until
    # the optimizer moves some θ.IC across `-β/2` mid-training. The consequences are bad in
    # two different ways: with a stabilized solver, OrdinaryDiffEq derives its cache types
    # from the state's bottom eltype, `one(Real)` is `1::Int64`, and ROCK2 builds its float
    # tableau as `Int64[...]` -- `InexactError: Int64(0.410...)` hundreds of frames away;
    # without one, Zygote simply returns *no gradient* for θ.IC and the initial condition
    # silently stops training.
    #
    # `float(x)` up front commits every branch to one float type and keeps the function
    # usable with Dual and tracked numbers.
    fx = float(x)
    βf = oftype(fx, β)
    if fx < -βf / 2
        return zero(fx)
    elseif fx < βf / 2
        return (fx + βf / 2)^2 / (2βf)
    else
        return fx
    end
end

"""
    ∂σ_zang(x; β = 2.0)

Derivative of the smooth thresholding function `σ_zang`.

# Arguments

  - `x::Real`: Input value to be thresholded.
  - `β::Real`: (optional) Parameter controlling the transition zone width. Default is 2.0.
"""
function ∂σ_zang(x; β = 2.0)
    if x < - β / 2
        return 0.0
    elseif x < β / 2
        return x / β + 0.5
    else
        return 1.0
    end
end

"""
    random_matrix(mean, std, corr_length)

Generate a random matrix with entries drawn from a multivariate normal distribution
whose mean is given by `mean` and whose covariance decays exponentially with
grid distance.

# Arguments

  - `mean::AbstractMatrix{<:Real}`: Matrix specifying the spatial mean values at each grid point.
    Entries equal to `0.0` are treated as inactive and skipped in sampling.
  - `std::Real`: Standard deviation scaling factor for the covariance kernel.
  - `corr_length::Real`: Correlation length parameter controlling how fast correlations
    decay with Euclidean distance between grid points.

# Returns

  - `H_sample::Matrix{Float64}`: A random realization of the same size as `mean`, with correlated entries
    drawn from `MvNormal(mean, Σ)`, where
    `Σ[i,j] = std * exp(-‖coords[i] - coords[j]‖ / corr_length)`.
"""
function random_matrix(mean, std, corr_length)
    nx, ny = size(mean)
    N = nx * ny
    # Sample coordinates
    μ = [mean[i, j] for i in 1:nx for j in 1:ny]
    coords = [(i, j) for i in 1:nx for j in 1:ny]
    Σ = zeros(N, N)
    for i in 1:N
        for j in 1:N
            d = norm(coords[i] .- coords[j])
            Σ[i, j] = std * exp(- d / corr_length)
        end
    end
    # Sample from multivariate normal
    mvnorm = MvNormal(μ, Σ)
    sample = rand(mvnorm)
    H_sample = zeros(nx, ny)
    for (k, (i, j)) in enumerate(coords)
        if mean[i, j] == 0.0
            continue
        end
        H_sample[i, j] = sample[k]
    end
    return H_sample
end
