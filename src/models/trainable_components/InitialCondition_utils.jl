
"""
    H₀_scale(glacier)

Default scale of `θ.IC[id].H` for each glacier, in metres (the maximum of `glacier.H₀`). It is
stored as `H₀ / scale`, so its values are of order one like the other parameters, and the
same learning rate works for all of them. If `glacier.H₀` changes after `InitialCondition`
is built, θ.IC no longer matches. The filters (`σ_zang`, `softplus`) act on metres.
"""
H₀_scale(glacier) = max(maximum(glacier.H₀), one(eltype(glacier.H₀)))

"""
    ic_params(θ, glacier_id)

Parameters of the initial condition of a glacier, `θ.IC[glacier_id]`, with the fields `H` and,
when the bed is free, `S`. It is a view: writing into it changes `θ`. The key is type stable,
see [`glacier_key`](@ref).
"""
ic_params(θ, glacier_id) = getproperty(θ.IC, glacier_key(θ.IC, glacier_id))

# Whether θ inverts the bed. All glaciers have the same fields, so the first one tells and the
# answer does not need a lookup by glacier index (it runs inside the differentiated functions)
function bed_is_free(θ, glacier_id)
    (isnothing(θ) || !haskey(θ, :IC)) && return false
    ic = θ.IC
    first_glacier = ic isa ComponentVector ? getproperty(ic, Val(first(keys(ic)))) :
                    first(values(ic))
    return haskey(first_glacier, :S)
end

"""
    ic_scales(simulation, glacier_idx)

Scales of `θ.IC[id].H` and `θ.IC[id].S` in metres, and the minimum thickness inside the outline,
as the named tuple `(; H, S, Hmin)`, from the `InitialCondition` of the model, or the defaults.
"""
function ic_scales(simulation, glacier_idx)
    glacier = simulation.glaciers[glacier_idx]
    IC = simulation.model.trainable_components.IC
    F = eltype(glacier.H₀)
    IC isa InitialCondition || return (H = H₀_scale(glacier), S = F(5), Hmin = F(1))
    # `scale_H = 0` is the default: the maximum thickness of the glacier of this simulation,
    # which is the one of the batch when the glaciers are split
    return (H = IC.scale_H > 0 ? IC.scale_H : H₀_scale(glacier), S = IC.scale_S,
        Hmin = IC.min_thickness)
end
# The scales are numbers read from the model and the glacier: Enzyme, which differentiates the
# bed inside the ice flow, does not have to compile this function
EnzymeCore.EnzymeRules.inactive_noinl(::typeof(ic_scales), args...) = nothing

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

  - `scale`: Scale of `θ.IC[id].H` in metres, see [`ic_scales`](@ref).

  - `H_min`: Minimum of the thickness in metres inside the outline of the glacier, with the
    `:softplus` and `:Zang1980` filters (`Hmin + filter(x - Hmin)`). Default is `0.0`.

The outline is the set of cells where `glacier.H₀` is not zero. Outside it the thickness is zero.

# Returns

  - A numeric value or array representing the filtered initial ice thickness for the specified glacier.
"""
function evaluate_H₀(
        θ::ComponentArray,
        glacier::Glacier2D,
        filter::Symbol,
        glacier_id::Integer;
        scale = H₀_scale(glacier),
        H_min = zero(eltype(glacier.H₀))
)
    # The product allocates a new array, so `θ` is never aliased (no `deepcopy`, which the
    # Enzyme-differentiated bed hook would have to go through)
    H₀ = ic_params(θ, glacier_id).H .* scale
    H₀ = @match filter begin
        :identity => H₀
        :softplus => softplus_min.(H₀, H_min)
        :Zang1980 => σ_zang_min.(H₀, H_min)
    end
    # Apply the outline. Written as a product rather than a masked write because
    # `InitialThicknessRegularization` calls this inside the loss, so it has to be
    # differentiable, and Zygote refuses the `copyto!` that indexing with the mask creates.
    return H₀ .* (glacier.H₀ .> 0)
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

  - `scale`, `H_min`: As in [`evaluate_H₀`](@ref).

# Returns

  - A numeric value or array representing the filtered initial ice thickness for the specified glacier.
"""
function evaluate_∂H₀(
        θ::ComponentArray,
        glacier::Glacier2D,
        filter::Symbol,
        glacier_id::Integer;
        scale = H₀_scale(glacier),
        H_min = zero(eltype(glacier.H₀))
)
    ∂H₀ = ic_params(θ, glacier_id).H .* scale
    ∂H₀ = @match filter begin
        # An array, not the scalar 1.0: the mask below indexes into it
        :identity => one.(∂H₀)
        :softplus => 1 ./ (1 .+ exp.(-(∂H₀ .- H_min)))
        :Zang1980 => ∂σ_zang.(∂H₀ .- H_min)
    end
    # d(filter(θ.IC * scale))/dθ.IC needs the extra `scale` factor from the chain rule.
    ∂H₀ = ∂H₀ .* scale
    # Outside the outline the thickness is zero
    ∂H₀[.!(glacier.H₀ .> 0)] .= 0.0
    return ∂H₀
end

"""
    softplus_stable(x)

`log(1 + exp(x))` that does not overflow: above 30 m it is `x`.
"""
softplus_stable(x) = x > 30 ? x : log1p(exp(min(x, 30)))

"""
    σ_zang_min(x, Hmin)
    softplus_min(x, Hmin)

The filter with a minimum value, `Hmin + filter(x - Hmin)`: it is smooth, it is `x` for
`x` well above `Hmin`, and it never goes below `Hmin`. For `Hmin = 0` it is the filter itself.
"""
σ_zang_min(x, Hmin) = Hmin + σ_zang(x - Hmin)
softplus_min(x, Hmin) = Hmin + softplus_stable(x - Hmin)

"""
    evaluate_B(θ, simulation, glacier_idx)

Bed of a glacier: `S₀ - H₀` when the bed is free (see [`InitialCondition`](@ref)), and
`glacier.B` otherwise. It is the same at all times.

The ice flow calls it at every right-hand side, so it is one fused loop that allocates only
the result. On the cells where ice is allowed it is `glacier.S + S × scale_S - H₀`, with
`H₀ = filter(H × scale_H)` inside the outline of the glacier (`glacier.H₀ > 0`) and `H₀ = 0`
outside it, the same operations as `S₀ - initial_thickness` with `S₀ = glacier.S + S × scale_S`.
It is `glacier.S` on the other cells.
"""
function evaluate_B(θ, simulation, glacier_idx)
    glacier = simulation.glaciers[glacier_idx]
    bed_is_free(θ, glacier_idx) || return glacier.B
    scales = @ignore_derivatives ic_scales(simulation, glacier_idx)
    filter = simulation.parameters.UDE.initial_condition_filter
    p = ic_params(θ, glacier_idx)
    S, mask, Hout, sS,
    sH, Hmin = glacier.S, glacier.mask, glacier.H₀, scales.S, scales.H, scales.Hmin
    if filter === :Zang1980
        return @. ifelse(mask, S,
            S + p.S * sS - ifelse(Hout > 0, σ_zang_min(p.H * sH, Hmin), zero(Hout)))
    elseif filter === :softplus
        return @. ifelse(mask, S,
            S + p.S * sS - ifelse(Hout > 0, softplus_min(p.H * sH, Hmin), zero(Hout)))
    elseif filter === :identity
        return @. ifelse(mask, S, S + p.S * sS - ifelse(Hout > 0, p.H * sH, zero(Hout)))
    end
    throw(ArgumentError(
        "initial_condition_filter must be :identity, :softplus or :Zang1980, got :$(filter)."))
end

"""
    initial_thickness(θ, simulation, glacier_idx)

Initial ice thickness of a glacier in metres: [`evaluate_H₀`](@ref) with the filter of the
parameters and the scale of the `InitialCondition`.
"""
function initial_thickness(θ, simulation, glacier_idx)
    scales = @ignore_derivatives ic_scales(simulation, glacier_idx)
    return evaluate_H₀(θ, simulation.glaciers[glacier_idx],
        simulation.parameters.UDE.initial_condition_filter, glacier_idx;
        scale = scales.H, H_min = scales.Hmin)
end

# The ice flow model and the losses use this bed
Huginn.bed(simulation::Inversion, glacier_idx, θ) = evaluate_B(θ, simulation, glacier_idx)

"""
    σ_zang(x; β = 2.0)

Smooth thresholding function for enforcing non-negativity and zero values for negative
values following I. Zang, "A smoothing-out technique for min—max optimization" (1980).

# Arguments

  - `x::Real`: Input value to be thresholded.
  - `β::Real`: (optional) Parameter controlling the transition zone width. Default is 2.0.
"""
function σ_zang(x; β = 2.0)
    # All the branches must return the same float type, also for Dual numbers under Zygote.
    # Otherwise the state becomes a `Matrix{Real}`, ROCK2 fails with an `InexactError` and
    # Zygote gives no gradient for θ.IC.
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
        return zero(x)
    elseif x < β / 2
        return x / β + 0.5
    else
        return one(x)
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
