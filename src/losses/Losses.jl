export L2Sum, LogSum
export LossH, LossV, LossHV
export loss, backward_loss
import Sleipnir: GeneralAbstractLoss, AbstractSimpleLoss, AbstractLoss, velocityProduct

"""
    L2Sum{I <: Integer} <: AbstractSimpleLoss

Struct that defines an L2 loss: the mean over the pixels of the mask of the squared misfit,
divided by the uncertainty `σ` of the data,

``loss(a,b) = \\frac{1}{N} \\sum_{i\\in\\text{mask}} \\left(\\frac{a_i - b_i}{σ_i}\\right)^2``

where `N` is the number of pixels in the mask. The loss is about 1 when the misfit is as large
as the uncertainty, whatever the resolution and the number of observed pixels. `σ` is a number
or a matrix, given by the data loss that uses `L2Sum` (for example [`LossH`](@ref)), and is 1
otherwise.
The parameter `distance` controls the pixels that should be used.
In order for a pixel to be used, it should be at least at `distance` from the glacier border.
The mask defining the glacier borders are computed using the ground truth ice thickness.

# Fields

  - `distance::I`: Distance to border.
"""
@kwdef struct L2Sum{I <: Integer} <: AbstractSimpleLoss
    distance::I = 3
end

"""
    LogSum{I <: Integer, F <: AbstractFloat} <: AbstractSimpleLoss

Struct that defines a logarithmic loss, the mean over the pixels of the mask of

``\\left(\\frac{\\log((a + ϵ) / (b + ϵ))}{σ}\\right)^2``

The misfit is relative, so `σ` is a relative uncertainty, and the `σ` of the data loss (in the
units of the data) is not used.

# Fields

  - `distance::I`: Distance to border.
  - `ϵ::F`: Epsilon used inside the loss function to handle division by zero and log(0).
    It somehow represents the minimum value the loss function should be sensible to.
  - `σ::F`: Relative uncertainty of the data, for example `0.1` for 10%. Default is `1.0`.
"""
@kwdef struct LogSum{I <: Integer, F <: AbstractFloat} <: AbstractSimpleLoss
    distance::I = 3
    ϵ::F = 0.1
    σ::F = 1.0
end

# Misfit divided by the uncertainty
residual(lossType::L2Sum, a, b, σ) = (a .- b) ./ σ
function residual(lossType::LogSum, a, b, σ)
    log.((a .+ lossType.ϵ) ./ (b .+ lossType.ϵ)) ./ lossType.σ
end
# Derivative of the residual with respect to `a`
∂residual∂a(lossType::L2Sum, a, b, σ) = one(eltype(a)) ./ σ
∂residual∂a(lossType::LogSum, a, b, σ) = 1 ./ ((a .+ lossType.ϵ) .* lossType.σ)

# Mean of the squared residuals over the pixels of `mask`, and 0 for an empty mask
masked_mean_sq(r, mask) = sum(abs2, r[mask]) / max(count(mask), 1)

"""
    LossH{L <: AbstractSimpleLoss, S <: Union{Nothing, Real}} <: AbstractLoss

Struct that defines the ice thickness loss. Each observation adds the mean over its observed
pixels of `((H - H_obs) / σ)²`, see [`L2Sum`](@ref).

# Fields

  - `loss::L`: Type of loss to use for the ice thickness. Default is `L2Sum()`.
  - `weighting::Symbol`: How each ice thickness observation is weighted in the loss, see
    [`observation_weights`](@ref). Default is `:uniform`.
  - `σ::S`: Uncertainty of the observed ice thickness, in m. With `nothing` (default), it is
    the uncertainty of each cell given by the data (`thicknessData.σ`), and 18 m if the data
    has none. A number replaces it.
"""
@kwdef struct LossH{L <: AbstractSimpleLoss, S <: Union{Nothing, Real}} <: AbstractLoss
    loss::L = L2Sum()
    weighting::Symbol = :uniform
    σ::S = nothing
end

"""
    LossV{L <: AbstractSimpleLoss, S <: Union{Nothing, Real}} <: AbstractLoss

Struct that defines the ice velocity loss. Each observation adds the mean over its observed
pixels of `((V - V_obs) / σ)²`, see [`L2Sum`](@ref).

# Fields

  - `loss::L`: Type of loss to use for the ice velocity. Default is `L2Sum()`.
  - `component::Symbol`: Component of the velocity field used in the loss.
    Options include :xy for both x and y component, and :abs for the norm/magnitude of the velocity.
    With `:xy` the loss is the sum of the terms of the two components, so a misfit of σ in both
    gives 2, and with `:abs` it gives 1.
  - `weighting::Symbol`: How each ice velocity observation is weighted in the loss, see
    [`observation_weights`](@ref). Default is `:uniform`.
  - `σ::S`: Uncertainty of the observed velocity, in m/yr. With `nothing` (default), it is the
    error of the velocity map given by the data (`velocityData.vabs_error`), and 10 m/yr if the
    data has none. A number replaces it.
"""
@kwdef struct LossV{L <: AbstractSimpleLoss, S <: Union{Nothing, Real}} <: AbstractLoss
    loss::L = L2Sum()
    component::Symbol = :xy
    weighting::Symbol = :uniform
    σ::S = nothing
end

"""
    default_σ(lossType)

Uncertainty used by a data loss when neither the loss nor the data give one.

  - `LossH`: 18 m. This is the error of a cell of 137 m measured on the Aletsch glacier with two
    independent survey campaigns. It is larger at a coarser resolution.
  - `LossV`, `LossAvgV`: 10 m/yr.
  - `LossDhdt`: 0.2 m/yr.
"""
function default_σ end

default_σ(::LossH) = 18.0
default_σ(::LossV) = 10.0

"""
    thickness_σ(lossType::LossH, glacier, t)

Uncertainty of the ice thickness observation of `glacier` at time `t`: `lossType.σ` if it is a
number, else the uncertainty per cell given by the data, else [`default_σ`](@ref). Cells
without a valid uncertainty (not observed) get 1, which has no effect since they are not used.
"""
function thickness_σ(lossType::LossH, glacier, t)
    isnothing(lossType.σ) || return lossType.σ
    data = glacier.thicknessData
    σs = isnothing(data) ? nothing : data.σ
    isnothing(σs) && return default_σ(lossType)
    σ = σs[findfirst(==(t), data.t)]
    return ifelse.(isfinite.(σ) .& (σ .> 0), σ, one(eltype(σ)))
end

"""
    velocity_σ(lossType, glacier, k)

Uncertainty of the `k`-th velocity observation of `glacier`: `lossType.σ` if it is a number,
else `velocityData.vabs_error[k]` if it is valid, else [`default_σ`](@ref).

`vabs_error` is the mean over the ice of the size of the error vector (x and y) of the velocity
map. With equal and independent errors in x and y, each component has `1/√2` of it, and so has
the speed (its error is the one along the flow), so `1/√2` of it is used for `:xy` and `:abs`.
"""
function velocity_σ(lossType, glacier, k)
    isnothing(lossType.σ) || return lossType.σ
    errors = isnothing(glacier.velocityData) ? nothing : glacier.velocityData.vabs_error
    (isnothing(errors) || isnothing(k)) && return default_σ(lossType)
    σ = errors[k]
    return isfinite(σ) && σ > 0 ? σ / sqrt(2) : default_σ(lossType)
end

"""
    LossHV{
        F <: AbstractFloat,
        LH <: AbstractLoss,
        LV <: AbstractLoss,
    } <: AbstractLoss

Struct that defines the ice thickness and ice velocity loss.
It consists of two fields that define the ice thickness and ice velocity loss.
It also has a scaling coefficient that balances the ice velocity term in the loss.

``loss(\\hat H,H) = loss_H(\\hat H,H) + scaling * loss_V(\\hat V,V)``

with ``\\hat V`` computed from ``\\hat H`` for the SIA.

# Fields

  - `hLoss::LH`: Type of loss to use for the ice thickness. Default is `LossH()`.
  - `vLoss::LV`: Type of loss to use for the ice velocity. Default is `LossV()`.
  - `scaling::F`: Scaling of the ice velocity term. Default is `1`.
"""
@kwdef struct LossHV{
    F <: AbstractFloat,
    LH <: AbstractLoss,
    LV <: AbstractLoss
} <: AbstractLoss
    hLoss::LH = LossH()
    vLoss::LV = LossV()
    scaling::F = 1.0
end

### Definition of simple loss functions

# The positional argument `normalization` is not used by the simple losses, which are a mean
# over the mask. `σ` is the uncertainty of the data, a number or a matrix.

"""
    loss(lossType::AbstractSimpleLoss, a, b, mask, normalization; σ = 1.0)

Mean over the pixels of `mask` of the squared residual of `a` with respect to the data `b`,
see [`L2Sum`](@ref) and [`LogSum`](@ref).
For the logarithmic loss, see Morlighem, M. et al., "Spatial patterns of basal drag inferred
using control methods from a full-Stokes and simpler models for Pine Island Glacier, West
Antarctica". Geophys. Res. Lett. 37, (2010). It has been shown that this loss function
enables robust estimation of drag coefficient.
"""
function loss(
        lossType::Union{L2Sum, LogSum},
        a::Matrix{F},
        b::Matrix{F},
        mask::BitMatrix,
        normalization::F;
        σ = one(F)
) where {F <: AbstractFloat}
    lossType isa LogSum && @assert (minimum(a) >= 0.0) & (minimum(b) >= 0.0)
    return masked_mean_sq(residual(lossType, a, b, σ), mask)
end
function backward_loss(
        lossType::Union{L2Sum, LogSum},
        a::Matrix{F},
        b::Matrix{F},
        mask::BitMatrix,
        normalization::F;
        σ = one(F)
) where {F <: AbstractFloat}
    r = residual(lossType, a, b, σ)
    ∂r∂a = ∂residual∂a(lossType, a, b, σ)
    return ifelse.(mask, 2 .* r .* ∂r∂a ./ max(count(mask), 1), zero(F))
end

# Without a mask: the pixels at `distance` from the border of the data for `L2Sum`, and all
# the pixels for `LogSum`
default_mask(lossType::L2Sum, b) = is_in_glacier(b, lossType.distance)
default_mask(lossType::LogSum, b) = trues(size(b))
function loss(
        lossType::Union{L2Sum, LogSum},
        a::Matrix{F},
        b::Matrix{F},
        normalization::F;
        σ = one(F)
) where {F <: AbstractFloat}
    return loss(lossType, a, b, default_mask(lossType, b), normalization; σ = σ)
end
function backward_loss(
        lossType::Union{L2Sum, LogSum},
        a::Matrix{F},
        b::Matrix{F},
        normalization::F;
        σ = one(F)
) where {F <: AbstractFloat}
    return backward_loss(lossType, a, b, default_mask(lossType, b), normalization; σ = σ)
end

function loss(
        lossType::L2Sum,
        a::Vector{Matrix{F}},
        b::Vector{Matrix{F}},
        normalization::F
) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [loss(lossType, ai, bi, normalization) for (ai, bi) in zip(a, b)]
end
function backward_loss(
        lossType::L2Sum,
        a::Vector{Matrix{F}},
        b::Vector{Matrix{F}},
        normalization::F
) where {F <: AbstractFloat}
    @assert length(a) == length(b) "Size of a and b don't match: length(a)=$(length(a)) but length(b)=$(length(b))"
    return [backward_loss(lossType, ai, bi, normalization) for (ai, bi) in zip(a, b)]
end

function loss(
        lossType::LossH,
        H_pred::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    if isnothing(H_ref)
        # That time step has no valid ground truth ice thickness data, so the contribution is zero
        return 0.0
    else
        mask = is_in_glacier(H_ref, lossType.loss.distance)
        σ = @ignore_derivatives thickness_σ(lossType, simulation.glaciers[glacier_idx], t)
        return loss(lossType.loss, H_pred, H_ref, mask, normalization; σ = σ) * Δt.H
    end
end
function backward_loss(
        lossType::LossH,
        H_pred::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    ∂L∂θ = zero(θ)
    ∂L∂H = if isnothing(H_ref)
        # That time step has no valid ground truth ice thickness data, so the contribution is zero
        zero(H_pred)
    else
        mask = is_in_glacier(H_ref, lossType.loss.distance)
        σ = thickness_σ(lossType, simulation.glaciers[glacier_idx], t)
        backward_loss(lossType.loss, H_pred, H_ref, mask, normalization; σ = σ)
    end
    return ∂L∂H * Δt.H, ∂L∂θ * Δt.H
end

function loss(
        lossType::LossV,
        H_pred::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    if isnothing(V_ref)
        # That time step has no valid ground truth ice surface velocity data, so the contribution is zero
        return 0.0
    end

    # Compute the predicted velocity Vx_pred, Vy_pred, V_pred
    if !isnothing(simulation.model.trainable_components)
        simulation.model.trainable_components.θ = θ
    end
    Vx_pred, Vy_pred, V_pred = Huginn.V_from_H(simulation, H_pred, t, θ)
    # TODO: in the future we should dispatch wrt the iceflow model

    mask = (V_ref .> 0.0)
    σ = @ignore_derivatives velocity_σ(lossType, simulation.glaciers[glacier_idx],
        velocity_index(simulation, glacier_idx, t))

    ℓ = if lossType.component == :xy
        loss(lossType.loss, Vx_pred, Vx_ref, mask, normalization; σ = σ) +
        loss(lossType.loss, Vy_pred, Vy_ref, mask, normalization; σ = σ)
    elseif lossType.component == :abs
        loss(lossType.loss, V_pred, V_ref, mask, normalization; σ = σ)
    else
        @error "Loss type not implemented."
    end

    return ℓ * Δt.V
end
# Index of the velocity observation at time `t`
function velocity_index(simulation, glacier_idx, t)
    findfirst(==(t),
        tdata(simulation.glaciers[glacier_idx].velocityData,
            simulation.parameters.simulation.mapping))
end
function backward_loss(
        lossType::LossV,
        H_pred::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    if isnothing(V_ref)
        # That time step has no valid ground truth ice surface velocity data, so the contribution is zero
        return zero(H_pred), zero(θ)
    end

    # Compute the predicted velocity Vx_pred, Vy_pred, V_pred
    if !isnothing(simulation.model.trainable_components)
        simulation.model.trainable_components.θ = θ
    end
    Vx_pred, Vy_pred, V_pred = Huginn.V_from_H(simulation, H_pred, t, θ)
    # TODO: in the future we should dispatch wrt the iceflow model

    mask = (V_ref .> 0.0)
    σ = velocity_σ(lossType, simulation.glaciers[glacier_idx],
        velocity_index(simulation, glacier_idx, t))

    if lossType.component == :xy
        ∂lV∂Vx = backward_loss(lossType.loss, Vx_pred, Vx_ref, mask, normalization; σ = σ)
        ∂lV∂Vy = backward_loss(lossType.loss, Vy_pred, Vy_ref, mask, normalization; σ = σ)
    elseif lossType.component == :abs
        ∂lV∂V = backward_loss(lossType.loss, V_pred, V_ref, mask, normalization; σ = σ)
        # Chain rule through V = √(Vx² + Vy²): ∂l/∂Vx = ∂l/∂V · Vx/V (and similarly Vy)
        ∂Vx_dir, ∂Vy_dir = VJP_λ_∂V∂Vxy(∂lV∂V, Vx_pred, Vy_pred)
        valid = mask .& (V_pred .> 0.0)
        ∂lV∂Vx = ifelse.(valid, ∂Vx_dir, 0.0)
        ∂lV∂Vy = ifelse.(valid, ∂Vy_dir, 0.0)
    end

    ∂L∂H = VJP_λ_∂surface_V∂H(simulation.parameters.UDE.grad.VJP_method,
        ∂lV∂Vx, ∂lV∂Vy, H_pred, θ, simulation, t)[1]
    ∂L∂θ = VJP_λ_∂surface_V∂θ(simulation.parameters.UDE.grad.VJP_method,
        ∂lV∂Vx, ∂lV∂Vy, H_pred, θ, simulation, t)[1]

    return ∂L∂H * Δt.V, ∂L∂θ * Δt.V
end

function loss(
        lossType::LossHV,
        H_pred::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    lH = loss(lossType.hLoss, H_pred, H_ref, V_ref, Vx_ref, Vy_ref,
        t, glacier_idx, θ, simulation, normalization, Δt)
    lV = loss(lossType.vLoss, H_pred, H_ref, V_ref, Vx_ref, Vy_ref,
        t, glacier_idx, θ, simulation, normalization, Δt)
    # The observation weights are already applied by each loss
    return lH + lossType.scaling * lV
end
function backward_loss(
        lossType::LossHV,
        H_pred::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    ∂lH∂H,
    ∂lH∂θ = backward_loss(lossType.hLoss, H_pred, H_ref, V_ref, Vx_ref, Vy_ref,
        t, glacier_idx, θ, simulation, normalization, Δt)
    ∂lV∂H,
    ∂lV∂θ = backward_loss(lossType.vLoss, H_pred, H_ref, V_ref, Vx_ref, Vy_ref,
        t, glacier_idx, θ, simulation, normalization, Δt)
    # The observation weights are already applied by each loss
    ∂L∂H = isnothing(∂lV∂H) ? ∂lH∂H : ∂lH∂H + lossType.scaling * ∂lV∂H
    ∂L∂θ = if isnothing(∂lV∂θ)
        ∂lH∂θ
    elseif isnothing(∂lH∂θ)
        lossType.scaling * ∂lV∂θ
    else
        ∂lH∂θ + lossType.scaling * ∂lV∂θ
    end
    return ∂L∂H, ∂L∂θ
end

loss_uses_velocity(lossType::LossH) = false
loss_uses_velocity(lossType::Union{LossV, LossHV}) = true
discreteLossSteps(lossType::AbstractLoss, tspan) = Vector{Float64}()

"""
    observation_weighting(lossType, var::Symbol)

Weighting of the observations of `var` (`:H` or `:V`) chosen in the loss, or `nothing` if
the loss doesn't use these observations.
"""
observation_weighting(lossType::AbstractLoss, var::Symbol) = nothing
function observation_weighting(lossType::LossH, var::Symbol)
    var == :H ? lossType.weighting : nothing
end
function observation_weighting(lossType::LossV, var::Symbol)
    var == :V ? lossType.weighting : nothing
end
function observation_weighting(lossType::LossHV, var::Symbol)
    w = observation_weighting(lossType.hLoss, var)
    return isnothing(w) ? observation_weighting(lossType.vLoss, var) : w
end

"""
    observation_weights(weighting::Symbol, t_obs, tspan)

Weight of each observation in the loss, for observations at times `t_obs`.

  - `:uniform`: every observation counts once, wherever it is in time. This is the right
    choice for sparse observations, such as a few thickness surveys or one velocity map.
  - `:time_span`: each observation is weighted by the part of `tspan` it stands for: the
    time closer to it than to any other observation. The first and the last observations
    also cover the time to the start and to the end of the simulation. The loss then
    approximates an integral over time, and a burst of close observations doesn't dominate.
    This is meant for dense time series.
"""
function observation_weights(weighting::Symbol, t_obs, tspan)
    if weighting == :uniform
        return ones(Sleipnir.Float, length(t_obs))
    elseif weighting == :time_span
        isempty(t_obs) && return Sleipnir.Float[]
        midpoints = (t_obs[begin:(end - 1)] .+ t_obs[(begin + 1):end]) ./ 2
        return diff(clamp.(vcat(tspan[1], midpoints, tspan[2]), tspan[1], tspan[2]))
    else
        throw(ArgumentError("Unknown weighting $(weighting). Use :uniform or :time_span."))
    end
end

"""
    observation_weights(lossType, tH_ref, tV_ref, tspan)

Weights of the ice thickness and ice velocity observations, as the named tuple `(; H, V)`
that the losses receive as `Δt`.
"""
function observation_weights(lossType::AbstractLoss, tH_ref, tV_ref, tspan)
    wH = something(observation_weighting(lossType, :H), :uniform)
    wV = something(observation_weighting(lossType, :V), :uniform)
    return (;
        H = observation_weights(wH, tH_ref, tspan),
        V = observation_weights(wV, tV_ref, tspan))
end
