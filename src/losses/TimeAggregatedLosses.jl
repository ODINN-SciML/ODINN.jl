export LossDhdt, LossAvgV, LossS

# Losses that depend on time aggregated quantities
abstract type TimeAggregatedLoss <: AbstractLoss end

# Fallback methods for subtypes of `AbstractLoss` that do not implement `loss` and `backward_loss`, which is typically the case of subtypes of `TimeAggregatedLoss`
function loss(
        lossType::AbstractLoss,
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
    0.0
end
function backward_loss(
        lossType::AbstractLoss,
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
    zero(H_pred), zero(θ)
    # Use FillArrays to declare a matrix full of zeros without allocation
    # (FillArrays.Zeros(size(H_pred)...), zero(θ))
end

"""
    LossDhdt <: TimeAggregatedLoss

A loss function that penalizes the difference between predicted and observed glacier surface elevation change rates (dh/dt).

This loss works with time-aggregated quantities, comparing the mean rate of height change computed from ice thickness predictions against reference dh/dt observations between the beginning and the end of a specified time interval.

# Details

The loss is computed as:
L = ((dhdt_pred - dhdt_ref) / σ)²

where:

  - `dhdt_pred`: Predicted mean rate of height change (computed from model ice thickness outputs)
  - `dhdt_ref`: Reference/observed rate of height change from data
  - The rate is computed using masked ice thickness differences by masking out pixels without ice based on the ice thickness at the beginning of the time window

# Fields

  - `σ::S`: Uncertainty of the observed rate, in m/yr. With `nothing` (default), it is
    `dhdtData.uncertainty`, and 0.2 m/yr if the data has none. A number replaces it.
"""
@kwdef struct LossDhdt{S <: Union{Nothing, Real}} <: TimeAggregatedLoss
    σ::S = nothing
end

default_σ(::LossDhdt) = 0.2
function dhdt_σ(lossType::LossDhdt, glacier)
    isnothing(lossType.σ) || return lossType.σ
    σ = glacier.dhdtData.uncertainty
    return isfinite(σ) && σ > 0 ? σ : default_σ(lossType)
end

function time_aggregated_loss(
        lossType::LossDhdt,
        H_pred::Vector{Matrix{F}},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::Vector{F},
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    tLoss = simulation.glaciers[glacier_idx].dhdtData.t
    dhdt_ref = simulation.glaciers[glacier_idx].dhdtData.dhdt
    tspan = glacier_tspan(simulation, glacier_idx)

    ind = Sleipnir.indFromT(tspan, tLoss, t)
    H0 = H_pred[ind[1]]
    H1 = H_pred[ind[2]]
    mask = H0 .> 1e-2
    # Glacier fully melted out at the start of the window: no ice-covered pixels
    # to compare against, so this window contributes no loss.
    if !any(mask)
        return zero(F)
    end
    dhdt = mean(H1[mask] .- H0[mask])/(tLoss[2]-tLoss[1])
    σ = @ignore_derivatives dhdt_σ(lossType, simulation.glaciers[glacier_idx])
    return ((dhdt-dhdt_ref)/σ)^2
end
function backward_time_aggregated_loss(
        lossType::LossDhdt,
        H_pred::Vector{Matrix{F}},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::Vector{F},
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    tLoss = simulation.glaciers[glacier_idx].dhdtData.t
    dhdt_ref = simulation.glaciers[glacier_idx].dhdtData.dhdt
    tspan = glacier_tspan(simulation, glacier_idx)
    mask = simulation.glaciers[glacier_idx].mask

    ind = Sleipnir.indFromT(tspan, tLoss, t)
    H0 = H_pred[ind[1]]
    H1 = H_pred[ind[2]]
    mask = H0 .> 1e-2

    # ∂L∂H = zero(H_pred) # With Julia 1.10 in test mode H_pred is of type Type{Matrix{Float64}} which seems to be a bug, we bypass it by making a copy
    ∂L∂H = [zero(H) for H in H_pred]
    # Glacier fully melted out at the start of the window: no ice-covered pixels
    # to compare against, so this window contributes no gradient.
    if !any(mask)
        return ∂L∂H, zero(θ)
    end
    dhdt = mean(H1[mask] .- H0[mask])/(tLoss[2]-tLoss[1])
    N = length(H0[mask])
    σ² = dhdt_σ(lossType, simulation.glaciers[glacier_idx])^2

    ∂L∂H[ind[1]] = -2*(dhdt-dhdt_ref)*mask/(N*(tLoss[2]-tLoss[1])*σ²)
    ∂L∂H[ind[2]] = 2*(dhdt-dhdt_ref)*mask/(N*(tLoss[2]-tLoss[1])*σ²)
    return ∂L∂H, zero(θ)
end

loss_uses_velocity(lossType::LossDhdt) = false

"""
    LossAvgV{F <: AbstractFloat, L <: AbstractSimpleLoss} <: TimeAggregatedLoss

A loss function that penalizes the difference between predicted and observed time-averaged glacier surface velocities.

This loss type computes a time-weighted average of predicted velocities over a specified time interval and compares it against reference velocity observations.
It is particularly useful for constraining glacier flow dynamics when velocity data is annual for example (single snapshot inversions).

# Fields

  - `velocityProduct::Symbol = :Millan22`: Velocity product to use as reference. Defaults to `:Millan22`.
  - `loss::L = L2Sum()`: The underlying loss function type used to compare predicted and reference velocities
  - `component::Symbol = :xy`: Which velocity component(s) to use in the loss:
      + `:xy`: Compare x and y velocity components separately (sum of losses)
      + `:abs`: Compare absolute velocity magnitude
  - `step::F = 1/12`: Time stepping for velocity aggregation (default: 1 month in yearly units)
  - `σ::S = nothing`: Uncertainty of the observed velocity, in m/yr, as in [`LossV`](@ref).

# Details

The loss computation involves:

 1. Creating a time grid from `t1` to `t2` with spacing `lossType.step`
 2. Computing predicted velocities at each time step via ice thickness predictions
 3. Time-averaging the velocities with weights proportional to time intervals
 4. Comparing the averaged velocity to reference observations using the specified loss function
"""
@kwdef struct LossAvgV{
    F <: AbstractFloat, L <: AbstractSimpleLoss, S <: Union{Nothing, Real}} <:
              TimeAggregatedLoss
    velocityProduct::Symbol = :Millan22
    loss::L = L2Sum()
    component::Symbol = :xy
    step::F = 1/12
    σ::S = nothing
end

default_σ(::LossAvgV) = 10.0

function time_aggregated_loss(
        lossType::LossAvgV,
        H_pred::Vector{Matrix{F}},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::Vector{F},
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    @assert length(simulation.glaciers[glacier_idx].velocityData.date1)==1 "With LossAvgV the velocity data should contain exactly one sample."

    t1 = Sleipnir.datetime_to_floatyear(only(simulation.glaciers[glacier_idx].velocityData.date1))
    t2 = Sleipnir.datetime_to_floatyear(only(simulation.glaciers[glacier_idx].velocityData.date2))

    # Get the reference velocity
    V_ref = only(simulation.glaciers[glacier_idx].velocityData.vabs)
    Vx_ref = only(simulation.glaciers[glacier_idx].velocityData.vx)
    Vy_ref = only(simulation.glaciers[glacier_idx].velocityData.vy)

    avg_Vx_pred, avg_Vy_pred,
    avg_V_pred = Huginn.averageV(θ, simulation, (t1, t2), lossType.step, t, H_pred)
    mask = (V_ref .> 0.0)
    σ = @ignore_derivatives velocity_σ(lossType, simulation.glaciers[glacier_idx], 1)

    ℓ = if lossType.component == :xy
        loss(lossType.loss, avg_Vx_pred, Vx_ref, mask, normalization; σ = σ) +
        loss(lossType.loss, avg_Vy_pred, Vy_ref, mask, normalization; σ = σ)
    elseif lossType.component == :abs
        loss(lossType.loss, avg_V_pred, V_ref, mask, normalization; σ = σ)
    else
        @error "Loss type not implemented."
    end

    return ℓ
end
function backward_time_aggregated_loss(
        lossType::LossAvgV,
        H_pred::Vector{Matrix{F}},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::Vector{F},
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}

    # 1. Determine indices in prediction that will be used for average velocity estimation
    t1 = Sleipnir.datetime_to_floatyear(only(simulation.glaciers[glacier_idx].velocityData.date1))
    t2 = Sleipnir.datetime_to_floatyear(only(simulation.glaciers[glacier_idx].velocityData.date2))
    tLoss = collect(t1:lossType.step:t2)
    dt = diff(tLoss)
    tLoss = tLoss[begin:(end - 1)] # Discard last point t=t2
    ind_pred = Sleipnir.indFromT(glacier_tspan(simulation, glacier_idx), tLoss, t)
    T = sum(dt)

    # 2. Get the reference velocity
    V_ref = only(simulation.glaciers[glacier_idx].velocityData.vabs)
    Vx_ref = only(simulation.glaciers[glacier_idx].velocityData.vx)
    Vy_ref = only(simulation.glaciers[glacier_idx].velocityData.vy)

    # 3. Compute the predicted velocity Vx_pred, Vy_pred, V_pred
    if !isnothing(simulation.model.trainable_components)
        simulation.model.trainable_components.θ = θ
    end
    res = map(i -> Huginn.V_from_H(simulation, H_pred[ind_pred[i]], tLoss[i], θ),
        1:length(dt)
    )
    Vx_pred = first.(res)
    Vy_pred = getindex.(res, 2)

    # 4. Aggregate the velocities
    avg_Vx_pred = sum((Vx_pred .* dt)/T)
    avg_Vy_pred = sum((Vy_pred .* dt)/T)
    avg_V_pred = (avg_Vx_pred .^ 2 .+ avg_Vy_pred .^ 2) .^ (1/2)
    mask = (V_ref .> 0.0)
    σ = velocity_σ(lossType, simulation.glaciers[glacier_idx], 1)

    if lossType.component == :xy
        ∂l∂Vx = backward_loss(
            lossType.loss, avg_Vx_pred, Vx_ref, mask, normalization; σ = σ)
        ∂l∂Vy = backward_loss(
            lossType.loss, avg_Vy_pred, Vy_ref, mask, normalization; σ = σ)
    elseif lossType.component == :abs
        ∂l∂V = backward_loss(lossType.loss, avg_V_pred, V_ref, mask, normalization; σ = σ)
        # Chain rule through V = √(Vx² + Vy²): ∂l/∂Vx = ∂l/∂V · Vx/V (and similarly Vy)
        ∂Vx_dir, ∂Vy_dir = VJP_λ_∂V∂Vxy(∂l∂V, avg_Vx_pred, avg_Vy_pred)
        valid = mask .& (avg_V_pred .> 0.0)
        ∂l∂Vx = ifelse.(valid, ∂Vx_dir, 0.0)
        ∂l∂Vy = ifelse.(valid, ∂Vy_dir, 0.0)
    end

    # ∂L∂H = zero(H_pred) # With Julia 1.10 in test mode H_pred is of type Type{Matrix{Float64}} which seems to be a bug, we bypass it by making a copy
    ∂L∂H = [zero(H) for H in H_pred]
    ∂L∂θ = zero(θ)

    cnt = 0
    for i in 1:length(t)
        if t[i] in tLoss
            cnt += 1
            ind = findfirst(t[i] .== tLoss)
            ∂l∂Vx_i = ∂l∂Vx * dt[ind] / T
            ∂l∂Vy_i = ∂l∂Vy * dt[ind] / T

            ∂L∂H[i] = VJP_λ_∂surface_V∂H(simulation.parameters.UDE.grad.VJP_method,
                ∂l∂Vx_i, ∂l∂Vy_i, H_pred[i], θ, simulation, t[i])[1]
            ∂L∂θ += VJP_λ_∂surface_V∂θ(simulation.parameters.UDE.grad.VJP_method,
                ∂l∂Vx_i, ∂l∂Vy_i, H_pred[i], θ, simulation, t[i])[1]
        end
    end
    @assert cnt==length(tLoss)

    return ∂L∂H, ∂L∂θ
end

loss_uses_velocity(lossType::LossAvgV) = true
velocityProduct(lossType::LossAvgV) = :Millan22

"""
    LossS{F <: AbstractFloat} <: TimeAggregatedLoss

Loss on the surface elevation. Each DEM of `glacier.surfaceData` (see `Sleipnir.SurfaceData`)
is compared with the modelled surface `S = B + H` at the date of the DEM:

``L = \\sum_{\\text{DEMs}} \\frac{1}{N} \\sum_{i} \\frac{(S_i - S^{obs}_i)^2}{σ_i^2}, \\qquad σ_i^2 = σ_{pixel,i}^2 + σ_{sys}^2``

The sum is over the `N` cells of the DEM where ice is allowed (`glacier.mask` is false) and
that have a value. So each DEM adds about 1 to the loss when the misfit is as large as its
uncertainty. The DEMs outside the time window of the glacier are not used.

With a free bed ([`InitialCondition`](@ref) with `bed = :free`, its default), the DEMs constrain
the surface, and the ice thickness observations then constrain the bed. An inversion that does
not train an `InitialCondition` has no free bed. A DEM at the start of the
time window is the misfit of the initial surface `S₀`. Only `SciMLSensitivityAdjoint`
supports it.

# Fields

  - `σ_sys::F`: Uncertainty that the DEM products do not give, in m, added to the one of the
    product (see the formula above). It stands for the penetration of the radar into snow and
    firn and for the random error of a DEM that has no value per pixel (NASADEM). Default is
    `5.0`, of the order of the values found in the literature. For the X-band (TanDEM-X) the
    penetration is negligible below about 2500 m a.s.l. and about 4 m on average at 4000 m in
    the Mont-Blanc area (Dehecq et al., 2016), and the bias reached 4 to 8 m in the accumulation
    area of a large Alpine glacier (Bannwart et al., 2024). For the C-band (SRTM), heights up to
    10 m too low were found at high elevation (Berthier et al., 2006), see also Gardelle et al.
    (2012). The penetration is a bias, not noise, and it is about 0 at low elevation, so one
    value for the whole glacier is a first approximation.

# References

  - Bannwart, J., Piermattei, L., Dussaillant, I., Krieger, L., Floricioiu, D., Berthier, E.,
    Roeoesli, C., Machguth, H. and Zemp, M. (2024). Elevation bias due to penetration of
    spaceborne radar signal on Grosser Aletschgletscher, Switzerland. Journal of Glaciology,
    70, e1. doi:10.1017/jog.2024.37
  - Berthier, E., Arnaud, Y., Vincent, C. and Rémy, F. (2006). Biases of SRTM in high-mountain
    areas: Implications for the monitoring of glacier volume changes. Geophysical Research
    Letters, 33, L08502. doi:10.1029/2006GL025862
  - Dehecq, A., Millan, R., Berthier, E., Gourmelen, N., Trouvé, E. and Vionnet, V. (2016).
    Elevation changes inferred from TanDEM-X data over the Mont-Blanc area: Impact of the
    X-band interferometric bias. IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing, 9(8), 3870-3882. doi:10.1109/JSTARS.2016.2581482
  - Gardelle, J., Berthier, E. and Arnaud, Y. (2012). Impact of resolution and radar penetration
    on glacier elevation changes computed from DEM differencing. Journal of Glaciology, 58(208),
    419-422. doi:10.3189/2012JoG11J175
"""
@kwdef struct LossS{F <: AbstractFloat} <: TimeAggregatedLoss
    σ_sys::F = 5.0
end

# DEMs of the glacier inside its time window
function surface_obs_in_window(simulation, glacier_idx)
    data = simulation.glaciers[glacier_idx].surfaceData
    isnothing(data) && return Int[]
    tspan = glacier_tspan(simulation, glacier_idx)
    return findall(t -> tspan[1] <= t <= tspan[2], data.t)
end

# Observed surface with zeros where it is not used, and the weight of each cell
function surface_obs_weights(lossType::LossS, glacier, k)
    S_obs = glacier.surfaceData.S[k]
    σ_pixel = glacier.surfaceData.σ[k]
    used = isfinite.(S_obs) .& isfinite.(σ_pixel) .& .!glacier.mask
    w = ifelse.(used, 1 ./ (σ_pixel .^ 2 .+ lossType.σ_sys^2), zero(eltype(σ_pixel))) ./
        max(count(used), 1)
    return ifelse.(used, S_obs, zero(eltype(S_obs))), w
end

function time_aggregated_loss(
        lossType::LossS,
        H_pred::Vector{Matrix{F}},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::Vector{F},
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    glacier = simulation.glaciers[glacier_idx]
    B = Huginn.bed(simulation, glacier_idx, θ)
    obs = @ignore_derivatives surface_obs_in_window(simulation, glacier_idx)
    tspan = @ignore_derivatives glacier_tspan(simulation, glacier_idx)
    ℓ = zero(F)
    for k in obs
        i = @ignore_derivatives only(
            Sleipnir.indFromT(tspan, [glacier.surfaceData.t[k]], t))
        S_obs, w = @ignore_derivatives surface_obs_weights(lossType, glacier, k)
        S = B .+ max.(H_pred[i], zero(eltype(H_pred[i])))
        ℓ += sum(w .* (S .- S_obs) .^ 2)
    end
    return ℓ
end

loss_uses_velocity(lossType::LossS) = false

# Fallback methods for subtypes of `AbstractLoss` that do not implement `time_aggregated_loss` and `backward_time_aggregated_loss`, which is typically the case of all losses which are not subtypes of `TimeAggregatedLoss`
function time_aggregated_loss(
        lossType::AbstractLoss,
        H_pred::Vector{Matrix{F}},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::Vector{F},
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    0.0
end
function backward_time_aggregated_loss(
        lossType::AbstractLoss,
        H_pred::Vector{Matrix{F}},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::Vector{F},
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    # (typeof(H_pred)([]), zero(θ))
    # Use FillArrays to declare a vector of matrices full of zeros without allocation
    (FillArrays.Fill(FillArrays.Zeros(size(H_pred[1])...), length(H_pred)), zero(θ))
end

function time_aggregated_loss(
        lossType::MultiLoss,
        H_pred::Vector{Matrix{F}},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::Vector{F},
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    λs = @ignore_derivatives(SVector(lossType.λs...))
    return sum(ntuple(
        i -> ( # Map doesn't work with Zygote, this is why we use this ntuple
            λs[i] * time_aggregated_loss( # Combine contribution of each loss
            lossType.losses[i],
            H_pred,
            H_ref,
            V_ref, Vx_ref, Vy_ref,
            t,
            glacier_idx,
            θ,
            simulation,
            normalization,
            Δt
        )
        ),
        length(lossType.losses)))
end
function backward_time_aggregated_loss(
        lossType::MultiLoss,
        H_pred::Vector{Matrix{F}},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::Vector{F},
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    res_backward_losses = map(
        sub_loss -> backward_time_aggregated_loss(
            sub_loss,
            H_pred,
            H_ref,
            V_ref, Vx_ref, Vy_ref,
            t,
            glacier_idx,
            θ,
            simulation,
            normalization,
            Δt
        ),
        lossType.losses
    )
    # Combine contribution of each gradient
    ∂L∂Hs = first.(res_backward_losses)
    ∂L∂θs = last.(res_backward_losses)
    ∂L∂H = sum(lossType.λs .* ∂L∂Hs)
    ∂L∂θ = sum(lossType.λs .* ∂L∂θs)
    return ∂L∂H, ∂L∂θ
end

function discretePostIntegralLossSteps(lossType::LossDhdt, simulation, glacier_idx)
    [simulation.glaciers[glacier_idx].dhdtData.t...] # Transform tuple into vector
end
function discretePostIntegralLossSteps(lossType::LossAvgV, simulation, glacier_idx)
    @assert length(simulation.glaciers[glacier_idx].velocityData.date1)==1 "With LossAvgV the velocity data should contain exactly one sample."
    # Determine indices in prediction that will be used for average velocity estimation
    t1 = Sleipnir.datetime_to_floatyear(only(simulation.glaciers[glacier_idx].velocityData.date1))
    t2 = Sleipnir.datetime_to_floatyear(only(simulation.glaciers[glacier_idx].velocityData.date2))
    tLoss = collect(t1:lossType.step:t2)
    tLoss = tLoss[begin:(end - 1)] # Discard last point t=t2
    return tLoss
end
function discretePostIntegralLossSteps(lossType::LossS, simulation, glacier_idx)
    data = simulation.glaciers[glacier_idx].surfaceData
    isnothing(data) && return Vector{Sleipnir.Float}()
    return Vector{Sleipnir.Float}(data.t[surface_obs_in_window(simulation, glacier_idx)])
end
function discretePostIntegralLossSteps(lossType::AbstractLoss, simulation, glacier_idx)
    Vector{Float64}()
end

"""
    check_loss(lossType, simulation)

Check, before any simulation runs, that a loss can be used with this inversion. The times a
loss needs must be inside the time window of each glacier: times outside it are never reached
by the solver, and the loss would later fail with an unclear `invalid index: nothing`, or be
ignored.
"""
check_loss(lossType::AbstractLoss, simulation) = nothing
function check_loss(lossType::MultiLoss, simulation)
    foreach(l -> check_loss(l, simulation), lossType.losses)
end
function check_loss(lossType::Union{LossH, LossV}, simulation)
    # The options are checked now, and not in the first call of the loss
    observation_weights(lossType.weighting, [0.0], (0.0, 1.0))
    lossType isa LossH && warn_empty_thickness_mask(lossType, simulation)
    return nothing
end
# The mask of `LossH` erodes a sparse survey (for example Glathida) to nothing
function warn_empty_thickness_mask(lossType::LossH, simulation)
    for glacier in simulation.glaciers
        H_obs = isnothing(glacier.thicknessData) ? [] :
                something(glacier.thicknessData.H, [])
        if any(H -> any(H .> 0) && !any(is_in_glacier(H, lossType.loss.distance)), H_obs)
            @warn "LossH: a thickness observation of glacier $(glacier.rgi_id) has an empty \
                mask with distance = $(lossType.loss.distance), so it is ignored. For sparse \
                surveys, use a loss with distance = 0, for example L2Sum(distance = 0)."
        end
    end
end
function check_loss(lossType::LossHV, simulation)
    # A time aggregated loss is not evaluated inside a `LossHV`: it would be silently ignored
    for l in (lossType.hLoss, lossType.vLoss)
        uses_loss(l, TimeAggregatedLoss) && throw(ArgumentError(
            "LossHV cannot hold a time aggregated loss (LossS, LossAvgV or LossDhdt), because \
            it would have no effect. Use a MultiLoss, for example \
            MultiLoss(losses = (LossHV(), LossS()))."))
    end
    check_loss(lossType.hLoss, simulation)
    check_loss(lossType.vLoss, simulation)
    return nothing
end
# The default `σ = 1` of `TikhonovRegularization` is in the units of the field, so for `A` (about
# 1e-17) or `C` the prior is negligible, and nothing else shows it. `scale` is the size of
# the parameter, and a change of `scale` between two cells is what the prior sees.
function warn_negligible_prior(lossType, name, scale, simulation)
    reg = lossType.reg
    reg isa TikhonovRegularization || return nothing
    for glacier in simulation.glaciers
        Δ = min(glacier.Δx, glacier.Δy)
        change = reg.operator == :gradient ? scale / Δ : scale / Δ^2
        if (change / reg.σ)^2 < 1e-6
            @warn "$(name) of glacier $(glacier.rgi_id): with σ = $(reg.σ) the prior is \
                negligible (a change of $(scale) between two cells costs \
                $((change / reg.σ)^2)), so it has no effect. Set `σ` in \
                `TikhonovRegularization` to the change that you allow."
            return nothing
        end
    end
    return nothing
end

# The regularizations of a gridded parameter need it to be trained
function check_loss(lossType::GriddedParameterRegularization, simulation)
    θ = simulation.model.trainable_components.θ
    physical = simulation.parameters.physical
    key,
    scale = lossType isa SlidingRegularization ? (:C, physical.maxC) :
            (:A, physical.maxA)
    name = nameof(typeof(lossType))
    (!isnothing(θ) && haskey(θ, key)) || throw(ArgumentError(
        "$(name) needs `$(key)` to be trained."))
    warn_negligible_prior(lossType, name, scale, simulation)
    return nothing
end
function check_loss(lossType::BedRegularization, simulation)
    θ = simulation.model.trainable_components.θ
    all(i -> bed_is_free(θ, i), eachindex(simulation.glaciers)) || throw(ArgumentError(
        "BedRegularization needs the bed to be free, with `InitialCondition(...; bed = :free)`."))
    return nothing
end
# The surface loss has no manual adjoint. With a free bed the DEMs are the only data of the
# surface, so a glacier needs at least one of them in its window, and the one at the start of
# the window is what pins the initial surface
function check_loss(lossType::LossS, simulation)
    simulation.parameters.UDE.grad isa SciMLSensitivityAdjoint || throw(ArgumentError(
        "LossS needs `SciMLSensitivityAdjoint`."))
    θ = simulation.model.trainable_components.θ
    step = simulation.parameters.solver.step
    for (i, glacier) in enumerate(simulation.glaciers)
        tspan = glacier_tspan(simulation, i)
        obs = surface_obs_in_window(simulation, i)
        dems = isnothing(glacier.surfaceData) ? "missing" : glacier.surfaceData.t
        free_bed = bed_is_free(θ, i)
        if isempty(obs)
            msg = "LossS: glacier $(glacier.rgi_id) has no DEM inside its time window \
                $(tspan). The DEMs are $(dems)."
            free_bed && throw(ArgumentError(msg * " The free bed needs the surface: add a \
                DEM, or use `InitialCondition(...; bed = :fixed)`."))
            @warn msg * " The loss does not use it."
        elseif free_bed && all(t -> t - tspan[1] > step, glacier.surfaceData.t[obs])
            @warn "LossS: glacier $(glacier.rgi_id) has no DEM at the start of its time \
                window $(tspan), only at $(glacier.surfaceData.t[obs]), so nothing pins the \
                initial surface of the free bed."
        end
    end
    return nothing
end
function check_loss(lossType::LossAvgV, simulation)
    for (i, glacier) in enumerate(simulation.glaciers)
        tspan = glacier_tspan(simulation, i)
        if isnothing(glacier.velocityData)
            throw(ArgumentError("LossAvgV needs velocity data, but glacier \
                $(glacier.rgi_id) has none."))
        end
        tLoss = discretePostIntegralLossSteps(lossType, simulation, i)
        if !all(t -> tspan[1] <= t <= tspan[2], tLoss)
            t1 = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date1))
            t2 = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date2))
            throw(ArgumentError("LossAvgV averages the velocity over the period of the \
                velocity observation, $(round(t1; digits = 3)) to $(round(t2; digits = 3)) \
                for glacier $(glacier.rgi_id), but the simulation only covers \
                tspan = $(tspan). This period must be inside tspan: extend tspan, or use \
                LossV, which compares the velocity at the observation dates only."))
        end
    end
end
function check_loss(lossType::LossDhdt, simulation)
    for (i, glacier) in enumerate(simulation.glaciers)
        tspan = glacier_tspan(simulation, i)
        if isnothing(glacier.dhdtData)
            throw(ArgumentError("LossDhdt needs elevation change data, but glacier \
                $(glacier.rgi_id) has none."))
        end
        t1, t2 = glacier.dhdtData.t
        if !(tspan[1] <= t1 && t2 <= tspan[2])
            throw(ArgumentError("LossDhdt compares the elevation change from $(t1) to \
                $(t2) for glacier $(glacier.rgi_id), but the simulation only covers \
                tspan = $(tspan). This period must be inside tspan."))
        end
    end
end
function check_loss(lossType::InitialThicknessRegularization, simulation)
    for (i, glacier) in enumerate(simulation.glaciers)
        tspan = glacier_tspan(simulation, i)
        if !isnothing(lossType.t₀) && !(tspan[1] <= lossType.t₀ <= tspan[2])
            throw(ArgumentError("InitialThicknessRegularization is applied at \
                t₀ = $(lossType.t₀), outside the time window $(tspan) of glacier \
                $(glacier.rgi_id), so it would be ignored. Use \
                InitialThicknessRegularization() to apply it at the start of each glacier."))
        end
    end
end
function discretePostIntegralLossSteps(lossType::MultiLoss, simulation, glacier_idx)
    ts = map(lossType.losses) do l
        discretePostIntegralLossSteps(l, simulation, glacier_idx)
    end
    return vcat(ts...)
end
