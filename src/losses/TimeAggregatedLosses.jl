export LossDhdt, LossAvgV

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
    # zero(H_pred), zero(θ)
    # Use FillArrays to declare a matrix full of zeros without allocation
    (FillArrays.Zeros(size(H_pred)...), zero(θ))
end

"""
    LossDhdt <: TimeAggregatedLoss

A loss function that penalizes the difference between predicted and observed glacier surface elevation change rates (dh/dt).

This loss works with time-aggregated quantities, comparing the mean rate of height change computed from ice thickness predictions against reference dh/dt observations over a specified time interval.

# Details

The loss is computed as:
L = (dhdt_pred - dhdt_ref)²

where:

  - `dhdt_pred`: Predicted mean rate of height change (computed from model ice thickness outputs)
  - `dhdt_ref`: Reference/observed rate of height change from data
  - The rate is computed using masked ice thickness differences by masking out pixels without ice based on the ice thickness at the beginning of the time window
"""
@kwdef struct LossDhdt <: TimeAggregatedLoss
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
    tspan = simulation.parameters.simulation.tspan

    ind = Sleipnir.indFromT(tspan, tLoss, t)
    H0 = H_pred[ind[1]]
    H1 = H_pred[ind[2]]
    mask = H0 .> 1e-2
    dhdt = mean(H1[mask] .- H0[mask])/(tLoss[2]-tLoss[1])
    return (dhdt-dhdt_ref)^2
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
    tspan = simulation.parameters.simulation.tspan
    mask = simulation.glaciers[glacier_idx].mask

    ind = Sleipnir.indFromT(tspan, tLoss, t)
    H0 = H_pred[ind[1]]
    H1 = H_pred[ind[2]]
    mask = H0 .> 1e-2
    dhdt = mean(H1[mask] .- H0[mask])/(tLoss[2]-tLoss[1])
    N = length(H0[mask])

    ∂L∂H = zero(H_pred)
    ∂L∂H[ind[1]] = -2*(dhdt-dhdt_ref)*mask/(N*(tLoss[2]-tLoss[1]))
    ∂L∂H[ind[2]] = 2*(dhdt-dhdt_ref)*mask/(N*(tLoss[2]-tLoss[1]))
    return ∂L∂H, zero(θ)
end

loss_uses_velocity(lossType::LossDhdt) = false

"""
    LossAvgV{F <: AbstractFloat, L <: AbstractSimpleLoss} <: TimeAggregatedLoss

A loss function that penalizes the difference between predicted and observed time-averaged glacier surface velocities.

This loss type computes a time-weighted average of predicted velocities over a specified time interval and compares it against reference velocity observations.
It is particularly useful for constraining glacier flow dynamics when velocity data is annual for example (single snapshot inversions).

# Fields

  - `loss::L = L2Sum()`: The underlying loss function type used to compare predicted and reference velocities
  - `component::Symbol = :xy`: Which velocity component(s) to use in the loss:
      + `:xy`: Compare x and y velocity components separately (sum of losses)
      + `:abs`: Compare absolute velocity magnitude
  - `step::F = 1/12`: Time stepping for velocity aggregation (default: 1 month in yearly units)

# Details

The loss computation involves:

 1. Creating a time grid from `t1` to `t2` with spacing `lossType.step`
 2. Computing predicted velocities at each time step via ice thickness predictions
 3. Time-averaging the velocities with weights proportional to time intervals
 4. Comparing the averaged velocity to reference observations using the specified loss function
"""
@kwdef struct LossAvgV{F <: AbstractFloat, L <: AbstractSimpleLoss} <: TimeAggregatedLoss
    loss::L = L2Sum()
    component::Symbol = :xy
    step::F = 1/12
end

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

    # 1. Determine indices in prediction that will be used for average velocity estimation
    t1 = Sleipnir.datetime_to_floatyear(only(simulation.glaciers[glacier_idx].velocityData.date1))
    t2 = Sleipnir.datetime_to_floatyear(only(simulation.glaciers[glacier_idx].velocityData.date2))
    tLoss = collect(t1:lossType.step:t2)
    dt = diff(tLoss)
    tLoss = tLoss[begin:(end - 1)] # Discard last point t=t2
    ind_pred = Sleipnir.indFromT(simulation.parameters.simulation.tspan, tLoss, t)
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

    ℓ = if lossType.component == :xy
        loss(lossType.loss, avg_Vx_pred, Vx_ref, mask, normalization) +
        loss(lossType.loss, avg_Vy_pred, Vy_ref, mask, normalization)
    elseif lossType.component == :abs
        loss(lossType.loss, avg_V_pred, V_ref, mask, normalization)
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
    ind_pred = Sleipnir.indFromT(simulation.parameters.simulation.tspan, tLoss, t)
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

    if lossType.component == :xy
        ∂l∂Vx = backward_loss(lossType.loss, avg_Vx_pred, Vx_ref, mask, normalization)
        ∂l∂Vy = backward_loss(lossType.loss, avg_Vy_pred, Vy_ref, mask, normalization)
    elseif lossType.component == :abs
        ∂l∂V = backward_loss(lossType.loss, avg_V_pred, V_ref, mask, normalization)
        ∂l∂Vx = ifelse.(mask, ∂l∂V .* (avg_Vx_pred .- Vx_ref) ./ (avg_V_pred .- V_ref), 0.0)
        ∂l∂Vy = ifelse.(mask, ∂l∂V .* (avg_Vy_pred .- Vy_ref) ./ (avg_V_pred .- V_ref), 0.0)
    end

    ∂L∂H = zero(H_pred)
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
    losses = map(
        sub_loss -> time_aggregated_loss(
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
    # Combine contribution of each loss
    return sum(lossType.λs .* losses)
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
function discretePostIntegralLossSteps(lossType::AbstractLoss, simulation, glacier_idx)
    Vector{Float64}()
end
function discretePostIntegralLossSteps(lossType::MultiLoss, simulation, glacier_idx)
    ts = map(lossType.losses) do l
        discretePostIntegralLossSteps(l, simulation, glacier_idx)
    end
    return vcat(ts...)
end
