export LossDhdt, LossAvgV

# Losses that depend on time aggregated quantities
abstract type AggregatedLoss <: AbstractLoss end

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

@kwdef struct LossDhdt <: AggregatedLoss
end

function aggregated_loss(
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
function backward_aggregated_loss(
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

@kwdef struct LossAvgV{F <: AbstractFloat, L <: AbstractSimpleLoss} <: AggregatedLoss
    loss::L = L2Sum()
    component::Symbol = :xy
    step::F = 1/12
end

function aggregated_loss(
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

    # 4. Aggregate the velocities
    res = map(i -> Huginn.V_from_H(simulation, H_pred[ind_pred[i]], tLoss[i], θ),
        1:length(dt)
    )
    Vx_pred = first.(res)
    Vy_pred = getindex.(res, 2)
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
function backward_aggregated_loss(
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

    # 4. Aggregate the velocities
    res = map(i -> Huginn.V_from_H(simulation, H_pred[ind_pred[i]], tLoss[i], θ),
        1:length(dt)
    )
    Vx_pred = first.(res)
    Vy_pred = getindex.(res, 2)
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

function aggregated_loss(
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
function backward_aggregated_loss(
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

function aggregated_loss(
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
        sub_loss -> aggregated_loss(
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
function backward_aggregated_loss(
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
    # TODO: check that we handle ∂L∂H as vectors properly
    res_backward_losses = map(
        sub_loss -> backward_aggregated_loss(
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
