export LossDhdt

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
    zero(H_pred), zero(θ)
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
    # @show length(mask)
    # @show t[ind[1]]
    # @show t[ind[2]]
    # @show sum(H0[mask])
    # @show sum(H1[mask])
    dhdt = mean(H1[mask] .- H0[mask])/(tLoss[2]-tLoss[1])
    # @show dhdt
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
    return ∂L∂H, [zero(θ) for i in 1:length(∂L∂H)]
end

loss_uses_velocity(lossType::LossDhdt) = false

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
    (typeof(H_pred)([]), Vector{typeof(θ)}([]))
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
function discretePostIntegralLossSteps(lossType::AbstractLoss, simulation, glacier_idx)
    Vector{Float64}()
end
function discretePostIntegralLossSteps(lossType::MultiLoss, simulation, glacier_idx)
    ts = map(lossType.losses) do l
        discretePostIntegralLossSteps(l, simulation, glacier_idx)
    end
    return vcat(ts...)
end
