export AbstractRegularization
export TikhonovRegularization
export VelocityRegularization, DiffusivityRegularization
export loss, backward_loss

# Abstract regularization type as subtype of loss
abstract type AbstractRegularization <: AbstractLoss end

abstract type AbstractSimpleRegularization <: AbstractLoss end
# Basic Regularization Types

"""
Also known as Ridge regression.
"""
@kwdef struct TikhonovRegularization{F <: AbstractFloat} <: AbstractSimpleRegularization
    power::F = 2.0
end

@kwdef struct VelocityRegularization{R <: AbstractSimpleRegularization} <: AbstractRegularization
    reg::R = TikhonovRegularization()
    components::Symbol = :abs
end

@kwdef struct DiffusivityRegularization{R <: AbstractSimpleRegularization} <: AbstractRegularization
    reg::R = TikhonovRegularization()
end

function loss(
    regType::TikhonovRegularization,
    a::Matrix{F},
    Δx::F,
    Δy::F,
) where {F <: AbstractFloat}
    return sum(∇²(a, Δx, Δy).^regType.power)
end
function backward_loss(
    regType::TikhonovRegularization,
    a::Matrix{F},
    Δx::F,
    Δy::F,
) where {F <: AbstractFloat}
    ∂L∂∇²a = regType.power .* ∇²(a, Δx, Δy).^(regType.power .- 1.0)
    return VJP_λ_∂∇²a_∂a(∂L∂∇²a, a, Δx, Δy)
end

function loss(
    regType::VelocityRegularization,
    H::Matrix{F},
    H_ref, # Shoudl remove this! 
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}

    # TODO: extract this from glacier
    Δx, Δy = 1.0, 1.0

    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx, Vy, V = Huginn.V_from_H(simulation, H, t, θ)

    if regType.components == :abs
        return loss(regType.reg, V, Δx, Δy)
    else
        @error "Reg type not implemented."
    end
end
function backward_loss(
    regType::VelocityRegularization,
    H::Matrix{F},
    H_ref, # Shoudl remove this! 
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}

    # TODO: extract this from glacier
    Δx, Δy = 1.0, 1.0

    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx, Vy, V = Huginn.V_from_H(simulation, H, t, θ)

    mask = is_in_glacier(H, 3) .& (V .> 0.0)

    if regType.components == :abs
        ∂Reg∂V = backward_loss(regType.reg, V, Δx, Δy)
        ∂Reg∂Vx, ∂lV∂Vy = zero(∂Reg∂V), zero(∂Reg∂V)
        ∂Reg∂Vx = ifelse.(mask, ∂Reg∂V .* Vx ./ V, 0.0)
        ∂Reg∂Vy = ifelse.(mask, ∂Reg∂V .* Vy ./ V, 0.0)
    else
        @error "Reg type not implemented."
    end

    ∂Reg∂H = VJP_λ_∂surface_V∂H(simulation.parameters.UDE.grad.VJP_method, ∂Reg∂Vx, ∂Reg∂Vy, H, θ, simulation, t)[1]
    ∂Reg∂θ = VJP_λ_∂surface_V∂θ(simulation.parameters.UDE.grad.VJP_method, ∂Reg∂Vx, ∂Reg∂Vy, H, θ, simulation, t)[1]

    return ∂Reg∂H, ∂Reg∂θ
end

# This next part of the code can probably done with something we already have I think

function ∇²(
    a::Matrix{F},
    Δx::F,
    Δy::F,
    ) where {F <: AbstractFloat}
    # First derivative
    ∂a∂x = Huginn.diff_x(a, Δx)
    ∂a∂y = Huginn.diff_y(a, Δy)
    # Evaluate in dual grid
    ∂a∂x_dual = Huginn.avg_y(∂a∂x)
    ∂a∂y_dual = Huginn.avg_x(∂a∂y)
    # Second derivative
    ∂2a∂x2_dual = Huginn.diff_x(∂a∂x_dual, Δx)
    ∂2a∂y2_dual = Huginn.diff_y(∂a∂y_dual, Δy)
    # Evaluate in primal grid
    ∂2a∂x2 = Huginn.avg_y(∂2a∂x2_dual)
    ∂2a∂y2 = Huginn.avg_x(∂2a∂y2_dual)

    ∇²a = zero(a)
    Huginn.inn(∇²a) .= ∂2a∂x2 .+ ∂2a∂y2
    return ∇²a
end

function VJP_λ_∂∇²a_∂a(
    λ::Matrix{R},
    a::Matrix{R},
    Δx::R,
    Δy::R,
) where {R <: Real}
    # First derivative
    ∂λ∂x = Huginn.diff_x(λ, Δx)
    ∂λ∂y = Huginn.diff_y(λ, Δy)
    # Evaluate in dual grid
    ∂λ∂x_dual = Huginn.avg_y(∂λ∂x)
    ∂λ∂y_dual = Huginn.avg_x(∂λ∂y)

    # First derivative
    ∂a∂x = Huginn.diff_x(a, Δx)
    ∂a∂y = Huginn.diff_y(a, Δy)
    # Evaluate in dual grid
    ∂a∂x_dual = Huginn.avg_y(∂a∂x)
    ∂a∂y_dual = Huginn.avg_x(∂a∂y)

    ∇λ∇a = zero(a)
    Huginn.inn(∇λ∇a) .= Huginn.avg(∂λ∂x_dual .* ∂a∂x_dual .+ ∂λ∂y_dual .* ∂a∂y_dual)
    return .- ∇λ∇a
end