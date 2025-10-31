export AbstractRegularization
export TikhonovRegularization
export InitialThicknessRegularization, VelocityRegularization, DiffusivityRegularization
export loss, backward_loss

# Abstract regularization type as subtype of loss
abstract type AbstractRegularization <: AbstractLoss end

abstract type AbstractSimpleRegularization <: AbstractLoss end

# Basic Regularization Types

"""
Also known as Ridge regression.
"""
struct TikhonovRegularization{I<:Integer} <: AbstractSimpleRegularization
    operator_forward::Function
    operator_reverse::Function
    distance::I

    function TikhonovRegularization(; operator = :laplacian, distance = 3)
        if operator == :laplacian
            return new{typeof(distance)}(∇², VJP_λ_∂∇²a_∂a, distance)
        else
            twrow("Operator named $(operator) not implemented inside Tikhonov regularization")
        end
    end
end

@kwdef struct InitialThicknessRegularization{R<:AbstractSimpleRegularization, F<:AbstractFloat} <: AbstractRegularization
    reg::R = TikhonovRegularization()
    t₀::F = 1994.0
end

@kwdef struct VelocityRegularization{R<:AbstractSimpleRegularization, I<:Integer} <: AbstractRegularization
    reg::R = TikhonovRegularization()
    components::Symbol = :abs
    distance::I = 3
end

@kwdef struct DiffusivityRegularization{R <: AbstractSimpleRegularization} <: AbstractRegularization
    reg::R = TikhonovRegularization()
end

function loss(
    regType::TikhonovRegularization,
    a::Matrix{F},
    Δx::F,
    Δy::F;
    normalization::F=1.
) where {F <: AbstractFloat}
    operator_forward = regType.operator_forward
    return sum(operator_forward(a, Δx, Δy).^2.0)
end
function backward_loss(
    regType::TikhonovRegularization,
    a::Matrix{F},
    Δx::F,
    Δy::F;
    normalization::F=1.
) where {F <: AbstractFloat}
    operator_forward = regType.operator_forward
    operator_reverse = regType.operator_reverse
    ∂L∂∇²a = 2.0 .* abs.(operator_forward(a, Δx, Δy))
    return operator_reverse(∂L∂∇²a, a, Δx, Δy)
end

function loss(
    lossType::InitialThicknessRegularization,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    @assert haskey(θ, :IC) """
    Regularization with respect to initial condition requires to set initial condition
    as a trainable parameter. If you want to calibrate the initial condition of the
    glacier, set the initial condition as parameter in the definition of the regressor.
    """
    # TODO: This should be evaluated just when t = t₀, not in general. However, Currently
    # this are evaluated at the points of the quadrature, which usually don't include the
    # extreme values of the time interval.
    # if t == lossType.t₀
        Δx, Δy = glacier.Δx, glacier.Δy
        H₀ = evaluate_H₀(θ, glacier, simulation.parameters.UDE.initial_condition_filter)
        regH = loss(lossType.reg, H₀, Δx, Δy; normalization = normalization)
        return regH
    # else
    #     return 0.0
    # end
end
function backward_loss(
    lossType::InitialThicknessRegularization,
    H_pred::Matrix{F},
    H_ref::Matrix{F},
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}
    # if t == lossType.t₀
        Δx, Δy = glacier.Δx, glacier.Δy
        H₀ = evaluate_H₀(θ, glacier, simulation.parameters.UDE.initial_condition_filter)
        ∂L∂H = zero(H₀)
        ∂L∂θ = zero(θ)
        # Regularization is only evaluated for the first time step of the simulation.
        # However, we save the value of the gradient for every single value of t
        ∂L∂θ.IC[glacier.rgi_id] = backward_loss(lossType.reg, H₀, Δx, Δy; normalization = normalization)
        return ∂L∂H, ∂L∂θ
    # else
    #     return zero(H₀), zero(θ)
    # end
end

function loss(
    regType::VelocityRegularization,
    H::Matrix{F},
    H_ref,
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}

    Δx, Δy = glacier.Δx, glacier.Δy

    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx, Vy, V = Huginn.V_from_H(simulation, H, t, θ)

    if regType.components == :abs
        return loss(regType.reg, V, Δx, Δy)
    else
        @error "Regularization $(regType) not implemented."
    end
end
function backward_loss(
    regType::VelocityRegularization,
    H::Matrix{F},
    H_ref,
    t::F,
    glacier,
    θ,
    simulation;
    normalization::F=1.,
) where {F <: AbstractFloat}

    Δx, Δy = glacier.Δx, glacier.Δy

    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx, Vy, V = Huginn.V_from_H(simulation, H, t, θ)

    mask = is_in_glacier(H, regType.distance) .& (V .> 0.0)

    if regType.components == :abs
        ∂Reg∂V = backward_loss(regType.reg, V, Δx, Δy)
        ∂Reg∂Vx, ∂lV∂Vy = zero(∂Reg∂V), zero(∂Reg∂V)
        ∂Reg∂Vx = ifelse.(mask, ∂Reg∂V .* Vx ./ V, 0.0)
        ∂Reg∂Vy = ifelse.(mask, ∂Reg∂V .* Vy ./ V, 0.0)
    else
        @error "Regularization $(regType) not implemented."
    end

    ∂Reg∂H = VJP_λ_∂surface_V∂H(simulation.parameters.UDE.grad.VJP_method, ∂Reg∂Vx, ∂Reg∂Vy, H, θ, simulation, t)[1]
    ∂Reg∂θ = VJP_λ_∂surface_V∂θ(simulation.parameters.UDE.grad.VJP_method, ∂Reg∂Vx, ∂Reg∂Vy, H, θ, simulation, t)[1]

    return ∂Reg∂H, ∂Reg∂θ
end

# This next part of the code can probably done with something we already have I think
"""
Laplacian operator
"""
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

"""
VJP of the Laplacian operator ∇²
"""
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