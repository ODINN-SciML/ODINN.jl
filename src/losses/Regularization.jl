export TikhonovRegularization
export InitialThicknessRegularization, VelocityRegularization, DiffusivityRegularization

# Abstract regularization type as subtype of loss
abstract type AbstractRegularization <: AbstractLoss end

# Abstract regularization type used in subtype structs of AbstractRegularization
abstract type AbstractSimpleRegularization <: AbstractLoss end

# Basic Regularization Types

"""
    TikhonovRegularization(; operator = :laplacian, distance = 3)

A simple regularization type implementing Tikhonov regularization (also known as ridge regularization)
for inverse problems.

This struct includes both the forward and reverse (adjoint) operators, which are required
for the computation of the gradients with respect to the model parameters.

# Keyword Arguments (Constructor)
- `operator::Symbol = :laplacian`: The regularization operator to use. Currently, only `:laplacian` is implemented, which penalizes large gradients by applying the Laplacian operator.
- `distance::Integer = 3`: A width parameter to determine how far from the margin evaluate the loss.

# Fields (Struct)
- `operator_forward::Function`: The forward regularization operator (e.g., `∇²`).
- `operator_reverse::Function`: The reverse-mode (VJP) of the operator (e.g., `VJP_λ_∂∇²a_∂a`).
- `distance::Integer`: The distance parameter controlling the extent of regularization.
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

"""
    InitialThicknessRegularization(; reg = TikhonovRegularization(), t₀ = 1994.0)

A composite regularization type designed for initial ice thickness.
It combines a simple spatial regularization (e.g., `TikhonovRegularization`) with a reference initial time.

# Keyword Arguments
- `reg::AbstractSimpleRegularization = TikhonovRegularization()`: The spatial regularization operator applied to the initial field. By default, a Tikhonov (Laplacian-based) regularization is used.
- `t₀::AbstractFloat = 1994.0`: The reference initial time (e.g., year) at which the regularization applies.
"""
@kwdef struct InitialThicknessRegularization{R <: AbstractSimpleRegularization, F <: AbstractFloat} <: AbstractRegularization
    reg::R = TikhonovRegularization()
    t₀::F = 1994.0
end

"""
    VelocityRegularization(; reg = TikhonovRegularization(), components = :abs, distance = 3)

Regularization for velocity fields, combining a spatial smoothing operator with optional component control.

# Keyword Arguments
- `reg::AbstractSimpleRegularization = TikhonovRegularization()`: Spatial regularization operator.
- `components::Symbol = :abs`: Determines which velocity components to regularize (e.g. `:abs`, `:x`, `:y`).
- `distance::Integer = 3`: Distance to glacier margin.
"""

@kwdef struct VelocityRegularization{R <: AbstractSimpleRegularization, I <: Integer} <: AbstractRegularization
    reg::R = TikhonovRegularization()
    components::Symbol = :abs
    distance::I = 3
end

"""
    DiffusivityRegularization(; reg = TikhonovRegularization())

Regularization for diffusivity fields using a specified spatial operator.

# Keyword Arguments
- `reg::AbstractSimpleRegularization = TikhonovRegularization()`: Spatial regularization operator applied to diffusivity.
"""
@kwdef struct DiffusivityRegularization{R <: AbstractSimpleRegularization} <: AbstractRegularization
    reg::R = TikhonovRegularization()
end

### Definition of simple regularization functions

function loss(
    regType::TikhonovRegularization,
    a::Matrix{F},
    Δx::F,
    Δy::F,
    mask::BitMatrix,
    normalization::F,
) where {F <: AbstractFloat}
    operator_forward = regType.operator_forward
    return sum(operator_forward(a, Δx, Δy)[mask].^2.0)
end
function backward_loss(
    regType::TikhonovRegularization,
    a::Matrix{F},
    Δx::F,
    Δy::F,
    mask::BitMatrix,
    normalization::F,
) where {F <: AbstractFloat}
    operator_forward = regType.operator_forward
    operator_reverse = regType.operator_reverse
    ∂L∂∇²a = zero(a)
    ∂L∂∇²a[mask] = 2.0 .* operator_forward(a, Δx, Δy)[mask]
    ∂L∂a = operator_reverse(∂L∂∇²a, a, Δx, Δy)
    return ∂L∂a
end

function loss(
    lossType::InitialThicknessRegularization,
    H_pred::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}
    @assert haskey(θ, :IC) """
    Regularization with respect to initial condition requires to set initial condition
    as a trainable parameter. If you want to calibrate the initial condition of the
    glacier, set the initial condition as parameter in the definition of the regressor.
    """
    if t == lossType.t₀
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        H₀ = evaluate_H₀(θ, glacier, simulation.parameters.UDE.initial_condition_filter, glacier_idx)
        mask = trues(size(H₀))
        regH = loss(lossType.reg, H₀, Δx, Δy, mask, normalization)
        return regH
    else
        return 0.0
    end
end
function backward_loss(
    lossType::InitialThicknessRegularization,
    H_pred::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}
    # Regularization is only evaluated for the first time step of the simulation.
    glacier = simulation.glaciers[glacier_idx]
    if t == lossType.t₀
        Δx, Δy = glacier.Δx, glacier.Δy
        H₀ = evaluate_H₀(θ, glacier, simulation.parameters.UDE.initial_condition_filter, glacier_idx)
        mask = trues(size(H₀))
        ∂L∂H = zero(H₀)
        ∂L∂θ = zero(θ)
        key = Symbol("$(glacier_idx)")
        ∂L∂θ.IC[key] = backward_loss(lossType.reg, H₀, Δx, Δy, mask, normalization)
        return ∂L∂H, ∂L∂θ
    else
        return zero(glacier.H₀), zero(θ)
    end
end

function loss(
    regType::VelocityRegularization,
    H::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}

    glacier = simulation.glaciers[glacier_idx]
    Δx, Δy = glacier.Δx, glacier.Δy

    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx, Vy, V = Huginn.V_from_H(simulation, H, t, θ)
    mask = is_in_glacier(H, regType.distance) .& (V .> 0.0)

    if regType.components == :abs
        return loss(regType.reg, V, Δx, Δy, mask, normalization)*Δt.V
    else
        @error "Regularization $(regType) not implemented."
    end
end
function backward_loss(
    regType::VelocityRegularization,
    H::Matrix{F},
    H_ref,
    V_ref, Vx_ref, Vy_ref,
    t::F,
    glacier_idx::Integer,
    θ,
    simulation,
    normalization::F,
    Δt,
) where {F <: AbstractFloat}

    glacier = simulation.glaciers[glacier_idx]
    Δx, Δy = glacier.Δx, glacier.Δy

    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end
    Vx, Vy, V = Huginn.V_from_H(simulation, H, t, θ)
    mask = is_in_glacier(H, regType.distance) .& (V .> 0.0)

    if regType.components == :abs
        ∂Reg∂V = backward_loss(regType.reg, V, Δx, Δy, mask, normalization)
        ∂Reg∂Vx = ifelse.(V.>0.0, ∂Reg∂V .* Vx ./ V, 0.0)
        ∂Reg∂Vy = ifelse.(V.>0.0, ∂Reg∂V .* Vy ./ V, 0.0)
    else
        @error "Regularization $(regType) not implemented."
    end

    ∂Reg∂H = VJP_λ_∂surface_V∂H(simulation.parameters.UDE.grad.VJP_method, ∂Reg∂Vx, ∂Reg∂Vy, H, θ, simulation, t)[1]
    ∂Reg∂θ = VJP_λ_∂surface_V∂θ(simulation.parameters.UDE.grad.VJP_method, ∂Reg∂Vx, ∂Reg∂Vy, H, θ, simulation, t)[1]

    return ∂Reg∂H*Δt.V, ∂Reg∂θ*Δt.V
end

"""
    ∇²(a::Matrix{F}, Δx::F, Δy::F) where {F<:AbstractFloat}

Computes the 2D Laplacian operator of a scalar field `a` on a regular grid
using finite differences and staggered (dual–primal) averaging.

# Arguments
- `a::Matrix{F}`: 2D scalar field to differentiate.
- `Δx::F`: Grid spacing in the x-direction.
- `Δy::F`: Grid spacing in the y-direction.

# Returns
- `Matrix{F}`: Approximation of the Laplacian ∇²a with boundary values set to `0.0`.
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
    ∇²a[2:end-1, 2:end-1] = ∂2a∂x2 .+ ∂2a∂y2
    return ∇²a
end

"""
    VJP_λ_∂∇²a_∂a(λ::Matrix{R}, a::Matrix{R}, Δx::R, Δy::R) where {R<:Real}

Computes the vector-Jacobian product (VJP) of the Laplacian operator `∇²`
with respect to its input field `a`.
This function effectively propagates sensitivities (adjoints) `λ` backward
through the Laplacian, as required in adjoint or reverse-mode differentiation.

# Arguments
- `λ::Matrix{R}`: Adjoint field associated with the Laplacian output.
- `a::Matrix{R}`: Input scalar field to the Laplacian operator.
- `Δx::R`: Grid spacing in the x-direction.
- `Δy::R`: Grid spacing in the y-direction.

# Returns
- `Matrix{R}`: The adjoint (VJP) with respect to `a`, i.e. `∂⟨λ, ∇²a⟩/∂a`.
"""
function VJP_λ_∂∇²a_∂a(
    λ::Matrix{R},
    a::Matrix{R},
    Δx::R,
    Δy::R,
) where {R <: Real}
    λ_inner = λ[2:end-1,2:end-1]
    ∂a∂x = diff_x_adjoint(avg_y_adjoint(diff_x_adjoint(avg_y_adjoint(λ_inner), Δx)), Δx)
    ∂a∂y = diff_y_adjoint(avg_x_adjoint(diff_y_adjoint(avg_x_adjoint(λ_inner), Δy)), Δy)
    return ∂a∂x + ∂a∂y
end


loss_uses_velocity(lossType::VelocityRegularization) = true
loss_uses_velocity(lossType::Union{AbstractRegularization, AbstractSimpleRegularization}) = false
discreteLossSteps(lossType::InitialThicknessRegularization, tspan) = [lossType.t₀]
