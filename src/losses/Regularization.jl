export TikhonovRegularization
export InitialThicknessRegularization, VelocityRegularization, RheologyRegularization,
       SlidingRegularization, DiffusivityRegularization

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
struct TikhonovRegularization{I <: Integer} <: AbstractSimpleRegularization
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
    InitialThicknessRegularization(t₀ = nothing; reg::AbstractSimpleRegularization = TikhonovRegularization(), correction::Bool = false)

A composite regularization type designed for initial ice thickness.
It combines a simple spatial regularization (e.g., `TikhonovRegularization`) with a reference initial time.

# Arguments

  - `t₀`: Time at which the regularization applies. With `nothing` (default), it is the start of the time window of each glacier, see `glacier_tspan`.

# Keyword Arguments

  - `reg::AbstractSimpleRegularization = TikhonovRegularization()`: The spatial regularization operator applied to the initial field. By default, a Tikhonov (Laplacian-based) regularization is used.
  - `correction::Bool = false`: If `true`, the regularization is applied to the correction of the initial thickness, `H₀ - glacier.H₀`, and not to `H₀` itself. The initial thickness of the glacier keeps its own shape, and only the changes made by the inversion are kept smooth. With the Laplacian, a constant or a linear change is not penalized. The reference is `glacier.H₀`, which is the starting point of `θ.IC` only if `InitialCondition` uses the same product; otherwise the `Inversion` shows a warning.
"""
struct InitialThicknessRegularization{
    R <: AbstractSimpleRegularization, T <: Union{Nothing, AbstractFloat}} <:
       AbstractRegularization
    reg::R
    t₀::T
    correction::Bool

    function InitialThicknessRegularization(
            t₀ = nothing; reg::AbstractSimpleRegularization = TikhonovRegularization(),
            correction::Bool = false)
        t₀ = isnothing(t₀) ? nothing : float(t₀)
        return new{typeof(reg), typeof(t₀)}(reg, t₀, correction)
    end
end

"""
    VelocityRegularization(; reg = TikhonovRegularization(), components = :abs, distance = 3)

Regularization for velocity fields, combining a spatial smoothing operator with optional component control.

# Keyword Arguments

  - `reg::AbstractSimpleRegularization = TikhonovRegularization()`: Spatial regularization operator.
  - `components::Symbol = :abs`: Determines which velocity components to regularize (e.g. `:abs`, `:x`, `:y`).
  - `distance::Integer = 3`: Distance to glacier margin.
"""
@kwdef struct VelocityRegularization{R <: AbstractSimpleRegularization, I <: Integer} <:
              AbstractRegularization
    reg::R = TikhonovRegularization()
    components::Symbol = :abs
    distance::I = 3
end

"""
    RheologyRegularization(; reg = TikhonovRegularization())

Regularization of the gridded rheology `A` in the context of classical inversions.
It can include a spatial smoothing operator through the field `reg`.

# Keyword Arguments

  - `reg::AbstractSimpleRegularization = TikhonovRegularization()`: Spatial regularization operator.
"""
@kwdef struct RheologyRegularization{R <: AbstractSimpleRegularization} <:
              AbstractRegularization
    reg::R = TikhonovRegularization()
end

"""
    SlidingRegularization(; reg = TikhonovRegularization())

Regularization of the gridded sliding coefficient `C` in the context of classical
inversions. It is the counterpart of [`RheologyRegularization`](@ref) for `C` and it can
include a spatial smoothing operator through the field `reg`.

The penalty is evaluated on `C` in physical units, that is after mapping `θ.C` through the
same tanh parameterization used by [`LawC`](@ref), so the weight `λ` is expressed relative
to the physical magnitude of `C`.

# Keyword Arguments

  - `reg::AbstractSimpleRegularization = TikhonovRegularization()`: Spatial regularization operator.
"""
@kwdef struct SlidingRegularization{R <: AbstractSimpleRegularization} <:
              AbstractRegularization
    reg::R = TikhonovRegularization()
end

"""
    DiffusivityRegularization(; reg = TikhonovRegularization())

Regularization for diffusivity fields using a specified spatial operator.

# Keyword Arguments

  - `reg::AbstractSimpleRegularization = TikhonovRegularization()`: Spatial regularization operator applied to diffusivity.
"""
@kwdef struct DiffusivityRegularization{R <: AbstractSimpleRegularization} <:
              AbstractRegularization
    reg::R = TikhonovRegularization()
end

### Definition of simple regularization functions

function loss(
        regType::TikhonovRegularization,
        a::Matrix{F},
        Δx::F,
        Δy::F,
        mask::BitMatrix,
        normalization::F
) where {F <: AbstractFloat}
    operator_forward = regType.operator_forward
    return sum(operator_forward(a, Δx, Δy)[mask] .^ 2.0)
end
function backward_loss(
        regType::TikhonovRegularization,
        a::Matrix{F},
        Δx::F,
        Δy::F,
        mask::BitMatrix,
        normalization::F
) where {F <: AbstractFloat}
    operator_forward = regType.operator_forward
    operator_reverse = regType.operator_reverse
    ∂L∂∇²a = zero(a)
    ∂L∂∇²a[mask] = 2.0 .* operator_forward(a, Δx, Δy)[mask]
    ∂L∂a = operator_reverse(∂L∂∇²a, a, Δx, Δy)
    return ∂L∂a
end

# Whether `t` is the start of the time window of the glacier, or `t₀` if it is given
function at_start(t, simulation, glacier_idx, t₀ = nothing)
    t == something(
        t₀, glacier_tspan(simulation, glacier_idx)[1])
end

"""
Field that `InitialThicknessRegularization` is applied to: `H₀`, or its correction `H₀ - glacier.H₀`.
The shift is constant in θ, so the derivative with respect to `H₀` has the same form in both cases.
"""
function regularized_field(lossType::InitialThicknessRegularization, H₀, glacier)
    lossType.correction ? H₀ .- glacier.H₀ : H₀
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
        Δt
) where {F <: AbstractFloat}
    @assert haskey(θ, :IC) """Regularization with respect to initial condition requires to set initial condition as a trainable parameter. If you want to calibrate the initial condition of the glacier, set the initial condition as parameter in the definition of the regressor."""
    if at_start(t, simulation, glacier_idx, lossType.t₀)
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        H₀ = evaluate_H₀(θ, glacier, simulation.parameters.UDE.initial_condition_filter, glacier_idx)
        mask = trues(size(H₀))
        regH = loss(lossType.reg, regularized_field(lossType, H₀, glacier),
            Δx, Δy, mask, normalization)
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
        Δt
) where {F <: AbstractFloat}
    # Regularization is only evaluated for the first time step of the simulation.
    glacier = simulation.glaciers[glacier_idx]
    if at_start(t, simulation, glacier_idx, lossType.t₀)
        Δx, Δy = glacier.Δx, glacier.Δy
        H₀ = evaluate_H₀(θ, glacier, simulation.parameters.UDE.initial_condition_filter, glacier_idx)
        mask = trues(size(H₀))
        ∂L∂H = zero(H₀)
        ∂L∂θ = zero(θ)
        key = Symbol("$(glacier_idx)")
        # Chain rule through the filter and the θ.IC scale: ∂L/∂θ.IC = ∂L/∂H₀ ⋅ ∂H₀/∂θ.IC
        ∂L∂H₀ = backward_loss(lossType.reg, regularized_field(lossType, H₀, glacier),
            Δx, Δy, mask, normalization)
        ∂H₀∂θ = evaluate_∂H₀(
            θ, glacier, simulation.parameters.UDE.initial_condition_filter, glacier_idx)
        ∂L∂θ.IC[key] = ∂L∂H₀ .* ∂H₀∂θ
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
        Δt
) where {F <: AbstractFloat}
    glacier = simulation.glaciers[glacier_idx]
    Δx, Δy = glacier.Δx, glacier.Δy

    if !isnothing(simulation.model.trainable_components)
        simulation.model.trainable_components.θ = θ
    end
    Vx, Vy, V = Huginn.V_from_H(simulation, H, t, θ)
    mask = is_in_glacier(H, regType.distance) .& (V .> 0.0)

    if regType.components == :abs
        return loss(regType.reg, V, Δx, Δy, mask, normalization) * Δt.V
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
        Δt
) where {F <: AbstractFloat}
    glacier = simulation.glaciers[glacier_idx]
    Δx, Δy = glacier.Δx, glacier.Δy

    if !isnothing(simulation.model.trainable_components)
        simulation.model.trainable_components.θ = θ
    end
    Vx, Vy, V = Huginn.V_from_H(simulation, H, t, θ)
    mask = is_in_glacier(H, regType.distance) .& (V .> 0.0)

    if regType.components == :abs
        ∂Reg∂V = backward_loss(regType.reg, V, Δx, Δy, mask, normalization)
        ∂Reg∂Vx = ifelse.(V .> 0.0, ∂Reg∂V .* Vx ./ V, 0.0)
        ∂Reg∂Vy = ifelse.(V .> 0.0, ∂Reg∂V .* Vy ./ V, 0.0)
    else
        @error "Regularization $(regType) not implemented."
    end

    ∂Reg∂H = VJP_λ_∂surface_V∂H(
        simulation.parameters.UDE.grad.VJP_method, ∂Reg∂Vx, ∂Reg∂Vy, H, θ, simulation, t)[1]
    ∂Reg∂θ = VJP_λ_∂surface_V∂θ(
        simulation.parameters.UDE.grad.VJP_method, ∂Reg∂Vx, ∂Reg∂Vy, H, θ, simulation, t)[1]

    return ∂Reg∂H * Δt.V, ∂Reg∂θ * Δt.V
end

function loss(
        regType::RheologyRegularization,
        H::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    if at_start(t, simulation, glacier_idx)
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        min_A = simulation.parameters.physical.minA
        max_A = simulation.parameters.physical.maxA

        key = Symbol("$(glacier_idx)")
        A = @. min_A+(max_A-min_A)*(tanh.(θ.A[key])+1)/2
        mask = trues(size(H) .- 1)

        return loss(regType.reg, A, Δx, Δy, mask, normalization)
    else
        return 0.0
    end
end
function backward_loss(
        regType::RheologyRegularization,
        H::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    if at_start(t, simulation, glacier_idx)
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        min_A = simulation.parameters.physical.minA
        max_A = simulation.parameters.physical.maxA

        key = Symbol("$(glacier_idx)")
        A = @. min_A+(max_A-min_A)*(tanh.(θ.A[key])+1)/2
        mask = trues(size(H) .- 1)
        ∂L∂θ = zero(θ)
        ∂L∂θ.A[key] = backward_loss(regType.reg, A, Δx, Δy, mask, normalization) .*
                      (max_A-min_A) .* (1 .- (tanh.(θ.A[key])) .^ 2) ./ 2

        return zero(H), ∂L∂θ
    else
        return zero(H), zero(θ)
    end
end

function loss(
        regType::SlidingRegularization,
        H::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    if at_start(t, simulation, glacier_idx)
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        max_C = simulation.parameters.physical.maxC

        key = Symbol("$(glacier_idx)")
        # Same tanh mapping as LawC, whose lower bound is always 0
        C = @. max_C*(tanh.(θ.C[key])+1)/2
        mask = trues(size(H) .- 1)

        return loss(regType.reg, C, Δx, Δy, mask, normalization)
    else
        return 0.0
    end
end
function backward_loss(
        regType::SlidingRegularization,
        H::Matrix{F},
        H_ref,
        V_ref, Vx_ref, Vy_ref,
        t::F,
        glacier_idx::Integer,
        θ,
        simulation,
        normalization::F,
        Δt
) where {F <: AbstractFloat}
    if at_start(t, simulation, glacier_idx)
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        max_C = simulation.parameters.physical.maxC

        key = Symbol("$(glacier_idx)")
        C = @. max_C*(tanh.(θ.C[key])+1)/2
        mask = trues(size(H) .- 1)
        ∂L∂θ = zero(θ)
        ∂L∂θ.C[key] = backward_loss(regType.reg, C, Δx, Δy, mask, normalization) .*
                      max_C .* (1 .- (tanh.(θ.C[key])) .^ 2) ./ 2

        return zero(H), ∂L∂θ
    else
        return zero(H), zero(θ)
    end
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
        Δy::F
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
    ∇²a[2:(end - 1), 2:(end - 1)] = ∂2a∂x2 .+ ∂2a∂y2
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
        Δy::R
) where {R <: Real}
    λ_inner = λ[2:(end - 1), 2:(end - 1)]
    ∂a∂x = diff_x_adjoint(avg_y_adjoint(diff_x_adjoint(avg_y_adjoint(λ_inner), Δx)), Δx)
    ∂a∂y = diff_y_adjoint(avg_x_adjoint(diff_y_adjoint(avg_x_adjoint(λ_inner), Δy)), Δy)
    return ∂a∂x + ∂a∂y
end

# `∇²` writes into the interior of a zero matrix, so Zygote refuses it and every Tikhonov
# based regularization fails under SciMLSensitivityAdjoint. The analytical reverse already
# exists and is the one `TikhonovRegularization` pairs with `∇²`, so we hand it to
# ChainRules rather than rewriting the forward.
function ChainRules.rrule(
        ::typeof(∇²),
        a::Matrix{F},
        Δx::F,
        Δy::F
) where {F <: AbstractFloat}
    ∇²a = ∇²(a, Δx, Δy)
    function ∇²_pullback(Δ)
        Δ isa ChainRules.AbstractZero && return (ChainRules.NoTangent(),
            ChainRules.ZeroTangent(), ChainRules.NoTangent(), ChainRules.NoTangent())
        ∂a = VJP_λ_∂∇²a_∂a(ChainRules.unthunk(Δ), a, Δx, Δy)
        return (ChainRules.NoTangent(), ∂a, ChainRules.NoTangent(), ChainRules.NoTangent())
    end
    return ∇²a, ∇²_pullback
end

loss_uses_velocity(lossType::VelocityRegularization) = true
function loss_uses_velocity(lossType::Union{
        AbstractRegularization, AbstractSimpleRegularization})
    false
end
function discreteLossSteps(lossType::InitialThicknessRegularization, tspan)
    [something(
        lossType.t₀, tspan[1])]
end
discreteLossSteps(lossType::RheologyRegularization, tspan) = [tspan[1]]
discreteLossSteps(lossType::SlidingRegularization, tspan) = [tspan[1]]
