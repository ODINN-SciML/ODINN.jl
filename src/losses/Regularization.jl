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
    InitialThicknessRegularization(t₀::AbstractFloat; reg::AbstractSimpleRegularization = TikhonovRegularization())

A composite regularization type designed for initial ice thickness.
It combines a simple spatial regularization (e.g., `TikhonovRegularization`) with a reference initial time.

# Keyword Arguments

  - `reg::AbstractSimpleRegularization = TikhonovRegularization()`: The spatial regularization operator applied to the initial field. By default, a Tikhonov (Laplacian-based) regularization is used.
  - `t₀::AbstractFloat = 1994.0`: The reference initial time (e.g., year) at which the regularization applies.
"""
struct InitialThicknessRegularization{
    R <: AbstractSimpleRegularization, F <: AbstractFloat} <: AbstractRegularization
    reg::R
    t₀::F

    function InitialThicknessRegularization(
            t₀::AbstractFloat; reg::AbstractSimpleRegularization = TikhonovRegularization())
        return new{typeof(reg), typeof(t₀)}(reg, t₀)
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
same tanh parameterisation used by [`LawC`](@ref), so the weight `λ` is expressed relative
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
        Δt
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
        # θ.IC is not H₀ itself: H₀ = positive(θ.IC ⋅ scale), with `positive` the
        # `initial_condition_filter` (softplus or Zang). So ∂L/∂θ.IC = ∂L/∂H₀ ⋅ ∂H₀/∂θ.IC
        ∂L∂H₀ = backward_loss(lossType.reg, H₀, Δx, Δy, mask, normalization)
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

# Regularizations of a gridded parameter, `A` or `C`, evaluated in physical units
const GriddedParameterRegularization = Union{RheologyRegularization, SlidingRegularization}

# Part of θ with the gridded parameter, and the bounds of its tanh map (the same as in
# `LawA` and `LawC`)
function gridded_parameter(::RheologyRegularization, θ, physical)
    (θ.A, physical.minA, physical.maxA)
end
# The lower bound of LawC is always 0
function gridded_parameter(::SlidingRegularization, θ, physical)
    (θ.C, zero(physical.maxC), physical.maxC)
end

function loss(
        regType::GriddedParameterRegularization,
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
    if t == simulation.parameters.simulation.tspan[1]
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        θ_p, lo, hi = gridded_parameter(regType, θ, simulation.parameters.physical)

        key = Symbol("$(glacier_idx)")
        field = @. lo+(hi-lo)*(tanh.(θ_p[key])+1)/2
        mask = trues(size(H) .- 1)

        return loss(regType.reg, field, Δx, Δy, mask, normalization)
    else
        return 0.0
    end
end
function backward_loss(
        regType::GriddedParameterRegularization,
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
    if t == simulation.parameters.simulation.tspan[1]
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        θ_p, lo, hi = gridded_parameter(regType, θ, simulation.parameters.physical)

        key = Symbol("$(glacier_idx)")
        field = @. lo+(hi-lo)*(tanh.(θ_p[key])+1)/2
        mask = trues(size(H) .- 1)
        ∂L∂θ = zero(θ)
        # `getproperty` gives a view, so this writes into ∂L∂θ
        ∂L∂θ_p, _, _ = gridded_parameter(regType, ∂L∂θ, simulation.parameters.physical)
        ∂L∂θ_p[key] = backward_loss(regType.reg, field, Δx, Δy, mask, normalization) .*
                      (hi-lo) .* (1 .- (tanh.(θ_p[key])) .^ 2) ./ 2

        return zero(H), ∂L∂θ
    else
        return zero(H), zero(θ)
    end
end

"""
    ∇²(a::Matrix{F}, Δx::F, Δy::F) where {F<:AbstractFloat}

Computes the 2D Laplacian operator of a scalar field `a` on a regular grid
with the standard 5-point finite difference stencil.

The previous version differentiated twice with an average across the other direction in
between (staggered grid). That operator is exactly zero on a checkerboard `(-1)^(i+j)`, so a
Laplacian regularization could not see cell-to-cell alternating patterns. The 5-point stencil
has eigenvalue `-4/Δx² - 4/Δy²` there.

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
    ∇²a = zero(a)
    @views ∇²a[2:(end - 1), 2:(end - 1)] = (a[3:end, 2:(end - 1)] .-
                                            2 .* a[2:(end - 1), 2:(end - 1)] .+
                                            a[1:(end - 2), 2:(end - 1)]) ./ Δx^2 .+
                                           (a[2:(end - 1), 3:end] .-
                                            2 .* a[2:(end - 1), 2:(end - 1)] .+
                                            a[2:(end - 1), 1:(end - 2)]) ./ Δy^2
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
    # The output of `∇²` is zero on the border, so only the interior of λ counts
    λ_in = zero(λ)
    λ_in[2:(end - 1), 2:(end - 1)] = λ[2:(end - 1), 2:(end - 1)]
    ∂a = -2 * (1 / Δx^2 + 1 / Δy^2) .* λ_in
    ∂a[2:end, :] .+= λ_in[1:(end - 1), :] ./ Δx^2
    ∂a[1:(end - 1), :] .+= λ_in[2:end, :] ./ Δx^2
    ∂a[:, 2:end] .+= λ_in[:, 1:(end - 1)] ./ Δy^2
    ∂a[:, 1:(end - 1)] .+= λ_in[:, 2:end] ./ Δy^2
    return ∂a
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
discreteLossSteps(lossType::InitialThicknessRegularization, tspan) = [lossType.t₀]
discreteLossSteps(lossType::GriddedParameterRegularization, tspan) = [tspan[1]]
