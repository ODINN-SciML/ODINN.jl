export TikhonovRegularization
export InitialThicknessRegularization, VelocityRegularization, RheologyRegularization,
       SlidingRegularization, DiffusivityRegularization, BedRegularization

# Abstract regularization type as subtype of loss
abstract type AbstractRegularization <: AbstractLoss end

# Abstract regularization type used in subtype structs of AbstractRegularization
abstract type AbstractSimpleRegularization <: AbstractLoss end

# Basic Regularization Types

"""
    TikhonovRegularization(; operator = :gradient, σ = 1.0, distance = 3)

Tikhonov regularization of a field `a`, a prior on its smoothness. It is the mean over the
cells of the glacier of

``\\left(\\frac{|D a|}{σ}\\right)^2``

where `D` is the gradient or the Laplacian, computed in physical units. `σ` is the typical size
of `D a` that the prior allows, so the term is about 1 for a field as rough as `σ`, like a data
term whose misfit is as large as its uncertainty (see [`L2Sum`](@ref)). A smooth field costs
the same at any resolution, and noise from one cell to the next costs more at a finer one.

With the gradient, the mean is taken in each direction over the edges between two cells of
the glacier: ``\\text{mean}((∂_x a / σ)^2) + \\text{mean}((∂_y a / σ)^2)``. This is the
regularization of IGM, where `σ = 1/√w` for an IGM weight `w`.

# Keyword Arguments (Constructor)

  - `operator::Symbol = :gradient`: `:gradient` or `:laplacian`.
  - `σ = 1.0`: Typical size of `D a`, in the units of the field per metre for the gradient (for
    example m/m for a bed), and per square metre for the Laplacian.
  - `distance::Integer = 3`: Not used. The mean is over the cells where ice is allowed
    (`glacier.mask`). It is kept so that old code that sets it still runs.
"""
struct TikhonovRegularization{I <: Integer, F <: AbstractFloat} <:
       AbstractSimpleRegularization
    operator::Symbol
    σ::F
    distance::I

    function TikhonovRegularization(; operator = :gradient, σ = 1.0, distance = 3)
        operator in (:gradient, :laplacian) || throw(ArgumentError(
            "The operator of TikhonovRegularization must be :gradient or :laplacian, \
            got :$(operator)."))
        σ > 0 || throw(ArgumentError("The σ of TikhonovRegularization must be positive."))
        return new{typeof(distance), typeof(float(σ))}(operator, float(σ), distance)
    end
end

"""
    InitialThicknessRegularization(t₀ = nothing; reg::AbstractSimpleRegularization = TikhonovRegularization(), correction::Bool = false)

A composite regularization type designed for initial ice thickness.
It combines a simple spatial regularization (e.g., `TikhonovRegularization`) with a reference initial time.

# Arguments

  - `t₀`: Time at which the regularization applies. With `nothing` (default), it is the start of the time window of each glacier, see `glacier_tspan`.

# Keyword Arguments

  - `reg::AbstractSimpleRegularization = TikhonovRegularization()`: The spatial regularization operator applied to the initial field. By default, a Tikhonov regularization of the gradient is used.
  - `correction::Bool = false`: If `true`, the regularization is applied to the correction of the initial thickness, `H₀ - glacier.H₀`, and not to `H₀` itself. The initial thickness of the glacier keeps its own shape, and only the changes made by the inversion are kept smooth. A constant change is not penalized, and with the Laplacian a linear change is not either. The reference is `glacier.H₀`, which is the starting point of `θ.IC` only if `InitialCondition` uses the same product; otherwise the `Inversion` shows a warning.
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
    Its default `σ = 1` is in the units of `A`, about `1e-17`, so the prior is negligible: set
    `σ` to the change of `A` that you allow between two cells per metre.
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
    Its default `σ = 1` is in the units of `C`, so for a small `C` the prior is negligible: set
    `σ` to the change of `C` that you allow between two cells, per square metre with the Laplacian.
"""
@kwdef struct SlidingRegularization{R <: AbstractSimpleRegularization} <:
              AbstractRegularization
    reg::R = TikhonovRegularization()
end

"""
    BedRegularization(; σ = 0.3, operator = :gradient, convexity = 0.0, convexity_power = 1.3)

Prior on the bed `B`, as in IGM: the whole bed is kept smooth, and the convexity term rewards
thicker ice, so that the smoothness does not flatten the bed where the data say nothing,

``R = \\text{reg}(B) - γ \\, \\text{mean}(H_0), \\qquad γ = \\text{convexity} \\cdot A^{\\text{convexity\\_power} - 2}``

where `A` is the area in m² of the cells where ice is allowed (`glacier.mask`, the outline plus a
buffer), and the mean is over those cells. It is evaluated at the start of the time window of
each glacier. It needs the bed to be trained,
so only `SciMLSensitivityAdjoint` supports it.

The default `σ = 0.3` m/m of the gradient matches the default regularization of IGM
(`regularization.thk = 10`, i.e. `σ = 1/√10`). With the weight `λ` of the `MultiLoss`, an IGM
setup with weight `w` and convexity weight `γ_IGM` maps to `λ/σ² = w/2` and `λγ = w γ_IGM`.

# Keyword Arguments

  - `σ = 0.3`: Typical size of the slope of the bed, in m/m with the gradient.
  - `operator = :gradient`: `:gradient` or `:laplacian`, see [`TikhonovRegularization`](@ref).
  - `reg::AbstractSimpleRegularization = TikhonovRegularization(operator = operator, σ = σ)`:
    Spatial regularization of the bed. For experts, it replaces `σ` and `operator`.
  - `convexity = 0.0`: Weight of the convexity term, 0 (no term) as in IGM.
  - `convexity_power = 1.3`: Power of the area in `γ`, from the volume-area scaling.
"""
struct BedRegularization{R <: AbstractSimpleRegularization, F <: AbstractFloat} <:
       AbstractRegularization
    reg::R
    convexity::F
    convexity_power::F

    function BedRegularization(;
            σ = 0.3, operator = :gradient,
            reg::AbstractSimpleRegularization = TikhonovRegularization(
                operator = operator, σ = σ),
            convexity = 0.0, convexity_power = 1.3)
        convexity, convexity_power = promote(float(convexity), float(convexity_power))
        return new{typeof(reg), typeof(convexity)}(reg, convexity, convexity_power)
    end
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

# Edges in x and in y whose two cells are in `mask`
function edge_masks(mask)
    (mask[2:end, :] .& mask[1:(end - 1), :],
        mask[:, 2:end] .& mask[:, 1:(end - 1)])
end

# Differences between neighbour cells divided by the spacing, with plain slicing so that
# Zygote and Enzyme differentiate them. The reverse is `diff_x_adjoint` and `diff_y_adjoint`.
∂x(a, Δx) = (a[2:end, :] .- a[1:(end - 1), :]) ./ Δx
∂y(a, Δy) = (a[:, 2:end] .- a[:, 1:(end - 1)]) ./ Δy

# Cells of the glacier: where ice is allowed, or all the cells if the glacier has no mask
function glacier_cells(glacier)
    size(glacier.mask) == size(glacier.H₀) ? .!glacier.mask : trues(size(glacier.H₀))
end
# Cells of the dual grid (fields such as `A` and `C`) with a corner in the glacier
function glacier_cells_dual(glacier)
    m = glacier_cells(glacier)
    return m[1:(end - 1), 1:(end - 1)] .| m[2:end, 1:(end - 1)] .|
           m[1:(end - 1), 2:end] .| m[2:end, 2:end]
end

function loss(
        regType::TikhonovRegularization,
        a::Matrix{F},
        Δx::F,
        Δy::F,
        mask::BitMatrix,
        normalization::F
) where {F <: AbstractFloat}
    if regType.operator == :gradient
        mx, my = edge_masks(mask)
        return (masked_mean_sq(∂x(a, Δx), mx) + masked_mean_sq(∂y(a, Δy), my)) / regType.σ^2
    else
        return masked_mean_sq(∇²(a, Δx, Δy), mask) / regType.σ^2
    end
end
function backward_loss(
        regType::TikhonovRegularization,
        a::Matrix{F},
        Δx::F,
        Δy::F,
        mask::BitMatrix,
        normalization::F
) where {F <: AbstractFloat}
    σ² = regType.σ^2
    if regType.operator == :gradient
        mx, my = edge_masks(mask)
        ∂L∂ax = ifelse.(mx, 2 .* ∂x(a, Δx) ./ (σ² * max(count(mx), 1)), zero(F))
        ∂L∂ay = ifelse.(my, 2 .* ∂y(a, Δy) ./ (σ² * max(count(my), 1)), zero(F))
        return diff_x_adjoint(∂L∂ax, Δx) .+ diff_y_adjoint(∂L∂ay, Δy)
    else
        ∂L∂∇²a = ifelse.(mask, 2 .* ∇²(a, Δx, Δy) ./ (σ² * max(count(mask), 1)), zero(F))
        return VJP_λ_∂∇²a_∂a(∂L∂∇²a, a, Δx, Δy)
    end
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
        H₀ = initial_thickness(θ, simulation, glacier_idx)
        mask = @ignore_derivatives glacier_cells(glacier)
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
        H₀ = initial_thickness(θ, simulation, glacier_idx)
        mask = @ignore_derivatives glacier_cells(glacier)
        ∂L∂H = zero(H₀)
        ∂L∂θ = zero(θ)
        # Chain rule through the filter and the θ.IC scale: ∂L/∂θ.IC = ∂L/∂H₀ ⋅ ∂H₀/∂θ.IC
        ∂L∂H₀ = backward_loss(lossType.reg, regularized_field(lossType, H₀, glacier),
            Δx, Δy, mask, normalization)
        scales = ic_scales(simulation, glacier_idx)
        ∂H₀∂θ = evaluate_∂H₀(
            θ, glacier, simulation.parameters.UDE.initial_condition_filter, glacier_idx;
            scale = scales.H, H_min = scales.Hmin)
        ic_params(∂L∂θ, glacier_idx).H .= ∂L∂H₀ .* ∂H₀∂θ
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
        mask = @ignore_derivatives glacier_cells_dual(glacier)

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
        mask = @ignore_derivatives glacier_cells_dual(glacier)
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
        mask = @ignore_derivatives glacier_cells_dual(glacier)

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
        mask = @ignore_derivatives glacier_cells_dual(glacier)
        ∂L∂θ = zero(θ)
        ∂L∂θ.C[key] = backward_loss(regType.reg, C, Δx, Δy, mask, normalization) .*
                      max_C .* (1 .- (tanh.(θ.C[key])) .^ 2) ./ 2

        return zero(H), ∂L∂θ
    else
        return zero(H), zero(θ)
    end
end

# Only the automatic adjoint is supported (see `check_loss`), so there is no `backward_loss`
function loss(
        regType::BedRegularization,
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
        cells = @ignore_derivatives glacier_cells(glacier)
        B = Huginn.bed(simulation, glacier_idx, θ)
        R = loss(regType.reg, B, glacier.Δx, glacier.Δy, cells, normalization)
        iszero(regType.convexity) && return R
        H₀ = haskey(θ, :IC) ? initial_thickness(θ, simulation, glacier_idx) : glacier.H₀
        n_cells = @ignore_derivatives max(count(cells), 1)
        area = n_cells * glacier.Δx * glacier.Δy
        γ = regType.convexity * area^(regType.convexity_power - 2)
        return R - γ * sum(H₀[cells]) / n_cells
    else
        return 0.0
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
discreteLossSteps(lossType::BedRegularization, tspan) = [tspan[1]]
