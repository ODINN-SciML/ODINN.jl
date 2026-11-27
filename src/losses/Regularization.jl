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
the glacier: ``\\text{mean}((∂_x a / σ)^2) + \\text{mean}((∂_y a / σ)^2)``. This is the smoothness
prior of Jouvet (2023).

# Keyword Arguments (Constructor)

  - `operator::Symbol = :gradient`: `:gradient` or `:laplacian`.
  - `σ = 1.0`: Typical size of `D a`, in the units of the field per metre for the gradient (for
    example m/m for a bed), and per square metre for the Laplacian.
  - `distance::Integer = 3`: Not used. The mean is over the cells where ice is allowed
    (`glacier.mask`).

# References

  - Jouvet, G. (2023). Inversion of a Stokes glacier flow model emulated by deep learning.
    Journal of Glaciology, 69(273), 13-26. doi:10.1017/jog.2022.41
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
    InitialThicknessRegularization(t₀ = nothing; reg = nothing, correction = true, σ_relative = 0.3)

Prior on the smoothness of the initial ice thickness `H₀` when it is trained. It combines a
simple spatial regularization (a [`TikhonovRegularization`](@ref)) with the time at which it
applies.

By default the prior is the Laplacian of the change that the inversion makes to the starting
thickness, `H₀ - glacier.H₀`, with a `σ` of 0.3 times the roughness of `glacier.H₀`. The shape
of `glacier.H₀` is then kept and only the changes are smooth. This avoids small scale changes,
which decay in a few weeks once the simulation starts and create a shock in the first time
steps. In a test on one glacier, it gave a lower error against thickness measurements that
were not used in the inversion than a prior on `H₀` itself, and no checkerboard pattern.

# Arguments

  - `t₀`: Time at which the regularization applies. With `nothing` (default), it is the start of the time window of each glacier, see `glacier_tspan`.

# Keyword Arguments

  - `reg = nothing`: The spatial regularization. With `nothing` (default), it is `TikhonovRegularization(operator = :laplacian)`.
  - `correction::Bool = true`: If `true`, the regularization is applied to the change `H₀ - glacier.H₀`. If `false`, it is applied to `H₀` itself. Start `InitialCondition` from the same product as `glacier.H₀`, so that the penalty is zero at the start.
  - `σ_relative`: If set, the `σ` of `reg` (a `TikhonovRegularization`) is replaced, for each glacier, by `σ_relative` times the roughness of `glacier.H₀` (the root mean square of the same operator over the same cells). One value then fits any glacier and grid. Default is `0.3` when `reg` is not given, and `nothing` when it is, so that the `σ` of your `reg` is used as it is.
"""
struct InitialThicknessRegularization{
    R <: AbstractSimpleRegularization, T <: Union{Nothing, AbstractFloat},
    S <: Union{Nothing, AbstractFloat}} <: AbstractRegularization
    reg::R
    t₀::T
    correction::Bool
    σ_relative::S

    function InitialThicknessRegularization(
            t₀ = nothing; reg::Union{Nothing, AbstractSimpleRegularization} = nothing,
            correction::Bool = true, σ_relative = isnothing(reg) ? 0.3 : nothing)
        reg = something(reg, TikhonovRegularization(operator = :laplacian))
        t₀ = isnothing(t₀) ? nothing : float(t₀)
        σ_relative = check_σ_relative(σ_relative, reg)
        return new{typeof(reg), typeof(t₀), typeof(σ_relative)}(reg, t₀, correction, σ_relative)
    end
end

function check_σ_relative(σ_relative, reg)
    isnothing(σ_relative) && return nothing
    reg isa TikhonovRegularization || throw(ArgumentError(
        "`σ_relative` needs a TikhonovRegularization, got $(typeof(reg))."))
    σ_relative > 0 || throw(ArgumentError("`σ_relative` must be positive."))
    return float(σ_relative)
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
    BedRegularization(; σ = nothing, operator = :gradient, reg = nothing, convexity = 0.0,
        convexity_power = 1.3, σ_relative = 1.0)

Prior on the bed `B`: the whole bed is kept smooth, and an optional convexity term rewards
thicker ice, so that the smoothness does not flatten the bed where the data say nothing,

``R = \\text{reg}(B) - γ \\, \\text{mean}(H_0), \\qquad γ = \\text{convexity} \\cdot A^{\\text{convexity\\_power} - 2}``

where `A` is the area in m² of the cells where ice is allowed (`glacier.mask`, the outline plus a
buffer), and the mean is over those cells. Both terms follow Jouvet (2023) (see also
Jouvet et al., 2022). The prior is evaluated at the start of the time window of each glacier.
It needs the bed to be trained, so only `SciMLSensitivityAdjoint` supports it.

By default, `σ` is `σ_relative = 1` times the roughness of the starting bed `glacier.B`, so
that one value fits any glacier and grid. To use a fixed `σ` instead, give `σ`, for example
`σ = 0.3` m/m of the gradient.

# Keyword Arguments

  - `σ = nothing`: Typical size of the slope of the bed, in m/m with the gradient. If given, `σ_relative` is not used. If `nothing` and `σ_relative` is `nothing` too, it is `0.3`.
  - `operator = :gradient`: `:gradient` or `:laplacian`, see [`TikhonovRegularization`](@ref).
  - `reg = nothing`: Spatial regularization of the bed, to replace `σ` and `operator`. Its `σ` is used as it is, unless you give `σ_relative`.
  - `convexity = 0.0`: Weight of the convexity term. With 0 there is no term.
  - `convexity_power = 1.3`: Power of the area in `γ`.
  - `σ_relative = 1.0`: `σ` is replaced, for each glacier, by `σ_relative` times the roughness of the starting bed `glacier.B` (the root mean square of the same operator over the same cells). It is `nothing` if you give `σ` or `reg`.

# References

  - Jouvet, G. (2023). Inversion of a Stokes glacier flow model emulated by deep learning.
    Journal of Glaciology, 69(273), 13-26. doi:10.1017/jog.2022.41
  - Jouvet, G., Cordonnier, G., Kim, B., Lüthi, M., Vieli, A. and Aschwanden, A. (2022). Deep
    learning speeds up ice flow modelling by several orders of magnitude. Journal of
    Glaciology, 68(270), 651-664. doi:10.1017/jog.2021.120
"""
struct BedRegularization{R <: AbstractSimpleRegularization, F <: AbstractFloat,
    S <: Union{Nothing, AbstractFloat}} <: AbstractRegularization
    reg::R
    convexity::F
    convexity_power::F
    σ_relative::S

    function BedRegularization(;
            σ = nothing, operator = :gradient,
            reg::Union{Nothing, AbstractSimpleRegularization} = nothing,
            convexity = 0.0, convexity_power = 1.3,
            σ_relative = (isnothing(σ) && isnothing(reg)) ? 1.0 : nothing)
        reg = something(reg, TikhonovRegularization(operator = operator, σ = something(σ, 0.3)))
        convexity, convexity_power = promote(float(convexity), float(convexity_power))
        σ_relative = check_σ_relative(σ_relative, reg)
        return new{typeof(reg), typeof(convexity), typeof(σ_relative)}(
            reg, convexity, convexity_power, σ_relative)
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
        # The slope lives on the edge between two cells: only edges with both cells in the mask count
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

# Field the H₀ prior applies to: H₀, or its change from glacier.H₀ (a constant shift, so the derivative is the same)
function regularized_field(lossType::InitialThicknessRegularization, H₀, glacier)
    lossType.correction ? H₀ .- glacier.H₀ : H₀
end

# The prior of a glacier: with `σ_relative`, σ is σ_relative times the roughness of the reference
# field, measured with the same operator and mean as the penalty. The reference is a constant.
function relative_reg(reg, σ_relative, ref, Δx, Δy, mask, normalization)
    isnothing(σ_relative) && return reg
    unit = TikhonovRegularization(operator = reg.operator, σ = 1.0)
    rough = sqrt(loss(unit, ref, Δx, Δy, mask, normalization))
    rough > 0 || throw(ArgumentError(
        "`σ_relative` needs a reference field that is not flat on the glacier."))
    return TikhonovRegularization(operator = reg.operator, σ = σ_relative * rough)
end

"""
    field_roughness(a, glacier; operator = :gradient)

Roughness of the field `a` on `glacier`: the root mean square of its gradient (`operator = :gradient`, in the units of `a` per metre) or of its Laplacian (`operator = :laplacian`, in the
units of `a` per square metre) over the cells of the glacier. It is the quantity that
`σ_relative` multiplies in [`InitialThicknessRegularization`](@ref) and
[`BedRegularization`](@ref).
"""
function field_roughness(a::Matrix{F}, glacier; operator::Symbol = :gradient) where {F <:
                                                                                     AbstractFloat}
    unit = TikhonovRegularization(operator = operator, σ = 1.0)
    return sqrt(loss(unit, a, F(glacier.Δx), F(glacier.Δy), glacier_cells(glacier), one(F)))
end

"""
    prior_σ(regType, glacier)

The `σ` that the prior `regType` (an [`InitialThicknessRegularization`](@ref) or a
[`BedRegularization`](@ref)) uses for `glacier`: the `σ` of its `reg`, or, if `σ_relative` is
set, `σ_relative` times the [`field_roughness`](@ref) of the starting field (`glacier.H₀` or
`glacier.B`).
"""
function prior_σ(regType::InitialThicknessRegularization, glacier)
    return relative_reg(
        regType.reg, regType.σ_relative, glacier.H₀, glacier.Δx, glacier.Δy,
        glacier_cells(glacier), one(eltype(glacier.H₀))).σ
end
function prior_σ(regType::BedRegularization, glacier)
    return relative_reg(regType.reg, regType.σ_relative, glacier.B, glacier.Δx, glacier.Δy,
        glacier_cells(glacier), one(eltype(glacier.B))).σ
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
        reg = @ignore_derivatives relative_reg(
            lossType.reg, lossType.σ_relative, glacier.H₀, Δx, Δy, mask, normalization)
        regH = loss(reg, regularized_field(lossType, H₀, glacier),
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
        # θ.IC is not H₀ itself: H₀ = positive(θ.IC ⋅ scale), with `positive` the
        # `initial_condition_filter` (softplus or Zang). So ∂L/∂θ.IC = ∂L/∂H₀ ⋅ ∂H₀/∂θ.IC
        reg = relative_reg(
            lossType.reg, lossType.σ_relative, glacier.H₀, Δx, Δy, mask, normalization)
        ∂L∂H₀ = backward_loss(reg, regularized_field(lossType, H₀, glacier),
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
    if at_start(t, simulation, glacier_idx)
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        θ_p, lo, hi = gridded_parameter(regType, θ, simulation.parameters.physical)

        key = Symbol("$(glacier_idx)")
        field = @. lo+(hi-lo)*(tanh.(θ_p[key])+1)/2
        mask = @ignore_derivatives glacier_cells_dual(glacier)

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
    if at_start(t, simulation, glacier_idx)
        glacier = simulation.glaciers[glacier_idx]
        Δx, Δy = glacier.Δx, glacier.Δy
        θ_p, lo, hi = gridded_parameter(regType, θ, simulation.parameters.physical)

        key = Symbol("$(glacier_idx)")
        field = @. lo+(hi-lo)*(tanh.(θ_p[key])+1)/2
        mask = @ignore_derivatives glacier_cells_dual(glacier)
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
        reg = @ignore_derivatives relative_reg(regType.reg, regType.σ_relative, glacier.B,
            glacier.Δx, glacier.Δy, cells, normalization)
        R = loss(reg, B, glacier.Δx, glacier.Δy, cells, normalization)
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
discreteLossSteps(lossType::GriddedParameterRegularization, tspan) = [tspan[1]]
discreteLossSteps(lossType::BedRegularization, tspan) = [tspan[1]]
