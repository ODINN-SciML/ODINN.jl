export InitialCondition

# Empty initial condition
struct emptyIC <: PerGlacierModel end

"""
    InitialCondition{
        ComponentVectorType <: ComponentVector,
        F <: AbstractFloat
    } <: PerGlacierModel

Per glacier initial condition container: the initial geometry of each glacier that the
inversion finds.

With `bed = :free` (default), the initial geometry is inverted. Its two free fields are the ice
thickness `H₀` and the surface `S₀` at the start, as in IGM, and the bed follows,
`B = S₀ - H₀`. The bed is the same at all times. `S₀` starts at the surface of the glacier
(`glacier.S`, the DEM), so a [`LossS`](@ref) with a DEM at the start of the time window keeps
it close to the data. The bed is free only with `SciMLSensitivityAdjoint`.

With `bed = :fixed`, only `H₀` is inverted, the bed is `glacier.B`, and the surface moves with
`H₀`.

The outline of the glacier is a hard constraint. It is the set of cells where the starting
thickness `glacier.H₀` is not zero (the glacier at the date of the DEM). There, the ice thickness
`H₀` is at least `min_thickness`. Outside the outline `H₀` is zero, also in the zone around the
outline where the glacier is allowed to advance during the run. Only `S₀` is free there. A free
zone around the outline was tested and is not used: see [Inversion of the bed](@ref).

For each glacier `id`, `θ.IC[id].H` holds `H₀` divided by `scale_H`, and `θ.IC[id].S` holds the
change of the surface `S₀ - glacier.S` divided by `scale_S`. The scales set the size of a step
of the optimizer in metres (step = learning rate × scale). Use `ODINN.initial_thickness(θ, simulation, id)` and `Huginn.bed(simulation, id, θ)` to get metres.

# Fields

  - `θ::ComponentVectorType`: The per glacier parameters of the initial condition.
  - `bed::Symbol`: `:free` or `:fixed`.
  - `scale_H::F`: Scale of `H₀` in metres for all glaciers. `0.0` means the maximum initial
    thickness of each glacier (`H₀_scale`).
  - `scale_S::F`: Scale of `S₀` in metres.
  - `min_thickness::F`: Minimum of `H₀` in metres inside the outline of the glacier.

# Constructor

    InitialCondition(
        params::Sleipnir.Parameters,
        glaciers::Vector{<: AbstractGlacier},
        initialization::Symbol = :Farinotti19;
        bed::Symbol = :free,
        scale_H = nothing,
        scale_S = 5.0,
        min_thickness = 1.0,
    )

# Arguments

  - `params::Sleipnir.Parameters`: Parameters struct.
  - `glaciers::Vector{<: AbstractGlacier}`: Vector of AbstractGlacier. The i-th entry in θ corresponds to glaciers[i].
  - `initialization::Symbol`: Symbol providing the way the initial condition should be initialized.
  - `bed::Symbol`: `:free` (default) to invert the bed with the initial thickness, `:fixed` to
    keep the bed of the glacier.
  - `scale_H`: Scale of `H₀` in metres. With `nothing` (default), it is the maximum initial
    thickness of each glacier (`H₀_scale`).
  - `scale_S`: Scale of `S₀` in metres. Default is `5.0`.
  - `min_thickness`: Minimum of `H₀` in metres inside the outline of the glacier. Default is `1.0`. It needs `initial_condition_filter = :Zang1980` (default) or `:softplus` in `UDEparameters`; it is not used with `:identity`.
    It is used with the `:Zang1980` and `:softplus` filters. Use `0.0` for no minimum.

# Example

```julia
InitialCondition(params, glaciers, :Millan22)                # bed = :free
InitialCondition(params, glaciers, :Farinotti19; bed = :fixed)
```
"""
mutable struct InitialCondition{
    ComponentVectorType <: ComponentVector,
    F <: AbstractFloat
} <: PerGlacierModel
    θ::ComponentVectorType
    bed::Symbol
    # A plain number, 0 for the default: no `Union{Nothing, F}`, as Enzyme copies the model.
    # It is not a vector with one scale per glacier because each batch has one glacier, with
    # index 1, and shares this object.
    scale_H::F
    scale_S::F
    min_thickness::F

    function InitialCondition(
            params::Sleipnir.Parameters,
            glaciers::Vector{<: AbstractGlacier},
            initialization::Symbol = :Farinotti19;
            bed::Symbol = :free,
            scale_H = nothing,
            scale_S = 5.0,
            min_thickness = 1.0
    )
        bed in (:free, :fixed) || throw(ArgumentError(
            "The bed of InitialCondition must be :free or :fixed, got :$(bed)."))
        # Float type
        ft = Sleipnir.Float
        # 0 is the default scale, the maximum initial thickness of each glacier (`H₀_scale`)
        scale_H = isnothing(scale_H) ? zero(ft) : ft(scale_H)
        scale_H >= 0 || throw(ArgumentError("scale_H must be positive, got $(scale_H)."))
        scale_S = ft(scale_S)
        min_thickness = ft(min_thickness)
        min_thickness >= 0 || throw(ArgumentError(
            "min_thickness must be zero or positive, got $(min_thickness)."))
        # Warn early: without a filter, H₀ can be negative and `min_thickness` is not used
        params.UDE.initial_condition_filter == :identity &&
            @warn "`initial_condition_filter = \
:identity`: H₀ can become negative, and `min_thickness` is not used. Use \
`:Zang1980` (default) or `:softplus` in `UDEparameters`."
        # Scaling factor
        gridScalingFactor = params.simulation.gridScalingFactor
        # Component Array type
        initial_condition_type = Tuple(Symbol("$(i)") for i in 1:length(glaciers))

        # Define a series of initial conditions
        if initialization == :Farinotti19
            Hs = map(1:length(glaciers)) do i
                H₀ = farinotti19_thickness(glaciers[i].rgi_id, params)
                Sleipnir.fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
                if gridScalingFactor > 1
                    H₀ = Sleipnir.block_average_pad_edge(H₀, gridScalingFactor)
                end
                H₀
            end
            initial_condition = NamedTuple{initial_condition_type}(Tuple(Hs))
        elseif initialization == :Farinotti19Random
            stdH = 10.0
            grid_length = 10
            initial_condition = NamedTuple{initial_condition_type}(
                Tuple(random_matrix(glaciers[i].H₀, stdH, grid_length)
            for i in 1:length(glaciers))
            )
        elseif initialization == :Millan22
            Hs = map(1:length(glaciers)) do i
                H₀ = millan22_thickness(glaciers[i].rgi_id, params)
                Sleipnir.fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
                if gridScalingFactor > 1
                    H₀ = Sleipnir.block_average_pad_edge(H₀, gridScalingFactor)
                end
                H₀
            end
            initial_condition = NamedTuple{initial_condition_type}(Tuple(Hs))
        else
            @error "Strategy for initialization of ice thicknesses not found."
        end

        # θ.IC is stored divided by the scales, as evaluate_H₀ and evaluate_∂H₀ expect
        initial_condition = NamedTuple{initial_condition_type}(Tuple(
            begin
                sH = scale_H > 0 ? scale_H : H₀_scale(glaciers[i])
                bed == :free ? (H = H ./ sH, S = zero(H)) : (H = H ./ sH,)
            end
        for (i, H) in enumerate(initial_condition)))
        θ = ComponentVector{ft}(θ = initial_condition)
        new{typeof(θ), ft}(θ, bed, scale_H, scale_S, min_thickness)
    end
end

# Display setup
function Base.show(io::IO, invertible_model::InitialCondition)
    println(io, "--- Initial condition to invert ---")
    println(io, "    Initial geometry per glacier, bed $(invertible_model.bed)")
    print(io, "    θ: ComponentVector of length $(length(invertible_model.θ))")
end

include("./InitialCondition_utils.jl")
