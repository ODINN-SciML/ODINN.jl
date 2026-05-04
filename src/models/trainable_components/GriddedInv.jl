export GriddedInv

"""
    GriddedInv{
        ComponentVectorType <: ComponentVector
    } <: PerGlacierModel

Per glacier invertible parameter container.
`GriddedInv` wraps a ComponentVector (θ) that stores one matrix parameter per glacier and implements the `PerGlacierModel` interface used by the inversion machinery.

# Fields

  - `θ::ComponentVectorType`: The per glacier parameter vector (one matrix per glacier).

# Constructor

    GriddedInv(
        params::Sleipnir.Parameters,
        glaciers::Vector{<: AbstractGlacier},
        var::Symbol,
    )

# Arguments

  - `params::Sleipnir.Parameters`: Parameters struct.
  - `glaciers::Vector{<: AbstractGlacier}`: Vector of AbstractGlacier. The i-th entry in θ corresponds to glaciers[i].
  - `var::Symbol`: Symbol naming the field on each glacier to use as the initial value.

# Example

```julia
GriddedInv(params, glaciers, :A)
```
"""
mutable struct GriddedInv{
    ComponentVectorType <: ComponentVector
} <: PerGlacierModel
    θ::ComponentVectorType

    function GriddedInv(
            params::Sleipnir.Parameters,
            glaciers::Vector{<: AbstractGlacier},
            var::Symbol;
            minval::Union{Nothing, Float64} = nothing,
            maxval::Union{Nothing, Float64} = nothing
    )
        inv_param_type = Tuple(Symbol("$(i)") for i in 1:length(glaciers))
        inv_param = NamedTuple{inv_param_type}(
            Tuple(fill(getfield(glaciers[i], var), size(glaciers[i].H₀) .- 1)
        for i in 1:length(glaciers))
        )
        θ = ComponentVector{Sleipnir.Float}(θ = inv_param)

        # Select bounds based on var
        if var == :A
            minv = isnothing(minval) ? params.physical.minA : minval
            maxv = isnothing(maxval) ? params.physical.maxA : maxval
        elseif var == :C
            minv = isnothing(minval) ? params.physical.minC : minval
            maxv = isnothing(maxval) ? params.physical.maxC : maxval
            minv < maxv ||
                error("GriddedInv: expected minC < maxC, got minC=$(minv), maxC=$(maxv)")
            vals = collect(θ)
            minθ = minimum(vals)
            maxθ = maximum(vals)
            if any(x -> x <= minv || x >= maxv, vals)
                error("[GriddedInv] ERROR: C value out of bounds before atanh mapping! min=$(minθ), max=$(maxθ), allowed ($(minv), $(maxv)). Initialize glacier.C within open bounds first (e.g. via target=:C glacier initialization).")
            end
        else
            error("GriddedInv: Only :A or :C are supported for var (got $(var))")
        end

        θ = atanh.((θ .- minv) .* (2/(maxv-minv)) .- 1.0)

        new{typeof(θ)}(θ)
    end
end

# Display setup
function Base.show(io::IO, invertible_model::GriddedInv)
    println(io, "--- Param to invert ---")
    println(io, "    Matrix per glacier")
    print(io, "    θ: ComponentVector of length $(length(invertible_model.θ))")
end
