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
"""
mutable struct GriddedInv{
    ComponentVectorType <: ComponentVector
} <: PerGlacierModel
    θ::ComponentVectorType

    function GriddedInv(
        params::Sleipnir.Parameters,
        glaciers::Vector{<: AbstractGlacier},
        var::Symbol,
    )
        inv_param_type = Tuple(Symbol("$(i)") for i in 1:length(glaciers))
        inv_param = NamedTuple{inv_param_type}(
            Tuple(fill(getfield(glaciers[i], var), size(glaciers[i].H₀) .-1) for i in 1:length(glaciers))
        )
        θ = ComponentVector{Sleipnir.Float}(θ = inv_param)

        new{typeof(θ)}(θ)
    end

end

# Display setup
function Base.show(io::IO, invertible_model::GriddedInv)
    println(io, "--- Param to invert ---")
    println(io, "    Matrix per glacier")
    print(io, "    θ: ComponentVector of length $(length(invertible_model.θ))")
end
