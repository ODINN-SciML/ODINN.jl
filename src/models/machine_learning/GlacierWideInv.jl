export GlacierWideInv

"""
    GlacierWideInv{
        ComponentVectorType <: ComponentVector
    } <: PerGlacierModel

Per glacier invertible parameter container.
`GlacierWideInv` wraps a ComponentVector (θ) that stores one scalar parameter per glacier and implements the `PerGlacierModel` interface used by the inversion machinery.

# Fields
- `θ::ComponentVectorType`: The per glacier parameter vector (one scalar value per glacier).

# Constructor

    GlacierWideInv(
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
GlacierWideInv(params, glaciers, :A)
```
"""
mutable struct GlacierWideInv{
    ComponentVectorType <: ComponentVector
} <: PerGlacierModel
    θ::ComponentVectorType

    function GlacierWideInv(
        params::Sleipnir.Parameters,
        glaciers::Vector{<: AbstractGlacier},
        var::Symbol,
    )
        inv_param_type = Tuple(Symbol("$(i)") for i in 1:length(glaciers))
        inv_param = NamedTuple{inv_param_type}(
            Tuple(fill(getfield(glaciers[i], var)) for i in 1:length(glaciers))
        )
        θ = ComponentVector{Sleipnir.Float}(θ = inv_param)

        # Invert parameterization
        minA = params.physical.minA
        maxA = params.physical.maxA
        θ = atanh.((θ .- minA).*(2/(maxA-minA)) .- 1.0)

        new{typeof(θ)}(θ)
    end

end

# Display setup
function Base.show(io::IO, invertible_model::GlacierWideInv)
    println(io, "--- Param to invert ---")
    println(io, "    Scalar value per glacier")
    print(io, "    θ: ComponentVector of length $(length(invertible_model.θ))")
end
