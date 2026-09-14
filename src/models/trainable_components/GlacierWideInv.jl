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
        var::Symbol;
        minval::Union{Nothing, Float64} = nothing,
        maxval::Union{Nothing, Float64} = nothing,
    )

# Arguments

  - `params::Sleipnir.Parameters`: Parameters struct.
  - `glaciers::Vector{<: AbstractGlacier}`: Vector of AbstractGlacier. The i-th entry in θ corresponds to glaciers[i].
  - `var::Symbol`: Symbol naming the field on each glacier to use as the initial value. Only `:A` and `:C` are supported.
  - `minval::Union{Nothing, Float64}`: Lower bound of the parameterization. Defaults to `params.physical.minA` for `:A`; unused for `:C`, whose lower bound is always zero.
  - `maxval::Union{Nothing, Float64}`: Upper bound of the parameterization. Defaults to `params.physical.maxA` for `:A` and `params.physical.maxC` for `:C`.

# Example

```julia
GlacierWideInv(params, glaciers, :A)
GlacierWideInv(params, glaciers, :C)
```
"""
mutable struct GlacierWideInv{
    ComponentVectorType <: ComponentVector
} <: PerGlacierModel
    θ::ComponentVectorType

    function GlacierWideInv(
            params::Sleipnir.Parameters,
            glaciers::Vector{<: AbstractGlacier},
            var::Symbol;
            minval::Union{Nothing, Float64} = nothing,
            maxval::Union{Nothing, Float64} = nothing
    )
        # Inverse of the tanh parameterization used by the corresponding law, so that θ₀
        # maps back to the glacier's initial value. LawA spans [minA, maxA], LawC [0, maxC].
        unparameterize = if var == :A
            minv = isnothing(minval) ? params.physical.minA : minval
            maxv = isnothing(maxval) ? params.physical.maxA : maxval
            v -> atanh(2 * (v - minv) / (maxv - minv) - 1)
        elseif var == :C
            maxv = isnothing(maxval) ? params.physical.maxC : maxval
            # C = 0 (no sliding) would give atanh(-1) = -Inf, so seed it just inside the bound
            v -> 0 < v < maxv ? atanh(2 * v / maxv - 1) : Sleipnir.Float(-5)
        else
            error("GlacierWideInv: only :A and :C are supported for var (got $(var))")
        end

        glacier_keys = Tuple(Symbol("$(i)") for i in 1:length(glaciers))
        θ₀ = Tuple(
            fill(unparameterize(Sleipnir.Float(getfield(g, var)))) for g in glaciers)
        θ = ComponentVector{Sleipnir.Float}(θ = NamedTuple{glacier_keys}(θ₀))

        new{typeof(θ)}(θ)
    end
end

# Display setup
function Base.show(io::IO, invertible_model::GlacierWideInv)
    println(io, "--- Param to invert ---")
    println(io, "    Scalar value per glacier")
    print(io, "    θ: ComponentVector of length $(length(invertible_model.θ))")
end
