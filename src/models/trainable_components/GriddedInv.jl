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

        if var == :A
            minv = isnothing(minval) ? params.physical.minA : minval
            maxv = isnothing(maxval) ? params.physical.maxA : maxval
            inv_param = NamedTuple{inv_param_type}(
                Tuple(fill(getfield(glaciers[i], var), size(glaciers[i].H₀) .- 1)
            for i in 1:length(glaciers))
            )
            θ = ComponentVector{Sleipnir.Float}(θ = inv_param)
            θ = atanh.((θ .- minv) .* (2/(maxv-minv)) .- 1.0)
        elseif var == :C
            # LawC: C = maxC * (tanh(x)+1)/2, inverse x = atanh(C*2/maxC - 1), C ∈ (0, maxC).
            # No prior (glacier.C ≤ 0): seed at C = midC → θ ≈ 0 (max gradient sensitivity);
            # seeding at C ≈ 0 lands in the saturated tanh tail and stalls descent.
            maxv = Sleipnir.Float(isnothing(maxval) ? params.physical.maxC : maxval)
            minv = Sleipnir.Float(isnothing(minval) ? params.physical.minC : minval)
            seed_default = atanh((minv + maxv) / maxv - 1)  # θ for C = (minC+maxC)/2
            inv_param = NamedTuple{inv_param_type}(
                Tuple(
                let c = Sleipnir.Float(getfield(glaciers[i], var))
                    seed = c <= 0 || c >= maxv ? seed_default :
                           atanh(c * 2 / maxv - 1)
                    fill(seed, size(glaciers[i].H₀) .- 1)
                end
            for i in 1:length(glaciers))
            )
            θ = ComponentVector{Sleipnir.Float}(θ = inv_param)
        else
            error("GriddedInv: Only :A or :C are supported for var (got $(var))")
        end

        new{typeof(θ)}(θ)
    end
end

# Display setup
function Base.show(io::IO, invertible_model::GriddedInv)
    println(io, "--- Param to invert ---")
    println(io, "    Matrix per glacier")
    print(io, "    θ: ComponentVector of length $(length(invertible_model.θ))")
end
