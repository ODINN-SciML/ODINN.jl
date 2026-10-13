export GriddedInv

"""
    C_seed(c, maxC)

Initial value of `θ.C` for a sliding coefficient `c`: the inverse of the map of `LawC`,
`C = maxC * (tanh(x) + 1) / 2`, valid for `c` in `(0, maxC)`. `c = 0` (no sliding) is seeded at
`x = -5` (`C ≈ 5e-5 maxC`), and `c ≥ maxC` at `x = 5`.
"""
function C_seed(c, maxC)
    c <= 0 && return Sleipnir.Float(-5)
    c >= maxC && return Sleipnir.Float(5)
    return atanh(c * 2 / maxC - 1)
end

# Warn when a C inversion starts in the flat tails of the tanh map (C = 0 is the default):
# dC/dx is ~2e-4 of its maximum there, so the inversion barely moves, which looks like a
# converged run
function warn_flat_C_seeds(seeds, glaciers, maxC, model_name)
    flat = findall(s -> abs(s) > 4, seeds)
    isempty(flat) && return nothing
    rgi_ids = [glaciers[i].rgi_id for i in flat]
    @warn "$(model_name)(:C): $(length(flat))/$(length(seeds)) glaciers have no usable initial C and are seeded in the flat part of the tanh map, where dC/dθ is ~$(round(sech(maximum(abs, seeds[flat]))^2; sigdigits = 2)) of its maximum. The inversion will barely move. Set `glacier.C` inside (0, maxC = $(maxC)), e.g. 0.1 * maxC." rgi_ids
end

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
            maxv = Sleipnir.Float(isnothing(maxval) ? params.physical.maxC : maxval)
            seeds = [C_seed(Sleipnir.Float(g.C), maxv) for g in glaciers]
            warn_flat_C_seeds(seeds, glaciers, maxv, "GriddedInv")

            inv_param = NamedTuple{inv_param_type}(
                Tuple(fill(seeds[i], size(glaciers[i].H₀) .- 1)
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
