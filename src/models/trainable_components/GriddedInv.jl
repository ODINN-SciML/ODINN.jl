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
            # LawC uses C = maxC * (tanh(x)+1)/2, so min is always 0.
            # Inverse: x = atanh(C*2/maxC - 1), valid for C ∈ (0, maxC).
            # For C=0 (no sliding), seed x=-5 → C ≈ 5e-5 * maxC.
            maxv = Sleipnir.Float(isnothing(maxval) ? params.physical.maxC : maxval)
            seeds = [let c = Sleipnir.Float(getfield(glaciers[i], var))
                         c <= 0 || c >= maxv ? Sleipnir.Float(-5) :
                         atanh(c * 2 / maxv - 1)
                     end
                     for i in 1:length(glaciers)]

            # Warn when a glacier starts in the flat tail of the tanh map. C = 0 is the
            # default, and it is also the worst possible starting point for the optimizer:
            # the seed is x = -5, where dC/dx = maxC*sech²(5)/2 is ~7e-5 of its maximum, so
            # the loss is nearly flat in x and the inversion barely moves. That looks like a
            # converged run rather than a failed one -- the loss decreases by a fraction of a
            # percent, LBFGS's line search gives up after a few iterations, and the reported C
            # field is still the seed. Set `glacier.C` to a representative value inside
            # (0, maxC) before inverting; 0.1 * maxC is a reasonable default.
            flat = findall(s -> abs(s) > 4, seeds)
            if !isempty(flat)
                ids = [glaciers[i].rgi_id for i in flat]
                @warn "GriddedInv(:C): $(length(flat))/$(length(seeds)) glaciers have no usable initial C and are seeded in the flat part of the tanh map, where dC/dθ is ~$(round(sech(maximum(abs, seeds[flat]))^2; sigdigits = 2)) of its maximum. The inversion will barely move. Set `glacier.C` inside (0, maxC = $(maxv)), e.g. 0.1 * maxC." rgi_ids=ids
            end

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
