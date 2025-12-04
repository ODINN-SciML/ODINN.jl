import Base.similar, Base.size

export MatrixCacheInterp
export ScalarCacheGlacierId, MatrixCacheGlacierId

"""
    MatrixCacheInterp(nodes_H, nodes_∇S, interp_θ)

A mutable cache structure for storing interpolation data on a 2D grid,
used to efficiently evaluate and reuse interpolated matrices and their gradients.
This interpolation makes complex inversions feasible since it allows the precomputation
of all gradients before the solving the reverse PDE associated to the adjoint variable.

# Fields

  - `value::Array{Float64, 2}`: Matrix to store value of the diffusivity.
  - `nodes_H::Vector{Float64}`: Grid nodes corresponding to the first interpolation dimension, typically representing values of ice thickness `H`.
  - `nodes_∇S::Vector{Float64}`: Grid nodes corresponding to the second interpolation dimension, typically representing absolute values of slope `∇S`.
  - `interp_θ::Interpolations.GriddedInterpolation{Vector{Float64}, 2, Matrix{Vector{Float64}}, Interpolations.Gridded{InterPolations.Linear{InterPolations.Throw{OnGrid}}}, Tuple{Vector{Float64}, Vector{Float64}}}`:
    A gridded linear interpolation object mapping `(nodes_H, nodes_∇S)` to parameter vectors `θ`.
    Used to compute interpolated parameters and enable fast evaluation during repeated model calls.
"""
mutable struct MatrixCacheInterp <: Cache
    value::Array{Float64, 2}
    nodes_H::Array{Float64, 1}
    nodes_∇S::Array{Float64, 1}
    interp_θ::Interpolations.GriddedInterpolation{
        Vector{Float64},
        2,
        Matrix{Vector{Float64}},
        Interpolations.Gridded{
            Interpolations.Linear{
            Interpolations.Throw{OnGrid}
        }
        },
        Tuple{Vector{Float64}, Vector{Float64}}
    }
end

"""
    ScalarCacheGlacierId <: Cache

A cache structure for storing a scalar value as a zero-dimensional array of `Float64` along with
their associated vector-Jacobian products (VJP).
It also stores the glacier ID as an integer.
This is typically used to invert a single scalar per glacier.
Fields:

  - `value::Array{Float64, 0}`: The cached scalar value.
  - `vjp_inp::Array{Float64, 0}`: VJP with respect to inputs. Must be defined but never used in
    practice since this cache is used for classical inversions and the law does not have inputs.
  - `vjp_θ::Vector{Float64}`: VJP with respect to parameters.
  - `glacier_id::Int64`: Glacier ID in the list of simulated glaciers.
"""
struct ScalarCacheGlacierId <: Cache
    value::Array{Float64, 0}
    vjp_inp::Array{Float64, 0}
    vjp_θ::Vector{Float64}
    glacier_id::Int64
end
function similar(c::ScalarCacheGlacierId)
    typeof(c)(similar(c.value), similar(c.vjp_inp), similar(c.vjp_θ), c.glacier_id)
end
size(c::ScalarCacheGlacierId) = size(c.value)
function Base.:(==)(a::ScalarCacheGlacierId, b::ScalarCacheGlacierId)
    a.value == b.value && a.vjp_inp == b.vjp_inp && a.vjp_θ == b.vjp_θ &&
        a.glacier_id == b.glacier_id
end

"""
    MatrixCacheGlacierId <: Cache

A cache structure for storing a matrix value (`Float64` 2D array) along with
their associated vector-Jacobian products (VJP).
It also stores the glacier ID as an integer.
This is typically used to invert a spatially varying field per glacier.
Fields:

  - `value::Array{Float64, 2}`: The cached matrix value.
  - `vjp_inp::Array{Float64, 2}`: VJP with respect to inputs.
  - `vjp_θ::Vector{Float64}`: VJP with respect to parameters.
  - `glacier_id::Int64`: Glacier ID in the list of simulated glaciers.
"""
struct MatrixCacheGlacierId <: Cache
    value::Array{Float64, 2}
    vjp_inp::Array{Float64, 2}
    vjp_θ::Vector{Float64}
    glacier_id::Int64
end
function similar(c::MatrixCacheGlacierId)
    typeof(c)(similar(c.value), similar(c.vjp_inp), similar(c.vjp_θ), c.glacier_id)
end
size(c::MatrixCacheGlacierId) = size(c.value)
function Base.:(==)(a::MatrixCacheGlacierId, b::MatrixCacheGlacierId)
    a.value == b.value && a.vjp_inp == b.vjp_inp && a.vjp_θ == b.vjp_θ &&
        a.glacier_id == b.glacier_id
end

"""
    feed_input_cache!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation,
        glacier_idx::Integer,
        θ,
        result
    )

Populate the input cache of an `SIA2DCache` instance with interpolation nodes for
ice thickness (`H`) and surface slope (`∇S`), based on the results of a previous
forward simulation. This function is required just when results of the forward pass
are required to evaluate the elements of the cache in the reverse step.

This function prepares the interpolation knots used later by the reverse evaluation of the
adjoint SIA2D model.

Right now, this function is just required for the inversion w.r.t to D, which is
indicated by the boolean variable `SIA2D_model.U_is_provided`. Other inversions may
not required the definition of this function.

# Arguments

  - `SIA2D_model::SIA2Dmodel`: The 2D shallow-ice approximation model instance.
  - `SIA2D_cache::SIA2DCache`: The cache object that stores precomputed interpolation nodes.
  - `simulation`: The simulation object containing glacier configurations and model settings.
  - `glacier_idx::Integer`: Index of the glacier within `simulation.glaciers` for which the cache is being populated.
  - `θ`: Model parameters (not directly used in this function but included for interface consistency).
  - `result`: Output of a previous forward run containing ice thickness fields `H`.
"""
function feed_input_cache!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation,
        glacier_idx::Integer,
        θ,
        result
)
    glacier = simulation.glaciers[glacier_idx]
    if SIA2D_model.U_is_provided
        # We create the interpolation based on all the values of H and ∇S seen during the
        # forward simulation
        SIA2D_cache.U.nodes_H = create_interpolation(
            collect(Iterators.flatten(result.H));
            n_interp_half = simulation.model.trainable_components.target.n_interp_half,
            dilation_factor = 1.05,
            minA_quantile = 10.0 # We start the quantile count at H = 10m to avoid very small values near zero
        )
        ∇S = Huginn.∇slope.([H .+ glacier.B for H in result.H], Ref(glacier.Δx), Ref(glacier.Δy))
        SIA2D_cache.U.nodes_∇S = create_interpolation(
            collect(Iterators.flatten(∇S));
            n_interp_half = simulation.model.trainable_components.target.n_interp_half,
            dilation_factor = 1.05
        )
    end
end
