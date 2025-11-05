export MatrixCacheInterp, feed_input_cache!

"""
    MatrixCacheInterp(nodes_H, nodes_∇S, interp_θ)

A mutable cache structure for storing interpolation data on a 2D grid,
used to efficiently evaluate and reuse interpolated matrices and their gradients.
This interpolation makes complex inversions feasible since it allows the precomputation
of all gradients before the solving the reverse PDE associated to the adjoint variable.

# Fields
- `nodes_H::Vector{Float64}`: Grid nodes corresponding to the first interpolation dimension, typically representing values of ice thickness `H`.
- `nodes_∇S::Vector{Float64}`: Grid nodes corresponding to the second interpolation dimension, typically representing absolute values of slope `∇S`.
- `interp_θ::Interpolations.GriddedInterpolation{Vector{Float64}, 2, Matrix{Vector{Float64}}, Interpolations.Gridded{InterPolations.Linear{InterPolations.Throw{OnGrid}}}, Tuple{Vector{Float64}, Vector{Float64}}}`:
  A gridded linear interpolation object mapping `(nodes_H, nodes_∇S)` to parameter vectors `θ`.
  Used to compute interpolated parameters and enable fast evaluation during repeated model calls.
"""
mutable struct MatrixCacheInterp <: Cache
    # value::Array{Float64, 2}
    # vjp_inp::Array{Float64, 2}
    # vjp_θ::Array{Float64, 3}
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
feed_input_cache! = function (
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
            n_interp_half = simulation.model.machine_learning.target.n_interp_half,
            dilation_factor = 1.05
        )
        ∇S = Huginn.∇slope.([H .+ glacier.B for H in result.H], Ref(glacier.Δx), Ref(glacier.Δy))
        SIA2D_cache.U.nodes_∇S = create_interpolation(
            collect(Iterators.flatten(∇S));
            n_interp_half = simulation.model.machine_learning.target.n_interp_half,
            dilation_factor = 1.05
        )
    end
end