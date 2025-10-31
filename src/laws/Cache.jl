export MatrixCacheInterp, feed_input_cache!

"""

Define interpolation cache for D inversion
"""
mutable struct MatrixCacheInterp <: Cache
    value::Array{Float64, 2}
    vjp_inp::Array{Float64, 2}
    vjp_θ::Array{Float64, 3}
    # TODO: I am sure this can be done better, for now we stote this like this!
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
Docs
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
        # forwar simulation
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