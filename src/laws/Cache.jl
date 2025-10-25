export MatrixCacheInterp

"""

Define interpolation cache for D inversion
"""
mutable struct MatrixCacheInterp <: Cache
    value::Array{Float64, 2}
    vjp_inp::Array{Float64, 2}
    vjp_θ::Array{Float64, 3}
    # I am sure this can be done better, for now we stote this like this!
    nodes_H::Array{Float64, 1}
    nodes_∇S::Array{Float64, 1}
    # interp_θ::Interpolations.GriddedInterpolation{
    #     Vector{Float64},
    #     1,
    #     Vector{Vector{Float64}},
    #     Interpolations.Gridded{
    #         Interpolations.Linear{
    #             Interpolations.Throw{OnGrid}
    #             }
    #         },
    #     Tuple{Vector{Float64}}
    #     }
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