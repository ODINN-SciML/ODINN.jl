
export SIA2Dmodel

###############################################
###### SHALLOW ICE APPROXIMATION MODELS #######
###############################################

mutable struct SIA2Dmodel{F <: AbstractFloat, I <: Int} <: SIAmodel
    A::Union{Ref{F}, Nothing}
    H::Union{Matrix{F}, Nothing}
    S::Union{Matrix{F}, Nothing}
    dSdx::Union{Matrix{F}, Nothing}
    dSdy::Union{Matrix{F}, Nothing}
    D::Union{Matrix{F}, Nothing}
    dSdx_edges::Union{Matrix{F}, Nothing}
    dSdy_edges::Union{Matrix{F}, Nothing}
    ∇S::Union{Matrix{F}, Nothing}
    Fx::Union{Matrix{F}, Nothing}
    Fy::Union{Matrix{F}, Nothing}
    V::Union{Matrix{F}, Nothing}
    Vx::Union{Matrix{F}, Nothing}
    Vy::Union{Matrix{F}, Nothing}
    Γ::Union{Ref{F}, Nothing}
    MB::Union{Matrix{F}, Nothing}
    MB_mask::Union{BitMatrix, Nothing}
    MB_total::Union{Matrix{F}, Nothing}
    glacier_idx::Union{Ref{I}, Nothing}
end

function SIA2Dmodel(params::Parameters; 
                    A::Union{Ref{F}, Nothing} = nothing,
                    H::Union{Matrix{F}, Nothing} = nothing,
                    S::Union{Matrix{F}, Nothing} = nothing,
                    dSdx::Union{Matrix{F}, Nothing} = nothing,
                    dSdy::Union{Matrix{F}, Nothing} = nothing,
                    D::Union{Matrix{F}, Nothing} = nothing,
                    dSdx_edges::Union{Matrix{F}, Nothing} = nothing,
                    dSdy_edges::Union{Matrix{F}, Nothing} = nothing,
                    ∇S::Union{Matrix{F}, Nothing} = nothing,
                    Fx::Union{Matrix{F}, Nothing} = nothing,
                    Fy::Union{Matrix{F}, Nothing} = nothing,
                    V::Union{Matrix{F}, Nothing} = nothing,
                    Vx::Union{Matrix{F}, Nothing} = nothing,
                    Vy::Union{Matrix{F}, Nothing} = nothing,
                    Γ::Union{Ref{F}, Nothing} = nothing,
                    MB::Union{Matrix{F}, Nothing} = nothing,
                    MB_mask::Union{BitMatrix, Nothing} = nothing,
                    MB_total::Union{Matrix{F}, Nothing} = nothing,
                    glacier_idx::Union{Ref{I}, Nothing} = nothing) where {F <: AbstractFloat, I <: Int}
    
    ft = params.simulation.float_type
    it = params.simulation.int_type
    SIA2D_model = SIA2Dmodel{ft,it}(A, H, S, dSdx, dSdy, D, dSdx_edges, dSdy_edges,
                            ∇S, Fx, Fy, V, Vx, Vy, Γ, MB, MB_mask, MB_total, glacier_idx)

    return SIA2D_model
end

"""
    initialize_iceflow_model!(; glacier::Glacier,
                                params::Parameters
                                ) where F <: AbstractFloat

Initialize iceflow model data structures to enable in-place mutation.

Keyword arguments
=================
    - `glacier`: `Glacier` to provide basic initial state of the ice flow model.
    - `parameters`: `Parameters` to configure some physical variables
"""
function initialize_iceflow_model!(iceflow_model::IF,  
                                    glacier_idx::I,
                                    glacier::Glacier,
                                     params::Parameters
                                     ) where {IF <: IceflowModel, I <: Int}
    nx, ny = glacier.nx, glacier.ny
    F = params.simulation.float_type
    iceflow_model.A = Ref{F}(params.physical.A)
    iceflow_model.H = deepcopy(glacier.H₀)::Matrix{F}
    iceflow_model.S = deepcopy(glacier.S)::Matrix{F}
    iceflow_model.dSdx = zeros(F,nx-1,ny)
    iceflow_model.dSdy= zeros(F,nx,ny-1)
    iceflow_model.D = zeros(F,nx-1,ny-1)
    iceflow_model.dSdx_edges = zeros(F,nx-1,ny-2)
    iceflow_model.dSdy_edges = zeros(F,nx-2,ny-1) 
    iceflow_model.∇S = zeros(F,nx-1,ny-1)
    iceflow_model.Fx = zeros(F,nx-1,ny-2)
    iceflow_model.Fy = zeros(F,nx-2,ny-1)
    iceflow_model.V = zeros(F,nx-1,ny-1)
    iceflow_model.Vx = zeros(F,nx-1,ny-1)
    iceflow_model.Vy = zeros(F,nx-1,ny-1)
    iceflow_model.Γ = Ref{F}(0.0)
    iceflow_model.MB = zeros(F,nx,ny)
    iceflow_model.MB_mask= zeros(F,nx,ny)
    iceflow_model.MB_total = zeros(F,nx,ny)
    iceflow_model.glacier_idx = Ref{I}(glacier_idx)
end



