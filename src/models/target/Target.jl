export AbstractTarget, AbstractSIA2DTarget

# Export all functions defined for each custom target type
export Diffusivity, ∂Diffusivity∂H, ∂Diffusivity∂∇H, ∂Diffusivity∂θ

abstract type AbstractTarget end
abstract type AbstractSIA2DTarget <: AbstractTarget end

"""
    SIA2D_target <: AbstractSIA2DTarget

Implementation of Target objective for learning elements of the SIA equation.
"""

### Add specific target objectives
include("target_utils.jl")
include("target_A.jl")
include("target_C.jl")
include("target_D_hybrid.jl")
include("target_D_pure.jl")
