export AbstractTarget, AbstractSIA2DTarget
export SIA2D_target

# Export all functions defined for each custom target type
export Diffusivity, ∂Diffusivity∂H, ∂Diffusivity∂∇H, ∂Diffusivity∂θ
export apply_parametrization, apply_parametrization!

abstract type AbstractTarget end
abstract type AbstractSIA2DTarget <: AbstractTarget end

"""
    SIA2D_target <: AbstractSIA2DTarget

Implementation of Target objective for learning elements of the SIA equation.
"""

### Add specific target objectives
include("target_utils.jl")
include("target_A.jl")
include("target_D_hybrid.jl")
include("target_D_pure.jl")
