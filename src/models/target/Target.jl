export AbstractTarget, AbstractSIA2DTarget

# Export all functions defined for each custom target type
export Diffusivity, ∂Diffusivity∂H, ∂Diffusivity∂∇H, ∂Diffusivity∂θ

abstract type AbstractTarget end

"""
    AbstractSIA2DTarget <: AbstractTarget

Abstract type for inversion targets that learn elements of the SIA2D ice flow
equation (e.g. the creep coefficient `A` or the diffusivity `D`). Concrete subtypes
must implement `Diffusivity`, `∂Diffusivity∂H`, `∂Diffusivity∂∇H` and
`∂Diffusivity∂θ`, which are used by the manual adjoint computations.
"""
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
