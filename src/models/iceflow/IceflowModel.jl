

# Abstract type as a parent type for ice flow models
abstract type IceflowModel end

# Subtype structure for Shallow Ice Approximation models
abstract type SIAmodel <: IceflowModel end

include("SIA2D/SIA2D.jl")