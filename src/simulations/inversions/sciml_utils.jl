# This file contains structs and functions to interface with SciML API
# Most of the implementation and the comments come from SciMLStructures' documentation
# Cf https://sciml.github.io/SciMLStructures.jl/stable/example/

import SciMLStructures as SS

import Base # Overload zero and copyto!
export InversionBinder

"""
    InversionBinder{FI <: Inversion, CA <: ComponentArray}

Struct used for the binding with SciMLSensitivity.
It is defined as a SciMLStructure and it contains the inversion structure and the vector of parameters to differentiate.

# Fields

  - `simulation::FI`: Inversion instance.
  - `θ::CA`: ComponentArray that contains the parameters used to differentiate the iceflow.
"""
mutable struct InversionBinder{FI <: Inversion, CA <: ComponentArray} <: Container
    simulation::FI
    θ::CA
end

# Mark the struct as a SciMLStructure
SS.isscimlstructure(::InversionBinder) = true
# It is mutable
SS.ismutablescimlstructure(::InversionBinder) = true

# Only contains `Tunable` portion
# We could also add a `Constants` portion to contain the values that are
# not tunable. The implementation would be similar to this one.
SS.hasportion(::SS.Tunable, ::InversionBinder) = true

function SS.canonicalize(::SS.Tunable, p::InversionBinder)
    # concatenate all tunable values into a single vector
    buffer = ComponentVector2Vector(p.θ)

    # repack takes a new vector of the same length as `buffer`, and constructs
    # a new `InversionBinder` object using the values from the new vector for tunables
    # and retaining old values for other parameters. This is exactly what replace does,
    # so we can use that instead.
    repack = let p = p
        function repack(newbuffer)
            SS.replace(SS.Tunable(), p, newbuffer)
        end
    end
    # the canonicalized vector, the repack function, and a boolean indicating
    # whether the buffer aliases values in the parameter object (here, it doesn't)
    return buffer, repack, false
end

function SS.replace(::SS.Tunable, p::InversionBinder, newbuffer)
    N = length(p.θ)
    @assert length(newbuffer) == N
    sim = deepcopy(p.simulation)
    Enzyme.make_zero!(sim)
    θ = Vector2ComponentVector(newbuffer, p.θ)
    return InversionBinder{typeof(sim), typeof(θ)}(sim, θ)
end

function SS.replace!(::SS.Tunable, p::InversionBinder, newbuffer)
    N = length(p.θ)
    @assert length(newbuffer) == N
    p.θ .= newbuffer
    return p
end

"""
    Base.zero(
        p::InversionBinder{FI, CA},
    ) where {FI <: Inversion, CA <: ComponentArray}

Overload Base.zero as we need a way to copy the SciMLStructure.
It is used in SciMLSensitivity to differentiate the callbacks.
"""
function Base.zero(
        p::InversionBinder{FI, CA},
) where {FI <: Inversion, CA <: ComponentArray}
    return InversionBinder(p.simulation, zero(p.θ))
end

"""
    Base.copyto!(
        dest::InversionBinder{FI, CA},
        src::InversionBinder{FI, CA},
    ) where {FI <: Inversion, CA <: ComponentArray}

Overload Base.copyto! as we need a way to copy the SciMLStructure.
It is used in SciMLSensitivity to differentiate the callbacks.
"""
function Base.copyto!(
        dest::InversionBinder{FI, CA},
        src::InversionBinder{FI, CA}
) where {FI <: Inversion, CA <: ComponentArray}
    dest.θ .= src.θ
end
