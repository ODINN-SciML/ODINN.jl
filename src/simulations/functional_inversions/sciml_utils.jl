# This file contains structs and functions to interface with SciML API
# Most of the implementation and the comments come from SciMLStructures' documentation
# Cf https://sciml.github.io/SciMLStructures.jl/stable/example/

import SciMLStructures as SS

import Base # Overload zero and copyto!
export FunctionalInversionBinder

"""
    FunctionalInversionBinder{FI <: FunctionalInversion, CA <: ComponentArray}

Struct used for the binding with SciMLSensitivity.
It is defined as a SciMLStructure and it contains the functional inversion structure and the vector of parameters to differentiate.

# Fields
- `simulation::FI`: Functional inversion instance.
- `θ::CA`: ComponentArray that contains the parameters used to differentiate the iceflow.
"""
mutable struct FunctionalInversionBinder{FI <: FunctionalInversion, CA <: ComponentArray} <: Container
    simulation::FI
    θ::CA
end


# Mark the struct as a SciMLStructure
SS.isscimlstructure(::FunctionalInversionBinder) = true
# It is mutable
SS.ismutablescimlstructure(::FunctionalInversionBinder) = true

# Only contains `Tunable` portion
# We could also add a `Constants` portion to contain the values that are
# not tunable. The implementation would be similar to this one.
SS.hasportion(::SS.Tunable, ::FunctionalInversionBinder) = true

function SS.canonicalize(::SS.Tunable, p::FunctionalInversionBinder)
    # concatenate all tunable values into a single vector
    buffer = ComponentVector2Vector(p.θ)

    # repack takes a new vector of the same length as `buffer`, and constructs
    # a new `FunctionalInversionBinder` object using the values from the new vector for tunables
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

function SS.replace(::SS.Tunable, p::FunctionalInversionBinder, newbuffer)
    N = length(p.θ)
    @assert length(newbuffer) == N
    sim = deepcopy(p.simulation)
    Enzyme.make_zero!(sim)
    θ = Vector2ComponentVector(newbuffer, p.θ)
    return FunctionalInversionBinder{typeof(sim), typeof(θ)}(sim, θ)
end

function SS.replace!(::SS.Tunable, p::FunctionalInversionBinder, newbuffer)
    N = length(p.θ)
    @assert length(newbuffer) == N
    p.θ .= newbuffer
    return p
end


"""
    Base.zero(
        p::FunctionalInversionBinder{FI, CA},
    ) where {FI <: FunctionalInversion, CA <: ComponentArray}

Overload Base.zero as we need a way to copy the SciMLStructure.
It is used in SciMLSensitivity to differentiate the callbacks.
"""
function Base.zero(
    p::FunctionalInversionBinder{FI, CA},
) where {FI <: FunctionalInversion, CA <: ComponentArray}
    return FunctionalInversionBinder(p.simulation, zero(p.θ))
end

"""
    Base.copyto!(
        dest::FunctionalInversionBinder{FI, CA},
        src::FunctionalInversionBinder{FI, CA},
    ) where {FI <: FunctionalInversion, CA <: ComponentArray}

Overload Base.copyto! as we need a way to copy the SciMLStructure.
It is used in SciMLSensitivity to differentiate the callbacks.
"""
function Base.copyto!(
    dest::FunctionalInversionBinder{FI, CA},
    src::FunctionalInversionBinder{FI, CA},
) where {FI <: FunctionalInversion, CA <: ComponentArray}
    dest.θ .= src.θ
end
