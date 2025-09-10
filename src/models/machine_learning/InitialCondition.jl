export InitialCondition

include("./InitialCondition_utils.jl")

# Empty initial condition
struct emptyIC <: MLmodel end
"""

"""
mutable struct InitialCondition{
    ComponentVectorType <: ComponentVector
    } <: MLmodel
    θ::ComponentVectorType

    function InitialCondition(
        params::P,
        glaciers::Vector{G},
        initialization::Symbol = :Farinotti2019,
    ) where {
        P<:Sleipnir.Parameters,
        G<:AbstractGlacier
    }
        # Float type
        ft = Sleipnir.Float

        # TODO: we do it for a single glacier now
        if initialization == :Farinotti2019
            θ = ComponentVector{ft}(θ = glaciers[1].H₀)
            # θ = ComponentVector{ft}(θ = [glacier.H₀ for glacier in glaciers])
        elseif initialization == :Farinotti2019Random
            stdH = 10.0
            grid_length = 10
            H₀ = random_matrix(glaciers[1].H₀, stdH, grid_length)
            θ = ComponentVector{ft}(θ = H₀)
        else
            @error "Strategy for initialization of ice thicknesses not found."
        end

        new{typeof(θ)}(
            θ
        )
    end

end