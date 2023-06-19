
@kwdef mutable struct Results{F <: AbstractFloat} 
    output::Vector{F}
end


function Results(;
        output::Vector{Float64} = []
            )

    # Build the results struct based on input values
    results = Results(output = output)

    return results
end