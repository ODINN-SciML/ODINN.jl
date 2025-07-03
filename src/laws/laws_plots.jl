export plot_law

function plot_law(
    law::AbstractLaw,
    simulation::Simulation,
    inputs::NamedTuple,
    glacier_idx::Integer,
    θ
)
    n_inputs = length(keys(inputs))
    input_names = sort(collect(keys(inputs)))
    xs = range(law.min_value, law.max_value; length=50)

    if n_inputs == 1
        xname = input_names[1]
        outputs = [eval_law(law, simulation, glacier_idx, merge(inputs, (; (xname)=>x)), θ; scalar=true) for x in xs]
        plot(xs, outputs, xlabel=string(xname), ylabel=string(law.name), title="Law Function Plot")
        return xs, outputs
    elseif n_inputs == 2
        xname, yname = input_names
        xvals = xs
        yvals = xs
        input_values = 
        zs = [eval_law(law, simulation, glacier_idx, merge(inputs, (; (xname)=>x, (yname)=>y)), θ; scalar=true) for x in xvals, y in yvals]
        display(surface(xvals, yvals, permutedims(zs), xlabel=string(xname), ylabel=string(yname), zlabel=string(law.name), title="Law Function Surface"))
        return (xvals, yvals), zs
    else
        error("Only 1D or 2D input plotting is supported.")
    end
end
