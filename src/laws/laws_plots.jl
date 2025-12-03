export plot_law

function save_law_plot(fig, n_inputs, input_names, law::AbstractLaw, simulation::Simulation, idx_fixed_input=0)
    # Build filename based on law name, input names, and fixed input info
    filename = "law_plot_" * string(law.name)
    if n_inputs == 1
        filename *= "_" * string(input_names[1])
    elseif n_inputs == 2
        if idx_fixed_input != 0
            fixed_name = string(input_names[idx_fixed_input])
            filename *= "_fixed_" * fixed_name * "_" * string(input_names[3-idx_fixed_input])
        else
            filename *= "_" * string(input_names[1]) * "_" * string(input_names[2])
        end
    end

    folder = joinpath(simulation.parameters.simulation.working_dir, "laws")
    mkpath(folder)
    filepath = joinpath(folder, filename * ".pdf")
    @info "Saving law plot to $filepath"
    Plots.savefig(fig, filepath)
end

"""
    plot_law(law::AbstractLaw, simulation::Simulation, inputs::NamedTuple, θ; glacier_idx=1, idx_fixed_input=0)

Plot a law function with one or two input variables.

# Arguments
- `law::AbstractLaw`: The law to plot (e.g., a sliding law or creep law).
- `simulation::Simulation`: The simulation containing glaciers and parameters.
- `inputs::NamedTuple`: Named tuple of input variables for the law (e.g., `(T=iTemp(), H̄=iH̄())`).
- `θ`: Parameters for the law (can be `nothing` for laws without parameters).

# Keyword Arguments
- `glacier_idx::Integer=1`: Index of the glacier to use for extracting input values (for 2D inputs).
- `idx_fixed_input::Integer=0`: For two-input laws, index (1 or 2) of the input to fix at its mean value. If `0`, plots a 3D surface.

# Returns
- A plot figure (1D line plot, 2D scatter, or 3D surface) and saves it to the simulation's working directory.

# Examples
```julia
# 1D plot (temperature input for Cuffey-Paterson law)
plot_law(A_law, simulation, (T=iTemp(),), nothing)
```
"""
function plot_law(
    law::AbstractLaw,
    simulation::Simulation,
    inputs::NamedTuple,
    θ;
    glacier_idx::Integer = 1,
    idx_fixed_input::Integer = 0,
    plot_full_input_range::Bool = false
)
    n_inputs = length(keys(inputs))
    input_names = sort(collect(keys(inputs)))

    if n_inputs == 1
        fig = plot_law_1d(law, simulation, inputs, glacier_idx, θ, input_names[1], plot_full_input_range)
        save_law_plot(fig, n_inputs, input_names, law, simulation, idx_fixed_input)
    elseif n_inputs == 2
        fig = plot_law_2d(law, simulation, inputs, glacier_idx, θ, input_names, idx_fixed_input)
    else
        error("Only 1D or 2D input plotting is supported.")
    end

    return fig
end

function plot_law_1d(
    law::AbstractLaw,
    simulation::Simulation,
    inputs::NamedTuple,
    glacier_idx::Integer,
    θ,
    input_name::Symbol,
    plot_full_input_range::Bool
)
    xlabel = replace(string(input_name), "_" => " ")
    ylabel = replace(string(law.name), "_" => " ")
    scalar = length(get_input(inputs[input_name], simulation, 1, 2010.0)) == 1 ? true : false
    
    if scalar 
        xvals = get_xvals(input_name, inputs, simulation, plot_full_input_range)
        input_tuples = [NamedTuple{(input_name,)}((xval,) ) for xval in xvals]
        outputs = [only(eval_law(law, simulation, i, input_tuples[i], θ)) for i in 1:length(xvals)]
        fig = Plots.plot(xvals, outputs, xlabel=xlabel, ylabel=ylabel, title="Law Function Plot", label=ylabel, linewidth=3, color=:blue)

    else
        xname = input_names[glacier_idx]
        xvals = get_input(inputs[xname], simulation, glacier_idx, 2010.0)
        input_tuple = NamedTuple{(xname,)}((xvals,))
        outputs = eval_law(law, simulation, glacier_idx, input_tuple, θ)
        fig = Plots.plot(xvals, outputs, xlabel=xlabel, ylabel=ylabel, title="Law Function Plot", label=ylabel, linewidth=3, color=:blue)
    end

    return fig
end

function plot_law_2d(
    law::AbstractLaw,
    simulation::Simulation,
    inputs::NamedTuple,
    glacier_idx::Integer,
    θ,
    input_names::Vector{Symbol},
    idx_fixed_input::Integer
)
    xname, yname = input_names
    xvals = get_input(inputs[xname], simulation, glacier_idx, 2010.0)
    yvals = get_input(inputs[yname], simulation, glacier_idx, 2010.0)

    if idx_fixed_input != 0
        return plot_law_2d_fixed(law, simulation, inputs, glacier_idx, θ, input_names, xvals, yvals, idx_fixed_input)
    else
        return plot_law_2d_surface(law, simulation, glacier_idx, θ, xname, yname, xvals, yvals)
    end
end

function plot_law_2d_fixed(
    law::AbstractLaw,
    simulation::Simulation,
    inputs::NamedTuple,
    glacier_idx::Integer,
    θ,
    input_names::Vector{Symbol},
    xvals,
    yvals,
    idx_fixed_input::Integer
)
    fixed_input_name = input_names[idx_fixed_input]
    fixed_input_vals = get_input(inputs[fixed_input_name], simulation, glacier_idx, 2010.0)
    fixed_mean = mean(fixed_input_vals)
    if idx_fixed_input == 1
        input_tuple = NamedTuple{(input_names[1], input_names[2])}((fill(fixed_mean, size(yvals)), yvals))
        non_fixed_vals = yvals
        non_fixed_name = input_names[2]
    elseif idx_fixed_input == 2
        input_tuple = NamedTuple{(input_names[1], input_names[2])}((xvals, fill(fixed_mean, size(xvals))))
        non_fixed_vals = xvals
        non_fixed_name = input_names[1]
    else
        error("`idx_fixed_input` must be 1 or 2 for two inputs.")
    end
    zs = eval_law(law, simulation, glacier_idx, input_tuple, θ)
    println("plotting 2D scatter with fixed input $(fixed_input_name) at mean value $(fixed_mean)")

    return Plots.scatter(non_fixed_vals, zs; xlabel=string(non_fixed_name), ylabel=string(law.name), title="Law Function 2D Plot (Fixed $(fixed_input_name))")
end

function plot_law_2d_surface(
    law::AbstractLaw,
    simulation::Simulation,
    glacier_idx::Integer,
    θ,
    xname::Symbol,
    yname::Symbol,
    xvals,
    yvals
)
    input_tuple = NamedTuple{(xname, yname)}((xvals, yvals))
    zs = eval_law(law, simulation, glacier_idx, input_tuple, θ)

    if size(zs) != size(xvals) && size(zs) != size(yvals)
        xvals = inn1(xvals)
        yvals = inn1(yvals)
    end

    # Ensure xvals, yvals, zs are vectors of equal length for scatter3d
    xv = vec(xvals)
    yv = vec(yvals)
    zv = vec(zs)

    cmin = minimum(zv)
    cmax = maximum(zv)
    return PlotlyJS.plot(
        PlotlyJS.scatter3d(
            x = xv,
            y = yv,
            z = zv,
            mode = "markers",
            marker = attr(
                size = 8,  
                color = zv,
                colorscale = "Viridis",
                colorbar = attr(title = string(law.name)),
                cmin = cmin,
                cmax = cmax
            )
        ),
        PlotlyJS.Layout(
            title = "Law Function Surface",
            scene = attr(
                xaxis = attr(title = string(xname)),
                yaxis = attr(title = string(yname)),
                zaxis = attr(title = string(law.name))
            )
        )
    )
end

function get_xvals(input_name::Symbol, inputs::NamedTuple, simulation::Simulation, plot_full_input_range::Bool)
    if plot_full_input_range
        # We plot customised input ranges depending on the input
        if input_name == :T
            T_max = simulation.parameters.physical.maxTlaw
            T_min = simulation.parameters.physical.minTlaw
            xvals = collect(T_min:0.5:T_max)
        end
    else
        xvals = [get_input(inputs[input_name], simulation, i, 2010.0) for i in 1:length(simulation.glaciers)]
    end
    return xvals
end