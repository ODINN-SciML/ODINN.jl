# export compute_numerical_gradient, stats_err_arrays, printVecScientific

"""
    compute_numerical_gradient(
        x,
        args,
        fct::Function,
        ϵ::F;
        varStr::String = ""
    ) where {T, F <: AbstractFloat}

Compute the gradient of a function by using numerical differences. The function is
evaluated N+1 times with N the length of x.
The function has two arguments: the variable with respect to which the gradient is
computed and extra arguments which are not differentiated.

Arguments:
- `x`: Variable x where to evaluate the gradient.
- `args`: Extra arguments.
- `fct::Function`: Function to differentiate.
- `epsilon::F`: Size of perturbation to use in the numerical differences.
- `varStr::String`: Variable to print in the progress bar. Example: "of A" will
    print "Computing gradient of A using finite differences with..."

Returns:
- `grad`: Numerical gradient.
"""
function compute_numerical_gradient(
    x,
    args,
    fct::Function,
    ϵ::F;
    varStr::String = ""
) where {F <: AbstractFloat}
    grad = zero(x)
    grad_vec = vec(grad) # Points to the same position in memory
    x_ϵ = deepcopy(x)
    x_ϵ_vec = vec(x_ϵ)
    f0 = fct(x, args)
    show_progress = !parse(Bool, get(ENV, "CI", "false"))
    pbar = Progress(length(x); desc="Computing gradient $(varStr) using finite differences with ϵ=$(@sprintf("%.1e", ϵ))...", enabled=show_progress)
    for i in range(1,length(x))
        x_ϵ .= x
        x_ϵ_vec[i] += ϵ
        grad_vec[i] = (fct(x_ϵ, args) - f0) / ϵ
        next!(pbar)
    end
    return grad
end

"""
    stats_err_arrays(a::T, b::T) where T

Compute the ratio, the angle and the relative error between two arrays.
The norm and scalar product are defined in the vectorial sense, meaning that this is
mathematically equivalent to flatten the arrays.
The arrays must be of the same type and have the same shape.

Arguments:
- `a::T`: First array.
- `b::T`: Second array to compare to the first one.

Returns:
- `ratio::Float64`: Ratio between the norm of the two arrays minus 1. Value close
    to zero means the arrays have approximately the same norm.
- `angle::Float64`: Scalar product between the two arrays normalized by the norm
    minus 1. Value close to zero means the arrays point towards the same direction.
- `relerr::Float64`: Relative error between the two arrays. Value close to zero
    means the arrays have approximately the same values. The norm of `a` is taken
    to normalize and compute the relative error.
"""
function stats_err_arrays(a::T, b::T) where T
    ratio = sqrt(sum(a.^2)) / sqrt(sum(b.^2)) - 1
    angle = sum(a.*b) / (sqrt(sum(a.^2)) * sqrt(sum(b.^2))) - 1
    relerr = sqrt(sum((a - b).^2)) / sqrt(sum((a).^2))
    return ratio, angle, relerr
end

printVecScientific(v) = join([@sprintf("%9.2e", e) for e in v], " ")
function printVecScientific(baseVarName, v, thres=nothing)
    print(baseVarName)
    for e in v
        if isnothing(thres)
            print(@sprintf("%9.2e", e))
        else
            if abs(e)<=thres
                printstyled(@sprintf("%9.2e", e); color=:green)
            else
                printstyled(@sprintf("%9.2e", e); color=:red)
            end
        end
        print(" ")
    end
    if !isnothing(thres)
        print("< $(thres)")
    end
    println("")
end
