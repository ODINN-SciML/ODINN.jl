module FiniteDifferencesExt

# When FiniteDifferences is loaded alongside ODINN, define utils to check the adjoint

using ODINN
using FiniteDifferences
import ODINN: grad_finite_diff

"""
    grad_finite_diff(
        simulation::Inversion;
        θ = nothing,
        gradient = nothing,
        finite_difference_order = 3,
        max_params = 60,
        mask_parameter_vector = false,
    )

Compare two gradients at a given parameter vector and compute statistics to check how close they are to each other:

    - The first gradient is either `gradient`, or, when `gradient === nothing`, a gradient
        computed with the adjoint method configured in `simulation.parameters.UDE.grad`.
    - The second gradient is computed with central finite differences of the ice-flow loss.

When the parameter vector contains more than `max_params` entries, finite differences
are evaluated on a randomly selected subset. Initial-condition parameters are always
masked in this case; when `mask_parameter_vector` is `true`, other parameter groups
are masked as well.

The simulation's trainable parameter vector is restored to its original value before
returning, including when gradient evaluation throws an exception.

# Arguments

  - `simulation::Inversion`: Inversion whose loss and gradients are compared.
  - `θ`: Parameter vector at which to evaluate the gradients. Defaults to the simulation's
    current trainable parameter vector.
  - `gradient`: Precomputed gradient to compare with finite differences. If omitted, it is
    computed from `simulation`.
  - `finite_difference_order`: Order of the central finite-difference formula.
  - `max_params`: Maximum number of parameter entries used for finite-difference evaluation.
  - `mask_parameter_vector`: Whether to sample non-initial-condition parameter groups when
    `max_params` is exceeded.

# Returns

A tuple `(ratio, angle, relative_error, (gradient, finite_difference_gradient))`, where
the first three values are absolute gradient-comparison statistics returned by
`stats_err_arrays`, and the final value contains the two gradients that were compared.
"""
function grad_finite_diff(
        simulation::Inversion;
        θ = nothing,
        gradient = nothing,
        finite_difference_order = 3,
        max_params = 60,
        mask_parameter_vector = false
)
    θ_previous = simulation.model.trainable_components.θ
    if isnothing(θ)
        θ = deepcopy(θ_previous)
    end

    function f(_θ, _simulation)
        _simulation.model.trainable_components.θ = _θ
        return ODINN.loss_iceflow_transient(_θ, _simulation, map)
    end

    try
        dθ = if isnothing(gradient)
            dθ = zero(θ)
            if simulation.parameters.UDE.grad isa SciMLSensitivityAdjoint
                dθ .= ODINN.grad_loss_iceflow!(θ, simulation, map)
            else
                SIA2D_grad!(dθ, θ, simulation)
            end
            dθ
        else
            deepcopy(gradient)
        end

        glaciers = simulation.glaciers
        params = simulation.parameters
        n_params = length(θ)
        if n_params > max_params
            # Evaluate gradient on subset of parameters to save some computation
            @info "Testing gradient with a subset of parameters of size $(max_params) since the original parameter vector θ is of dimension $(n_params)."

            # Component array with binary entry
            θ_mask = θ .== nothing

            for key in keys(θ)
                if key == :IC
                    # Initial condition
                    for i in 1:length(glaciers)
                        glacier = glaciers[i]
                        M = ODINN.evaluate_H₀(θ, glacier, params.UDE.initial_condition_filter, i)
                        non_zero = M .> 1.0
                        idxs = rand(findall(non_zero), max_params)
                        mask = falses(size(M)...)
                        mask[idxs] .= 1
                        key_glacier = Symbol("$(i)")
                        θ_mask.IC[key_glacier] .= mask
                    end
                elseif (key == :A) && (Symbol("1") in keys(θ.A)) &&
                       length(θ.A) != length(glaciers)
                    # Gridded classical inversion
                    for i in 1:length(glaciers)
                        glacier = glaciers[i]
                        M = glacier.H₀
                        non_zero = M .> 1.0
                        idxs = rand(findall(non_zero), max_params)
                        mask = falses(size(M) .- 1)
                        mask[idxs] .= 1
                        key_glacier = Symbol("$(i)")
                        θ_mask.A[key_glacier] .= mask
                    end
                else
                    # Mask parameter vector
                    if mask_parameter_vector && (length(θ[key]) > max_params)
                        indx = ODINN.sample(1:length(θ[key]), max_params; replace = false)
                    else
                        indx = 1:length(θ[key]) |> collect
                    end
                    view(θ_mask, key)[indx] .= true
                end
            end

            function f_subset(_θ, simulation, θ_mask)
                α = deepcopy(θ)
                α[θ_mask] .= _θ
                return f(α, simulation)
            end
            dθ_FD, = FiniteDifferences.grad(
                central_fdm(finite_difference_order, 1), α -> f_subset(α, simulation, θ_mask), θ[θ_mask])
            dθ = dθ[θ_mask]
        else
            # Compute gradient with all parameters
            dθ_FD, = FiniteDifferences.grad(
                central_fdm(finite_difference_order, 1),
                _θ -> f(_θ, simulation), θ)
        end
        ratio, angle, relerr = stats_err_arrays(dθ, dθ_FD)

    finally
        simulation.model.trainable_components.θ = θ_previous
    end
    return abs(ratio), abs(angle), abs(relerr), (dθ, dθ_FD)
end

end
