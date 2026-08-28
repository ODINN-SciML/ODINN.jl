
"""
    grad_finite_diff(
        simulation::Inversion,
        θ;
        gradient = nothing,
        thres = [0., 0., 0.],
        finite_difference_method = :FiniteDifferences,
        finite_difference_order = 3,
        max_params = 60,
        mask_parameter_vector = false,
    )

Compare the gradient between the one given by the adjoint and a finite differences approximation at a given parameter vector.
It is computed using the adjoint method configured in `simulation.parameters.UDE.grad`.
"""
function grad_finite_diff(
        simulation::Inversion;
        θ = nothing,
        gradient = nothing,
        finite_difference_method = :FiniteDifferences,
        finite_difference_order = 3,
        max_params = 60,
        mask_parameter_vector = false,
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
        if finite_difference_method == :FiniteDifferences
            if n_params > max_params
                # Evaluate gradient on subset of parameters to save some computation
                @info "Testing gradient with a subset of parameters of size $(max_params) since the original parameter vector θ is of dimension $(n_params)."

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

                function f_subset(_θ)
                    α = deepcopy(θ)
                    α[θ_mask] .= _θ
                    return f(α, simulation)
                end
                dθ_FD, = FiniteDifferences.grad(
                    central_fdm(finite_difference_order, 1), f_subset, θ[θ_mask])
                dθ = dθ[θ_mask]
            else
                dθ_FD, = FiniteDifferences.grad(
                    central_fdm(finite_difference_order, 1),
                    _θ -> f(_θ, simulation), θ)
            end
            ratio, angle, relerr = stats_err_arrays(dθ, dθ_FD)
            return ratio, angle, relerr, (dθ, dθ_FD)
        elseif finite_difference_method == :Manual
            ratio = Float64[]
            angle = Float64[]
            relerr = Float64[]
            for exponent in 3:7
                dθ_num = compute_numerical_gradient(
                    θ, simulation, f, 10.0^(-exponent); varStr = "of θ")
                ratio_k, angle_k, relerr_k = stats_err_arrays(dθ, dθ_num)
                push!(ratio, ratio_k)
                push!(angle, angle_k)
                push!(relerr, relerr_k)
            end
            ratio = minimum(abs.(ratio))
            angle = minimum(abs.(angle))
            relerr = minimum(abs.(relerr))
            return ratio, angle, relerr, (dθ,)
        else
            throw(ArgumentError("Finite difference method $(finite_difference_method) not implemented."))
        end

    finally
        simulation.model.trainable_components.θ = θ_previous
    end
end
