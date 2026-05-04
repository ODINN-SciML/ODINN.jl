using Pkg

Pkg.activate(normpath(joinpath(@__DIR__, "..")))

# Point unreleased ecosystem packages to local dev checkouts.
Pkg.develop(path = "/Users/Bolib001/.julia/dev/Muninn")
Pkg.develop(path = "/Users/Bolib001/.julia/dev/Sleipnir")
Pkg.develop(path = "/Users/Bolib001/.julia/dev/Huginn")
Pkg.develop(path = "/Users/Bolib001/Desktop/Jordi/Julia/MassBalanceMachine.jl")

using Revise
using ODINN
using MassBalanceMachine
using Plots
using Statistics
using Printf

const RGI_ID = "RGI60-08.00203"
const TSPAN = (2010.0, 2012.0)
const MODEL_NAME = "mlp_noSvf_wgms11_small_0.1"

function _build_params()
    return Parameters(
        simulation = SimulationParameters(
            working_dir = Sleipnir.prepro_dir,
            tspan = TSPAN,
            multiprocessing = false,
            workers = 1,
            climate_data_source = :ERA5,
            rgi_paths = get_rgi_paths(),
            use_MB = true,
            use_iceflow = true,
            test_mode = false
        ),
        solver = SolverParameters(
            step = 1 / 12,
            progress = true,
            save_everystep = false
        )
    )
end

function _feature_field(clim_step, feature::String)
    if feature == "ELEVATION_DIFFERENCE"
        return clim_step.elevation_diff
    elseif feature == "aspect"
        return clim_step.aspect
    elseif feature == "fal"
        return clim_step.albedo
    elseif feature == "slhf"
        return clim_step.slhf
    elseif feature == "slope"
        return clim_step.slope
    elseif feature == "sshf"
        return clim_step.sshf
    elseif feature == "ssrd"
        return clim_step.ssrd
    elseif feature == "str"
        return clim_step.str
    elseif feature == "t2m"
        return clim_step.temp
    elseif feature == "tp"
        return clim_step.snow .+ clim_step.rain
    end
    error("Unsupported feature: $feature")
end

function _stabilize_field_for_plot(field::Matrix{Sleipnir.Float})
    stable = copy(field)
    finite_idx = findall(isfinite, stable)
    isempty(finite_idx) && return nothing, "all-non-finite"

    vals = stable[finite_idx]
    vmin, vmax = extrema(vals)
    if vmin == vmax
        # Avoid Makie error when auto-colorrange collapses to a single value.
        i = first(finite_idx)
        eps_val = max(abs(vmin) * Sleipnir.Float(1e-6), Sleipnir.Float(1e-6))
        stable[i] += eps_val
        return stable, "constant"
    end
    return stable, "ok"
end

function _save_matrix_plot(
        field::Matrix{Sleipnir.Float},
        path::String;
        title::String,
        colorbar_label::String)
    finite_vals = filter(isfinite, vec(field))
    if isempty(finite_vals)
        p = plot(title = title, framestyle = :none)
        annotate!(p, 0.5, 0.5, text("All values are non-finite", 10))
        savefig(p, path)
        return
    end

    vmin, vmax = extrema(finite_vals)
    p = if vmin == vmax
        delta = max(abs(vmin) * Sleipnir.Float(1e-6), Sleipnir.Float(1e-6))
        heatmap(
            field;
            title = title,
            color = :balance,
            clim = (vmin - delta, vmax + delta),
            colorbar_title = colorbar_label,
            aspect_ratio = :equal
        )
    else
        heatmap(
            field;
            title = title,
            color = :balance,
            colorbar_title = colorbar_label,
            aspect_ratio = :equal
        )
    end
    savefig(p, path)
end

function run_diagnostics()
    params = _build_params()
    mb_model = load_model(MODEL_NAME)
    model = Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = mb_model
    )

    glaciers = initialize_glaciers([RGI_ID], params)
    prediction = Prediction(model, glaciers, params)
    run!(prediction)

    glacier = glaciers[1]
    results = prediction.results[1]

    out_dir = joinpath(ODINN.root_plots, "mb_blob_diagnostics_hardangerjokulen")
    mkpath(out_dir)

    # Raw MB diagnostics at first MB callback step
    step_mb = params.simulation.step_MB
    t_diag = first(params.simulation.tspan) + step_mb

    get_cumulative_climate!(glacier.climate, t_diag, step_mb)
    downscale_2D_climate!(
        glacier;
        include_topography = true,
        topography_window_m = Muninn.topography_window_m(mb_model)
    )

    clim_step = glacier.climate.climate_2D_step
    mb_raw = Muninn.compute_MB(mb_model, clim_step, step_mb)

    diag_fields = Dict{String, Matrix{Sleipnir.Float}}(
        "mb_raw_first_step" => mb_raw,
        "feature_elevation_difference" => Sleipnir.Float.(clim_step.elevation_diff),
        "feature_slope" => Sleipnir.Float.(clim_step.slope),
        "feature_aspect" => Sleipnir.Float.(clim_step.aspect),
        "feature_t2m" => Sleipnir.Float.(clim_step.temp),
        "feature_tp" => Sleipnir.Float.(clim_step.snow .+ clim_step.rain),
        "feature_fal" => Sleipnir.Float.(clim_step.albedo),
        "feature_slhf" => Sleipnir.Float.(clim_step.slhf),
        "feature_sshf" => Sleipnir.Float.(clim_step.sshf),
        "feature_ssrd" => Sleipnir.Float.(clim_step.ssrd),
        "feature_str" => Sleipnir.Float.(clim_step.str)
    )

    # Save baseline context plots
    save_figure(
        plot_glacier_dem(glacier; plotContour = true),
        joinpath(out_dir, "DEM.png")
    )
    save_figure(
        plot_cumulative_mb(results; colormap = :balance, plotContour = true),
        joinpath(out_dir, "cumulative_mb.png")
    )

    for (name, field) in diag_fields
        stable_field, status = _stabilize_field_for_plot(field)
        if isnothing(stable_field)
            println("Plotting $(name): all values are non-finite.")
            stable_field = field
        elseif status == "constant"
            println("Plotting $(name): constant field detected.")
        end

        _save_matrix_plot(
            stable_field,
            joinpath(out_dir, "$(name).png");
            title = "Diagnostic $(name)",
            colorbar_label = name
        )
    end

    # Correlations over glacier ice only
    ice_mask = glacier.H₀ .> 0.0
    mb_vec = vec(mb_raw[ice_mask])

    println("\n=== MB Blob Diagnostics (RGI: $(RGI_ID), model: $(MODEL_NAME)) ===")
    println("Output directory: $(out_dir)")
    println("Input features used by CustomMLP: $(mb_model.input_features)")
    println("\nCorrelation(feature, raw_MB) over ice cells:")

    for feature in mb_model.input_features
        field = _feature_field(clim_step, feature)
        field_vec = vec(field[ice_mask])
        if std(field_vec) > 0.0
            @printf("  %-24s : % .5f\n", feature, cor(mb_vec, field_vec))
        else
            @printf("  %-24s : constant field\n", feature)
        end
    end

    # Quick gradient metric to flag abrupt patches
    dx = abs.(diff(mb_raw; dims = 1))
    dy = abs.(diff(mb_raw; dims = 2))
    p95_dx = quantile(vec(dx), 0.95)
    p95_dy = quantile(vec(dy), 0.95)
    println("\nMB abruptness metrics:")
    println("  95th percentile |dMB/dx|: $(p95_dx)")
    println("  95th percentile |dMB/dy|: $(p95_dy)")

    # ── Ablation experiments ──────────────────────────────────────────────────
    # For each spatially-varying feature, replace the feature's 2-D field with
    # its spatial mean (uniform value, so the MLP sees no spatial structure from
    # that feature) and re-run compute_MB.  The ablated MB map is saved and its
    # difference from the baseline is printed together with a simple abruptness
    # metric.  A large drop in abruptness when a feature is ablated identifies
    # that feature as the source of the blob.

    spatial_features = ["slope", "aspect", "ELEVATION_DIFFERENCE", "t2m"]

    println("\n=== Ablation experiments (each feature spatially flattened) ===")
    println(@sprintf("%-24s  %8s  %8s  %8s  %8s", "feature_ablated",
            "p95|dMB/dx|", "p95|dMB/dy|", "ΔRMSE_vs_base", "Δabrupt%"))

    baseline_p95_dx = p95_dx
    baseline_p95_dy = p95_dy

    function _ablated_clim_step(clim_step, feature::String)
        # Shallow-copy so existing matrices are not mutated for other iterations.
        cs = deepcopy(clim_step)
        if feature == "slope"
            cs.slope .= mean(filter(isfinite, vec(cs.slope)))
        elseif feature == "aspect"
            cs.aspect .= mean(filter(isfinite, vec(cs.aspect)))
        elseif feature == "ELEVATION_DIFFERENCE"
            cs.elevation_diff .= mean(filter(isfinite, vec(cs.elevation_diff)))
        elseif feature == "t2m"
            cs.temp .= mean(filter(isfinite, vec(cs.temp)))
        end
        return cs
    end

    for feat in spatial_features
        cs_abl = _ablated_clim_step(clim_step, feat)
        mb_abl = Muninn.compute_MB(mb_model, cs_abl, step_mb)

        # Abruptness metrics for ablated MB
        dx_abl = abs.(diff(mb_abl; dims = 1))
        dy_abl = abs.(diff(mb_abl; dims = 2))
        p95_dx_abl = quantile(vec(dx_abl), 0.95)
        p95_dy_abl = quantile(vec(dy_abl), 0.95)

        # RMSE difference between ablated and baseline
        diff_field = mb_abl .- mb_raw
        rmse_diff = sqrt(mean(diff_field .^ 2))

        # Percentage change in total abruptness
        pct_change_x = 100.0 * (p95_dx_abl - baseline_p95_dx) / max(baseline_p95_dx, 1e-12)
        pct_change_y = 100.0 * (p95_dy_abl - baseline_p95_dy) / max(baseline_p95_dy, 1e-12)
        mean_pct_change = (pct_change_x + pct_change_y) / 2.0

        println(@sprintf("  %-22s  %8.4g  %8.4g  %8.4g  %+8.1f%%",
                feat, p95_dx_abl, p95_dy_abl, rmse_diff, mean_pct_change))

        # Save ablated MB map
        _save_matrix_plot(
            mb_abl,
            joinpath(out_dir, "ablated_mb_no_$(feat).png");
            title = "MB without spatial structure in $(feat)",
            colorbar_label = "MB (m w.e.)"
        )

        # Save difference map (ablated - baseline)
        _save_matrix_plot(
            diff_field,
            joinpath(out_dir, "ablation_diff_$(feat).png");
            title = "ΔMB when $(feat) flattened (ablated - baseline)",
            colorbar_label = "ΔMB (m w.e.)"
        )
    end

    println("\nAblation plots saved to: $(out_dir)")
    println("\nDone. Review PNGs in: $(out_dir)")
end

run_diagnostics()
