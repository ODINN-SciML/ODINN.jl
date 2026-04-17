#!/usr/bin/env julia
#=
  test_all_plots_realistic.jl

  Runs a short forward simulation on a real glacier and tests the full plotting
  stack using realistic glacier and climate data.

  Run from the ODINN.jl project root:
      julia --project=. scripts/test_all_plots_realistic.jl
=#

using Pkg

Pkg.activate(normpath(joinpath(@__DIR__, "..")))

# Keep the local ecosystem packages aligned with this workspace checkout.
Pkg.develop(path = "/Users/Bolib001/.julia/dev/Muninn")
Pkg.develop(path = "/Users/Bolib001/.julia/dev/Sleipnir")
Pkg.develop(path = "/Users/Bolib001/.julia/dev/Huginn")
Pkg.develop(path = "/Users/Bolib001/Desktop/Jordi/Julia/MassBalanceMachine.jl")

using ODINN
using Sleipnir
using CairoMakie
using Dates

const OUTPUT_DIR = joinpath(@__DIR__, "..", "plots", "visual_tests_realistic")
const RGI_ID = "RGI60-11.01450"
const TSPAN = (2010.0, 2015.0)
const PLOT_TEST_DDF = 2.0 / 1000.0
const PLOT_TEST_ACC_FACTOR = 1.4 / 1000.0
const LAW_PLOT_RGI_ID = "RGI60-11.03638"
const LAW_PLOT_STEP = 1 / 12
const LAW_PLOT_TSPAN = (2010.0, 2015.0)

mkpath(OUTPUT_DIR)

function save_test(fig, name::String)
    path = joinpath(OUTPUT_DIR, name * ".pdf")
    CairoMakie.save(path, fig)
    println("  ✓ $name")
    return path
end

# Legacy quiver implementation copied from Sleipnir origin/main (pre-refactor).
function legacy_plot_glacier_quivers_oldlogic(
        results::Sleipnir.Results,
        variables::Vector{Symbol};
        timeIdx::Union{Nothing, Int64} = nothing,
        figsize::Union{Nothing, Tuple{Int64, Int64}} = nothing,
        lengthscale::Float64 = 0.00001,
        tiplength::Float64 = 0.5
)
    figKwargs = isnothing(figsize) ? Dict{Symbol, Any}() :
                Dict{Symbol, Any}(:size => figsize)

    x = results.x
    y = results.y

    velocity_vars = [:V, :V_ref]
    for var in intersect(velocity_vars, variables)
        if hasproperty(results, var)
            current_matrix = getfield(results, var)
            if !isnothing(current_matrix) && !isempty(current_matrix)
                if current_matrix isa Vector
                    @assert length(current_matrix) > 0 "Variable $(var) is an empty vector"
                    @assert isnothing(timeIdx) || size(current_matrix, 1) >= timeIdx "The provided index=$(timeIdx) is greater than the size of the vector for $(var) which is $(size(current_matrix,1))"
                end
            end
        end
    end

    num_vars = length(variables)
    rows, cols = if num_vars == 1
        1, 1
    elseif num_vars == 2
        1, 2
    else
        error("Unsupported number of variables.")
    end

    figKwargs[:layout] = GridLayout(rows, cols)
    fig = Figure(; figKwargs...)
    for (ax_col, var) in enumerate(variables)
        ax = Axis(fig[1, ax_col], aspect = DataAspect())
        data = getfield(results, var)
        title = string(var)

        if data isa Vector
            @assert length(data) > 0 "Variable $(var) is an empty vector"
            @assert isnothing(timeIdx) || size(data, 1) >= timeIdx "The provided index=$(timeIdx) is greater than the size of the vector for $(var) which is $(size(data,1))"
            data = isnothing(timeIdx) ? data[end] : data[timeIdx]
            Vx = getfield(results, var == :V ? :Vx : :Vx_ref)
            Vy = getfield(results, var == :V ? :Vy : :Vy_ref)
            dataVx = isnothing(timeIdx) ? Vx[end] : Vx[timeIdx]
            dataVy = isnothing(timeIdx) ? Vy[end] : Vy[timeIdx]
        end

        X, Y = Sleipnir.meshgrid(x, y)
        positions = Point2f.(reshape(X, :), reshape(Y, :))
        directions = Vec2f.(dataVx, -dataVy)
        arrows2d!(
            ax, positions, directions; tiplength = tiplength, lengthscale = lengthscale)

        ax.title = title
        ax.xlabel = "Longitude"
        ax.ylabel = "Latitude"
        ax.yticklabelrotation = π / 2
        ax.ylabelpadding = 5
        ax.yticklabelalign = (:center, :bottom)
    end

    resize_to_layout!(fig)
    return fig
end

function stable_plot_mb_model(params)
    println("  ℹ Using conservative TImodel1 SMB for stable plotting.")
    return TImodel1(params; DDF = PLOT_TEST_DDF, acc_factor = PLOT_TEST_ACC_FACTOR)
end

function build_prediction()
    params = Parameters(
        simulation = SimulationParameters(
            working_dir = Sleipnir.prepro_dir,
            tspan = TSPAN,
            multiprocessing = false,
            workers = 1,
            climate_data_source = :ERA5,
            rgi_paths = get_rgi_paths(),
            use_MB = true,
            use_velocities = true,
            use_glathida_data = true,
            use_iceflow = true,
            test_mode = false
        ),
        solver = SolverParameters(
            step = 1 / 12,
            progress = true,
            save_everystep = true
        )
    )

    model = Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = stable_plot_mb_model(params)
    )

    glaciers = initialize_glaciers([RGI_ID], params)
    prediction = Prediction(model, glaciers, params)
    run!(prediction)

    @assert !isempty(prediction.results) "Prediction returned no results."
    return params, glaciers[1], prediction.results[1]
end

function use_observed_fields_when_available!(glacier::Sleipnir.Glacier2D, results::Sleipnir.Results)
    nx, ny = results.nx, results.ny

    # Populate all reference fields needed by plotting from real data only
    # (either glacier initialization products or forward-prediction outputs).

    # GlaThiDa thickness (observed) from glacier initialization.
    if (isempty(results.H_glathida) || size(results.H_glathida) != (nx, ny)) &&
       !isempty(glacier.H_glathida) && size(glacier.H_glathida) == (nx, ny)
        results.H_glathida = copy(glacier.H_glathida)
        println("  ✓ Using observed H_glathida from glacier initialization.")
    end

    # Thickness reference for plots: prefer observed GlaThiDa, then initialized H₀.
    if isempty(results.H_ref) || isempty(results.H_ref[1]) ||
       size(results.H_ref[1]) != (nx, ny)
        if !isempty(glacier.H_glathida) && size(glacier.H_glathida) == (nx, ny)
            results.H_ref = [copy(glacier.H_glathida)]
            println("  ✓ Using observed thickness reference from GlaThiDa.")
        elseif !isempty(glacier.H₀) && size(glacier.H₀) == (nx, ny)
            results.H_ref = [copy(glacier.H₀)]
            println("  ✓ Using initialized thickness reference from H₀.")
        end
    end

    # Velocity references from initialized glacier velocity products.
    if isempty(results.V_ref) || isempty(results.V_ref[1]) ||
       size(results.V_ref[1]) != (nx, ny)
        if !isempty(glacier.V) && size(glacier.V) == (nx, ny)
            results.V_ref = [copy(glacier.V)]
            println("  ✓ Using initialized velocity magnitude as V_ref.")
        end
    end
    if isempty(results.Vx_ref) || isempty(results.Vx_ref[1]) ||
       size(results.Vx_ref[1]) != (nx, ny)
        if !isempty(glacier.Vx) && size(glacier.Vx) == (nx, ny)
            results.Vx_ref = [copy(glacier.Vx)]
            println("  ✓ Using initialized velocity x-component as Vx_ref.")
        end
    end
    if isempty(results.Vy_ref) || isempty(results.Vy_ref[1]) ||
       size(results.Vy_ref[1]) != (nx, ny)
        if !isempty(glacier.Vy) && size(glacier.Vy) == (nx, ny)
            results.Vy_ref = [copy(glacier.Vy)]
            println("  ✓ Using initialized velocity y-component as Vy_ref.")
        end
    end

    # Ensure no required plot fields remain missing.
    @assert !isempty(results.H) && !isempty(results.H[end]) "Missing prediction thickness H for plotting"
    @assert !isempty(results.V) && !isempty(results.V[end]) "Missing prediction velocity V for plotting"
    @assert !isempty(results.S) "Missing surface elevation S for plotting"
    @assert !isempty(results.B) "Missing bed elevation B for plotting"
    @assert !isempty(results.H_ref) && !isempty(results.H_ref[1]) "Missing thickness reference H_ref for plotting"
    @assert !isempty(results.V_ref) && !isempty(results.V_ref[1]) "Missing velocity reference V_ref for plotting"
    @assert !isempty(results.Vx_ref) && !isempty(results.Vx_ref[1]) "Missing velocity reference Vx_ref for plotting"
    @assert !isempty(results.Vy_ref) && !isempty(results.Vy_ref[1]) "Missing velocity reference Vy_ref for plotting"
    @assert !isempty(results.H_glathida) "Missing observed H_glathida for plotting"

    return nothing
end

function build_law_plot_prediction()
    law_inputs = (; CPDD = iCPDD(window = Week(1)),
        topo_roughness = iTopoRough(window = 200.0, curvature_type = :variability))

    params = Parameters(
        simulation = SimulationParameters(
            working_dir = Sleipnir.prepro_dir,
            tspan = LAW_PLOT_TSPAN,
            multiprocessing = false,
            workers = 1,
            climate_data_source = :ERA5,
            rgi_paths = get_rgi_paths(),
            use_MB = false,
            use_velocities = false,
            use_glathida_data = false,
            use_iceflow = true,
            test_mode = false,
            gridScalingFactor = 4
        ),
        solver = SolverParameters(
            step = LAW_PLOT_STEP,
            progress = true,
            save_everystep = true
        )
    )

    model = Model(
        iceflow = SIA2Dmodel(params; C = SyntheticC(params; inputs = law_inputs)),
        mass_balance = nothing
    )

    glaciers = initialize_glaciers([LAW_PLOT_RGI_ID], params)
    tstops = collect(LAW_PLOT_TSPAN[1]:LAW_PLOT_STEP:LAW_PLOT_TSPAN[2])
    prediction = generate_ground_truth_prediction(glaciers, params, model, tstops)

    return prediction, law_inputs
end

function run_law_plot_tests()
    println("\n=== Law plotting utilities ===")
    prediction, law_inputs = build_law_plot_prediction()

    # Reproduce documentation examples for 2D SyntheticC law plotting.
    fig = plot_law(prediction.model.iceflow.C, prediction, law_inputs, nothing)
    save_test(fig, "law_plot_surface")

    fig = plot_law(prediction.model.iceflow.C, prediction, law_inputs, nothing;
        idx_fixed_input = 1)
    save_test(fig, "law_plot_fixed_cpdd")

    fig = plot_law(prediction.model.iceflow.C, prediction, law_inputs, nothing;
        idx_fixed_input = 2)
    save_test(fig, "law_plot_fixed_topo_roughness")
end

function run_plot_tests(glacier::Sleipnir.Glacier2D, results::Sleipnir.Results)
    println("\n=== Gridded plotting utilities ===")

    # Keep static map comparisons on a representative non-zero frame.
    plot_time_idx = 1

    fig = Sleipnir.plot_gridded_data(
        results.S, results; title = "Surface elevation", colorbar_label = "m a.s.l.")
    save_test(fig, "gridded_single_matrix")

    fig = Sleipnir.plot_gridded_data(results.H, results; title = "Thickness (last)", colorbar_label = "m")
    save_test(fig, "gridded_timeseries_last")

    fig = Sleipnir.plot_gridded_data(results.S, results; plotContour = true, title = "DEM + contours")
    save_test(fig, "gridded_with_contour")

    fig = Sleipnir.plot_cumulative_mb(results; title = "Cumulative MB test")
    if fig !== nothing
        save_test(fig, "cumulative_mb")
    else
        println("  ⚠ plot_cumulative_mb returned nothing (empty MB)")
    end

    fig = Sleipnir.plot_glacier_dem(results)
    save_test(fig, "glacier_dem_from_results")

    fig = Sleipnir.plot_glacier_dem(glacier)
    save_test(fig, "glacier_dem_from_glacier")

    fig_tmp = Figure()
    Axis(fig_tmp[1, 1]; title = "save_figure test")
    saved_path = Sleipnir.save_figure(fig_tmp, joinpath(OUTPUT_DIR, "save_figure_test.pdf"))
    println("  ✓ save_figure -> $saved_path")

    println("\n=== Glacier plotting utilities ===")

    fig = Sleipnir.plot_glacier(results, "heatmaps", [:H]; timeIdx = plot_time_idx)
    save_test(fig, "glacier_heatmaps_1var")

    fig = Sleipnir.plot_glacier(results, "heatmaps", [:H, :V]; timeIdx = plot_time_idx)
    save_test(fig, "glacier_heatmaps_2var")

    fig = Sleipnir.plot_glacier(results, "heatmaps", [:H, :V, :S]; timeIdx = plot_time_idx)
    save_test(fig, "glacier_heatmaps_3var")

    # Use fields that are consistently populated in forward predictions.
    fig = Sleipnir.plot_glacier(results, "heatmaps", [:H, :V, :S, :B]; timeIdx = plot_time_idx)
    save_test(fig, "glacier_heatmaps_4var")

    try
        fig = Sleipnir.plot_glacier(results, "quivers", [:V]; timeIdx = plot_time_idx)
        save_test(fig, "glacier_quivers_1var")
    catch e
        println("  ⚠ quivers_1var failed: ", sprint(showerror, e))
    end

    # Compare prediction and observation-based reference velocity maps.
    fig = Sleipnir.plot_glacier(results, "quivers", [:V, :V_ref]; timeIdx = plot_time_idx)
    save_test(fig, "glacier_quivers_2var")

    # Previous logic comparison generated using legacy pre-refactor function.
    fig = legacy_plot_glacier_quivers_oldlogic(results, [:V, :V_ref]; timeIdx = plot_time_idx)
    save_test(fig, "glacier_quivers_2var_old_logic")

    fig = Sleipnir.plot_glacier(results, "evolution difference", [:H];
        tspan = results.tspan, metrics = ["difference"])
    save_test(fig, "glacier_evolution_difference")

    fig = Sleipnir.plot_glacier(results, "evolution difference", [:H];
        tspan = results.tspan, metrics = ["difference", "hist"])
    save_test(fig, "glacier_evolution_diff_and_hist")

    fig = Sleipnir.plot_glacier(results, "evolution statistics", [:H];
        tspan = results.tspan, metrics = ["median"])
    save_test(fig, "glacier_evolution_statistics_median")

    fig = Sleipnir.plot_glacier(results, "evolution statistics", [:H];
        tspan = results.tspan, metrics = ["average", "median", "min", "max"])
    save_test(fig, "glacier_evolution_statistics_multi")

    fig = Sleipnir.plot_glacier(results, "integrated volume", [:H]; tspan = results.tspan)
    save_test(fig, "glacier_integrated_volume")

    fig = Sleipnir.plot_bias(results, [:H_glathida, :S])
    save_test(fig, "glacier_bias")

    fig = Sleipnir.plot_glacier(results, "dem", [:S])
    save_test(fig, "glacier_dem_via_router")

    println("\n=== Miscellaneous ===")

    X, Y = Sleipnir.meshgrid(results.x, results.y)
    @assert size(X) == (results.nx, results.ny) "meshgrid X size mismatch"
    @assert size(Y) == (results.nx, results.ny) "meshgrid Y size mismatch"
    println("  ✓ meshgrid ($(size(X)))")

    fig = Sleipnir.plot_cumulative_mb(results; annual_MB = true, title = "Annual MB test")
    if fig !== nothing
        save_test(fig, "cumulative_mb_annual")
    else
        println("  ⚠ plot_cumulative_mb (annual) returned nothing")
    end

    fig = Sleipnir.plot_gridded_data(results.V[end], results;
        title = "Velocity (log)", colorbar_label = "m/yr", logPlot = true)
    save_test(fig, "gridded_logscale")

    fig = Sleipnir.plot_gridded_data(results.H[end], results;
        title = "Thickness (clamped)", colorrange = (0.0, 200.0), colorbar_label = "m")
    save_test(fig, "gridded_custom_colorrange")

    println("\n=== Video plotting utilities ===")

    step_video = length(results.t) > 1 ? (results.t[2] - results.t[1]) : 1.0
    video_path = joinpath(OUTPUT_DIR, "glacier_thickness_video.mp4")
    Sleipnir.plot_glacier_vid(
        "thickness",
        results,
        glacier,
        results.tspan,
        step_video,
        video_path;
        framerate = 12,
        baseTitle = "Ice thickness"
    )
    @assert isfile(video_path) "Video file was not generated"
    println("  ✓ glacier_thickness_video")

    println("\n=== All realistic-data plot tests complete ===")
    println("Output directory: $OUTPUT_DIR")
end

function main()
    println("Building realistic glacier simulation for $(RGI_ID)...")
    _, glacier, results = build_prediction()
    use_observed_fields_when_available!(glacier, results)
    println("  ✓ Simulation completed")
    println("  nx=$(results.nx), ny=$(results.ny), nt=$(length(results.t))")

    run_plot_tests(glacier, results)
    run_law_plot_tests()
end

main()
