function _write_synthetic_era5_monthly(path::String)
    mkpath(dirname(path))
    start_date = Date(2008, 1, 1)
    end_date = Date(2018, 12, 1)
    dates = collect(start_date:Month(1):end_date)
    ntime = length(dates)

    NCDatasets.NCDataset(path, "c") do ds
        NCDatasets.defDim(ds, "time", ntime)
        vtime = NCDatasets.defVar(ds, "time", Float64, ("time",))
        vtime.attrib["units"] = "days since 2008-01-01 00:00:00"
        vtime.attrib["calendar"] = "proleptic_gregorian"
        vtime[:] = Float64.(Dates.value.(dates .- start_date))

        vtemp = NCDatasets.defVar(ds, "temp", Float32, ("time",))
        vprcp = NCDatasets.defVar(ds, "prcp", Float32, ("time",))
        vgrad = NCDatasets.defVar(ds, "gradient", Float32, ("time",))
        vfal = NCDatasets.defVar(ds, "fal", Float32, ("time",))
        vslhf = NCDatasets.defVar(ds, "slhf", Float32, ("time",))
        vsshf = NCDatasets.defVar(ds, "sshf", Float32, ("time",))
        vssrd = NCDatasets.defVar(ds, "ssrd", Float32, ("time",))
        vstr = NCDatasets.defVar(ds, "str", Float32, ("time",))

        idx = Float32.(1:ntime)
        annual = @. sin(2.0f0 * Float32(pi) * (idx - 4.0f0) / 12.0f0)
        prcpcyc = @. cos(2.0f0 * Float32(pi) * (idx - 1.0f0) / 12.0f0)

        vtemp[:] = @. -7.0f0 + 9.0f0 * annual
        vprcp[:] = @. max(0.0f0, 0.045f0 + 0.015f0 * prcpcyc)
        vgrad[:] = @. -0.0060f0 + 0.0008f0 * annual
        vfal[:] = clamp.(0.60f0 .- 0.18f0 .* annual, 0.25f0, 0.85f0)
        vslhf[:] = @. Float32(-8.0e6) + Float32(3.0e6) * annual
        vsshf[:] = @. Float32(5.0e6) + Float32(2.0e6) * prcpcyc
        vssrd[:] = max.(0.0f0, @. Float32(1.8e7) + Float32(1.2e7) * annual)
        vstr[:] = @. Float32(-6.0e6) + Float32(1.5e6) * prcpcyc

        ds.attrib["climate_source"] = "ERA5 CDS"
        ds.attrib["climate_frequency"] = "monthly"
        ds.attrib["ref_hgt"] = Float32(2500.0)
    end

    return path
end

function _ensure_synthetic_era5_fixture(rgi_paths::Dict, rgi_ids::Vector{String})
    fixture_root = mktempdir()

    for rgi_id in rgi_ids
        src_rgi_path = joinpath(Sleipnir.prepro_dir, rgi_paths[rgi_id])
        test_rgi_path = joinpath(fixture_root, rgi_id)
        mkpath(test_rgi_path)

        for fname in ("glacier_grid.json", "gridded_data.nc")
            src = joinpath(src_rgi_path, fname)
            dst = joinpath(test_rgi_path, fname)
            isfile(src) || error("Required glacier file $fname not found at $src")
            cp(src, dst; force = true)
        end

        monthly_path = joinpath(test_rgi_path, "climate_historical_monthly_ERA5.nc")
        _write_synthetic_era5_monthly(monthly_path)

        for f in readdir(test_rgi_path; join = true)
            if startswith(basename(f), "raw_climate_") && endswith(f, ".nc")
                rm(f)
            end
        end

        rgi_paths[rgi_id] = test_rgi_path
    end
end

function _write_custom_mlp_jsons(tmpdir::String)
    params_json = joinpath(tmpdir, "params.json")
    model_json = joinpath(tmpdir, "model.json")

    params_data = Dict(
        "model" => Dict("layers" => [8, 8]),
        "training" => Dict(
            "batch_size" => 16,
            "optim" => "ADAM",
            "lr" => 0.001,
            "Nepochs" => 50,
            "beta1" => 0.9,
            "beta2" => 0.999,
            "weight_decay" => 0.0,
            "momentum" => 0.0,
            "device" => "cpu",
            "shuffle" => true
        )
    )

    model_data = Dict(
        "inputs" => ["t2m", "tp"],
        "norm" => [[-20.0, 15.0], [0.0, 0.1]],
        "model" => Dict(
            "0.weight" => [
                [0.42, 0.18],
                [-0.31, 0.09],
                [0.27, 0.35],
                [-0.16, 0.28],
                [0.08, -0.22],
                [-0.23, 0.16],
                [0.38, 0.11],
                [-0.19, 0.31]
            ],
            "0.bias" => [0.05, 0.08, 0.03, 0.01, 0.0, 0.02, 0.04, 0.01],
            "2.weight" => [
                [0.31, -0.18, 0.12, 0.24, -0.08, 0.15, 0.20, -0.10],
                [-0.15, 0.25, 0.20, -0.10, 0.20, -0.18, 0.08, 0.15],
                [0.10, 0.20, -0.30, 0.08, 0.12, -0.10, 0.22, 0.15],
                [-0.10, -0.08, 0.10, 0.30, -0.15, 0.20, -0.12, 0.10],
                [0.20, -0.10, 0.15, -0.08, 0.30, 0.06, -0.20, 0.12],
                [-0.08, 0.15, -0.12, 0.18, -0.10, -0.28, 0.06, -0.20],
                [0.12, -0.22, 0.10, -0.10, 0.20, -0.08, 0.25, -0.08],
                [-0.22, 0.08, -0.08, 0.11, -0.12, 0.20, -0.10, 0.28]
            ],
            "2.bias" => [0.02, 0.0, -0.01, 0.01, 0.0, -0.02, 0.01, 0.0],
            "4.weight" => [[-0.28, 0.22, -0.11, -0.19, 0.33, -0.07, 0.14, -0.24]],
            "4.bias" => [-0.08]
        )
    )

    open(params_json, "w") do f
        JSON.print(f, params_data)
    end
    open(model_json, "w") do f
        JSON.print(f, model_data)
    end

    return params_json, model_json
end

function test_forward_with_custommlp()
    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)
    _ensure_synthetic_era5_fixture(rgi_paths, rgi_ids)

    workdir = mktempdir()
    params = Parameters(
        simulation = SimulationParameters(
            working_dir = workdir,
            use_MB = true,
            use_iceflow = true,
            use_velocities = false,
            climate_data_source = :ERA5,
            tspan = (2010.0, 2010.25),
            step_MB = 1.0 / 12.0,
            multiprocessing = false,
            workers = 1,
            test_mode = true,
            gridScalingFactor = 4,
            rgi_paths = rgi_paths
        ),
        solver = SolverParameters(
            step = 1.0 / 12.0,
            progress = false,
            save_everystep = false
        )
    )

    params_json, model_json = _write_custom_mlp_jsons(mktempdir())
    mb_model = CustomMLP(params_json, model_json)
    model = Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = mb_model
    )

    glaciers = initialize_glaciers(rgi_ids, params)
    @test glaciers[1].climate.climate_data_source == :ERA5

    prediction = Prediction(model, glaciers, params)
    run!(prediction)

    @test length(prediction.results) == 1
    @test prediction.results[1] !== nothing
end
