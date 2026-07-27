# # Surface mass balance calibration tutorial

# This tutorial shows how to calibrate a temperature-index (TI) surface mass
# balance (SMB) model against geodetic mass balance observations from
# [Hugonnet et al. (2021)](https://doi.org/10.1038/s41586-021-03436-z).
#
# The idea is simple: Hugonnet et al. provide a glacier-wide mean mass balance
# (in m w.e. yr⁻¹) for the 2000–2020 period. We adjust the parameters of a
# `TImodel1` (a TI model with a single degree-day factor, DDF) so that the
# modelled mean annual mass balance matches that observation for each glacier.

# ## Running the whole code

using ODINN

## Glaciers to calibrate. Their Hugonnet geodetic mass balance is loaded
## automatically when the glaciers are initialized.
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"] # Argentière and Aletschgletscher

rgi_paths = get_rgi_paths()

## The Hugonnet observation period (2000–2020) is used as the calibration tspan.
params = Parameters(
    simulation = SimulationParameters(
    tspan = (2000.0, 2020.0),
    multiprocessing = false,
    workers = 1,
    use_MB = true,
    use_velocities = false,
    rgi_paths = rgi_paths
)
)

## Initializing the glaciers also loads the Hugonnet geodetic mass balance into
## each `glacier.geodetic_MB` (mean MB in m w.e. yr⁻¹) and `glacier.dhdtData`.
glaciers = initialize_glaciers(rgi_ids, params)

## A TI mass balance model with a single DDF. `iceflow = nothing` because the
## TI calibration only uses the static glacier geometry and climate.
model = Model(iceflow = nothing, mass_balance = TImodel1(params))

## Calibrate the mass balance model against the geodetic observations. This
## fills `model.mass_balance` with one calibrated `TImodel1` per glacier.
calibrate_MB_model!(model, glaciers, params)
calibrated_models = model.mass_balance

## Inspect the calibrated parameters and how well the modelled MB matches the
## Hugonnet observation for each glacier.
for (glacier, cal_model) in zip(glaciers, calibrated_models)
    cal_mb = compute_mean_annual_MB(cal_model, glacier, 2000.0, 2020.0)
    println("Glacier ", glacier.rgi_id)
    println("  Hugonnet geodetic MB : ", round(glacier.geodetic_MB; digits = 4), " m w.e. yr⁻¹")
    println("  Calibrated DDF       : ", round(cal_model.DDF * 1000; digits = 4), " mm w.e. °C⁻¹ d⁻¹")
    println("  prcp_fac             : ", round(cal_model.prcp_fac; digits = 4))
    println("  temp_bias            : ", round(cal_model.temp_bias; digits = 4), " °C")
    println("  Calibrated model MB  : ", round(cal_mb; digits = 4), " m w.e. yr⁻¹")
end

# ## Step-by-step explanation

# ### Loading the geodetic observations
#
# When `initialize_glaciers` is called, Sleipnir looks up each glacier in the
# Hugonnet et al. (2021) geodetic dataset and stores the result on the glacier:
#
# - `glacier.geodetic_MB`: the glacier-wide mean mass balance over 2000–2020,
#   in m w.e. yr⁻¹ — this is the calibration target.
# - `glacier.dhdtData`: the observation period and surface elevation change rate.
#
# No extra step is required: the data ships with Sleipnir for the reference
# glaciers used here, and is loaded from the full Hugonnet table when available.

# ### Choosing the mass balance model
#
# `TImodel1` is a temperature-index model with a single degree-day factor
# (`DDF`). Its mean annual mass balance over a period is the difference between
# accumulation (snow) and ablation (melt = `DDF` × positive degree days). The
# calibration tunes, in a cascade:
#
# 1. the degree-day factor `DDF`, with a glacier-specific precipitation factor
#    `prcp_fac` derived from winter precipitation (OGGM's approach);
# 2. if `DDF` alone cannot reach the target, the `prcp_fac`;
# 3. and finally a `temp_bias` if neither of the above brackets the target.

# ### Running the calibration
#
# `calibrate_MB_model!(model, glaciers, params)` expands the single `TImodel1`
# template into a vector of per-glacier calibrated models, fitting each glacier
# against its `geodetic_MB`. For a whole simulation you can instead set
# `calibrate_MB = true` in `SimulationParameters`, which calibrates the mass
# balance model automatically before the run.
#
# At scale (hundreds or thousands of glaciers), run the calibration through a
# Huginn `Prediction` or an ODINN `Inversion`: their `Parameters` constructor
# loads Muninn on the workers, so the calibration distributes across processes
# when `multiprocessing = true`.

# ### Using the calibrated models
#
# `compute_mean_annual_MB(cal_model, glacier, t0, t1)` returns the modelled mean
# annual mass balance, which should now be close to `glacier.geodetic_MB`. The
# calibrated models are ready to be used in forward simulations, exactly like
# the mass balance model in the [Forward simulation tutorial](forward_simulation.md).
