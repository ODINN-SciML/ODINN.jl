# Gungnir

[`Gungnir`](https://github.com/ODINN-SciML/Gungnir) is a Python preprocessing pipeline that produces the glacier and climate data files consumed by the Julia ODINN ecosystem. It uses [OGGM](https://github.com/OGGM/oggm) to retrieve glacier geometry (DEMs, ice thickness, outlines) from the Randolph Glacier Inventory (RGI), and downloads climate reanalyses (W5E5 or ERA5) to force the mass balance models. The output is written to `~/.ODINN/ODINN_prepro/` as NetCDF files, which `Sleipnir.initialize_glaciers()` reads via `Rasters.jl` at simulation time.

'Gungnir` is the only package in the ODINN ecosystem writen and executed in Python. It sits at the bottom of the dependency hierarchy: Gungnir → Sleipnir → Muninn/Huginn → ODINN.

## When do you need to run Gungnir yourself?

For many use cases, **you do not need to run Gungnir**. Pre-built preprocessed datasets for a selection of glaciers can be downloaded automatically when you call `initialize_glaciers()` from `Sleipnir`. Gungnir is only required when:

  - You want to simulate a **glacier not covered** by the existing pre-built datasets.
  - You want to switch or customize the **climate data source** (e.g. use ERA5 daily instead of W5E5, or add a new atmospheric variable).
  - You want to use **higher-resolution or updated DEMs** rather than the OGGM defaults.
  - You are setting up a **new computing environment** where the pre-built files are unavailable and cannot be auto-downloaded.

## Installation and setup

Clone the repository and create the conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate oggm_env_gungnir
```

Alternatively, use the `Makefile` to create the environment and register the Jupyter kernel in one step:

```bash
make env
```

Alternatively, if you just want to install the `gungnir` module, you can clone this repository and do:

```bash
pip install gungnir
# or in developer mode:
# pip install -e gungnir
```

ERA5 high-resolution downloads for `MassBalanceMachine.jl` additionally require a CDS API key. See the [CDS API setup guide](https://cds.climate.copernicus.eu/how-to-api) for registration and configuration of `~/.cdsapirc`.

W5E5 data (the default) does not require an API key and is recommended for most use cases.

## Climate data sources

| Source             | Temporal resolution                   | Variables                                        | Notes                                                   |
|:------------------ |:------------------------------------- |:------------------------------------------------ |:------------------------------------------------------- |
| **W5E5** (default) | Daily                                 | temp, prcp                                       | No API key required; simpler setup                      |
| **ERA5**           | Monthly (default) or daily (optional) | temp, prcp, gradient, fal, slhf, sshf, ssrd, str | Requires CDS API key; richer variable set for ML models |

ERA5 is needed when using `MassBalanceMachine.jl` neural network models, which rely on additional atmospheric variables (shortwave radiation `ssrd`, surface fluxes, etc.).

## Running the preprocessing pipeline

Create a text file listing the RGI IDs of the glaciers to preprocess (one per line, `#` comments allowed):

```
# European Alps
RGI60-11.00897;
RGI60-11.01450;
```

Then run:

```bash
conda activate oggm_env_gungnir
python gungnir/gungnir/preprocessing.py glaciers.txt
```

By default, data is written to `~/.ODINN/ODINN_prepro`. You can provide an explicit output directory as a second argument:

```bash
python gungnir/gungnir/preprocessing.py glaciers.txt /path/to/output
```

After running, `Sleipnir.initialize_glaciers(rgi_ids, params)` will detect and load the generated files automatically.
This function assumes that the preprocessed data are in `~/.ODINN/ODINN_prepro`.

## The Gungnir → Sleipnir handoff

This is the boundary between Python and Julia: Gungnir (Python) writes NetCDF files to disk; everything from `Sleipnir.initialize_glaciers()` onward is Julia.

Gungnir writes output under `~/.ODINN/ODINN_prepro/`. The root contains two JSON index files:

  - `rgi_paths.json` — maps each RGI ID to the relative path of its glacier directory
  - `rgi_names.json` — maps each RGI ID to the glacier name

Individual glacier directories live at `per_glacier/<region>/<subregion>/<RGI_ID>/`, for example `per_glacier/RGI60-07/RGI60-07.00/RGI60-07.00042/`. Each directory contains:

  - `gridded_data.nc` — static gridded attributes: DEM (`topo`), glacier mask, ice thickness from Farinotti 2019 (`consensus_ice_thickness`) and Millan 2022 (`millan_ice_thickness`), surface velocity fields (`millan_vx`, `millan_vy`), slope, aspect, and border distance.
  - `climate_historical_daily_W5E5.nc` — daily W5E5 climate forcing (temp, prcp) near the glacier centroid.
  - `climate_historical_monthly_ERA5.nc` or `climate_historical_daily_ERA5.nc` — ERA5 climate forcing at monthly or daily resolution depending on the mode used during preprocessing (see [Climate data sources](#climate-data-sources)).

On the Julia side, `Sleipnir.initialize_glaciers()` reads `rgi_paths.json` to locate each glacier, loads the NetCDF files as `RasterStack` objects via `Rasters.jl`, builds `Glacier2D` structs with a `Climate2D` attached, and downscales the climate time series to the glacier grid via `downscale_2D_climate!()`.

## How to add a new climate source or atmospheric variable

Climate retrieval is wired directly into the preprocessing loop in `gungnir/gungnir/preprocessing.py` — there is no plugin registry. The existing sources are the template: W5E5 via `process_w5e5_data` (from MBsandbox), and ERA5 via `ensure_era5_file_for_gdir` in `gungnir/gungnir/era5_climate.py`.

 1. To add a new climate source, create a downloader/processor that writes a per-glacier NetCDF file, following the ERA5 pattern in `era5_climate.py`.
 2. Call it for each glacier inside `preprocessing_glaciers` in `preprocessing.py`, next to the existing W5E5/ERA5 calls.
 3. Ensure the new variables are written to the output NetCDF file with field names that match the `Climate2Dstep` fields defined in `Sleipnir`.
 4. If new fields are needed in `Climate2Dstep`, add them in `Sleipnir/src/glaciers/climate/Climate2D.jl` and update `downscale_2D_climate!` accordingly.

Contributions to Gungnir are tracked in the [`Gungnir` GitHub repository](https://github.com/ODINN-SciML/Gungnir).
