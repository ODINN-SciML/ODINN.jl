# Gungnir

[`Gungnir`](https://github.com/ODINN-SciML/Gungnir) is a Python preprocessing pipeline that produces the glacier and climate data files consumed by the Julia ODINN ecosystem. It uses [OGGM](https://github.com/OGGM/oggm) to retrieve glacier geometry (DEMs, ice thickness, outlines) from the Randolph Glacier Inventory (RGI), and downloads climate reanalyses (W5E5 or ERA5) to force the mass balance models. The output is written to `~/.ODINN/ODINN_prepro/` as HDF5/JLD2 files, which `Sleipnir.initialize_glaciers()` reads at simulation time.

Unlike the Julia packages in the ecosystem, `Gungnir` is **not** a Julia package — it runs in a Python environment and has no Julia API. It sits at the bottom of the dependency hierarchy: Gungnir → Sleipnir → Muninn/Huginn → ODINN.

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

If you only need the `gungnir` Python module (without the full notebook environment):

```bash
pip install gungnir
# or in developer mode:
pip install -e gungnir
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
RGI60-11.00897
RGI60-11.01450
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

## The Gungnir → Sleipnir handoff

Gungnir writes per-glacier directories under `~/.ODINN/ODINN_prepro/<RGI_ID>/`. Each directory contains:

  - `glacier_stats.h5` — static geometry: bed, surface, thickness, coordinates
  - `climate_<source>.h5` — gridded climate time series at glacier resolution

`Sleipnir.initialize_glaciers()` reads these files, builds `Glacier2D` objects with the appropriate `Climate2D` attached, and downscales climate to the glacier grid via `downscale_2D_climate!()`. From that point forward, all computation is in Julia.

## How to add a new climate source or atmospheric variable

 1. Add a new downloader function in `gungnir/climate/` following the existing W5E5 or ERA5 pattern.
 2. Register the new source name in the `CLIMATE_SOURCES` registry in `gungnir/config.py`.
 3. Ensure the new variables are written to the output HDF5 file with field names that match the `Climate2Dstep` fields defined in `Sleipnir`.
 4. If new fields are needed in `Climate2Dstep`, add them in `Sleipnir/src/glaciers/climate/Climate2D.jl` and update `downscale_2D_climate!` accordingly.

Contributions to Gungnir are tracked in the [`Gungnir` GitHub repository](https://github.com/ODINN-SciML/Gungnir).
