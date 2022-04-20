# ODINN

<!---
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JordiBolibar.github.io/ODINN.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JordiBolibar.github.io/ODINN.jl/dev)
[![Build Status](https://github.com/JordiBolibar/ODINN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JordiBolibar/ODINN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://travis-ci.com/JordiBolibar/ODINN.jl.svg?branch=main)](https://travis-ci.com/JordiBolibar/ODINN.jl)
[![Coverage](https://codecov.io/gh/JordiBolibar/ODINN.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JordiBolibar/ODINN.jl)
-->

<img src="https://github.com/ODINN-SciML/odinn_toy/blob/main/plots/ODINN_logo_final.png" width="250">

## OGGM (Open Global Glacier Model) + DIfferential equation Neural Networks

Global glacier model using Neural Differential Equations to model and discover processes of climate-glacier interactions.  

It uses neural networks and differential equations in order to combine mechanistic models describing glaciological processes (e.g. enhanced temperature-index model or the Shallow Ice Approximation) with machine learning. Neural networks are used to learn parts of the equations, which then can be interpreted in a mathematical form (e.g. using SINDy) in order to update the original equation from the process. ODINN uses the Open Global Glacier Model (OGGM, Maussion et al., 2019) as a basic framework to retrieve all the topographical and climate data for the initial state of the simulations. This is done calling Python from Julia using PyCall. Then, all the simulations and processing are performed in Julia, benefitting from the high performance and the SciML ecosystem. 

## Installing ODINN

In order to install `ODINN` in a given environment, just do in the REPL:
```julia
julia> ] # enter Pkg mode
(@v1.7) pkg> activate MyEnvironment # or activate whatever path for the Julia environment
(MyEnvironment) pkg> add ODINN
```

## ODINN initialization: integration with OGGM and multiprocessing

In order to call OGGM in Python from Julia, a Python installation is needed, which then can be used in Julia using [PyCall](https://github.com/JuliaPy/PyCall.jl). We recommend splitting the Julia (i.e. ODINN) and Python (i.e. OGGM) files in separate folders, which we chose to name `Julia` and `Python`, both placed at root level. As indicated in the [OGGM documentation](https://docs.oggm.org/en/stable/installing-oggm.html), when installing OGGM it is best to create a new dedicated conda environment for it (e.g. `oggm_env`). In the same environment, install also the [OGGM Mass-Balance sandbox](https://github.com/OGGM/massbalance-sandbox) following the instructions in the repository.

The path to this conda environment needs to be specified in the `ENV["PYTHON"]` variable in Julia, for PyCall to find it. This configuration is very easy to implement, it just requires activating the conda environment before the first time you run ODINN in your machine. In the terminal (not in a Julia session), run:

```
conda activate oggm_env # replace `oggm_env` with whatever conda environment where you have installed OGGM and the MBSandbox
```

Then, you need to configure PyCall to use the Python path for that conda environment:

```julia
julia # start Julia session

julia> global ENV["PYTHON"] = read(`which python`, String)[1:end-1] # trim backspace
julia> import Pkg; Pkg.build("PyCall")
julia> exit()

# Now you can run your code using ODINN in a new Julia session; e.g.:
using ODINN
```

So now you can start working with ODINN with PyCall correctly configured. These configuration step only needs to be done the first time, so from now on ODINN should be able to correctly find your Python libraries. If you ever want to change your conda environment, you would just need to repeat the steps above. The next step is to start a new Julia session and import `ODINN` (or just run your script which uses ODINN, e.g. `toy_model.jl`). If you want to run ODINN using multiprocessing you can enable it using the following command in Julia:

```julia
processes = 16
ODINN.enable_multiprocessing(processes)
```

From this point, it is possible to use ODINN with multiprocessing and to run Python from Julia running the different commands available in the PyCall documentation. In order to get a better idea on how this works, we recommend checking the toy model example [toy_model.jl](https://github.com/ODINN-SciML/ODINN/blob/main/src/scripts/toy_model.jl). 

### Using OGGM for the initial conditions of the training/simulations

ODINN works as a back-end of OGGM, utilizing all its tools to retrieve RGI data, topographical data, climate data and other datasets from the OGGM shop. We use these data to specify the initial state of the simulations, and to retrieve the climate data to force the model. Everything related to the mass balance and ice flow dynamics models is written 100% in Julia. This allows us to run tests with this toy model for any glacier on Earth. In order to choose a glacier, you just need to specify the RGI ID, which you can find [here](https://www.glims.org/maps/glims). 

## Running the toy model

A demostration with a toy model is showcased in `src/scripts/toy_model.jl`. The `Manifest.toml` and `Project.toml` include all the required dependencies. If you are running this code from zero, you may need to install the libraries using `Pkg.instantiate()`. In case you want to include this package to the project manifest, you can also use `Pkg.resolve()` before instantiating the project. You can replace the preamble in `src/scripts/toy_model.jl` to 

```julia
import Pkg
Pkg.activate(dirname(Base.current_project()))
Pkg.precompile()
Pkg.instantiate()
```
