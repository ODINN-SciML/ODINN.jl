# ODINN

[![Build Status](https://github.com/ODINN-SciML/ODINN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ODINN-SciML/ODINN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ODINN-SciML/ODINN.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ODINN-SciML/ODINN.jl)
[![CompatHelper](https://github.com/ODINN-SciML/ODINN.jl/actions/workflows/CompatHelper.yml/badge.svg)](https://github.com/ODINN-SciML/ODINN.jl/actions/workflows/CompatHelper.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8033313.svg)](https://doi.org/10.5281/zenodo.8033313)

<img src="https://github.com/ODINN-SciML/ODINN.jl/blob/new_API/plots/ODINN_sticker_original.png?raw=true" width="250">

### ⚠️ New publication available! ⚠️

For a detailed description of the model and the application of Universal Differential Equations to glacier ice flow modelling, take a look at [our recent publication at Geoscientific Model Development](https://gmd.copernicus.org/articles/16/6671/2023/gmd-16-6671-2023.html). 

## About ODINN.jl

Global glacier evolution model using Universal Differential Equations to model and discover processes of climate-glacier interactions. 

`ODINN.jl` uses neural networks and differential equations in order to combine mechanistic models describing glacier physical processes (e.g. ice creep, basal sliding, surface mass balance) with machine learning. Neural networks are used to learn parts of the equations, which then can be interpreted in a mathematical form (e.g. using SINDy) in order to update the original equation from the process. ODINN uses the Open Global Glacier Model ([OGGM](oggm.org/), Maussion et al., 2019) as a basic framework to retrieve all the topographical and climate data for the initial state of the simulations. This is done calling Python from Julia using PyCall. Then, all the simulations and processing are performed in Julia, benefitting from its high performance and the SciML ecosystem. 

<center><img src="https://github.com/ODINN-SciML/odinn_toy/blob/main/plots/overview_figure.png" width="700"></center>

> **Overview of `ODINN.jl`’s workflow to perform functional inversions of glacier physical processes using Universal Differential Equations**. The parameters ($θ$) of a function determining a given physical process ($D_θ$), expressed by a neural network $NN_θ$, are optimized in order to minimize a loss function. In this example, the physical to be inferred law was constrained only by climate data, but any other proxies of interest can be used to design it. The climate data, and therefore the glacier mass balance, are downscaled (i.e. it depends on $S$), with $S$ being updated by the solver, thus dynamically updating the state of the simulation for a given timestep.

## Installing ODINN 

In order to install `ODINN` in a given environment, just do in the REPL:
```julia
julia> ] # enter Pkg mode
(@v1.10) pkg> activate MyEnvironment # or activate whatever path for the Julia environment
(MyEnvironment) pkg> add ODINN
```

### Installing ODINN's Python dependencies

ODINN depends on some Python packages, mainly [OGGM](https://github.com/OGGM/oggm) and [xarray](https://github.com/pydata/xarray). In order to install the necessary Python dependencies in an easy manner, we are providing a Python environment (`oggm_env`) in `environment.yml`. To install and activate the environment, we recommend using [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html):

```
micromamba create -f environment.yml
micromamba activate oggm_env
```

In order to call OGGM in Python from Julia, we use [PyCall.jl](https://github.com/JuliaPy/PyCall.jl). PyCall hooks on the Python installation and uses Python in a totally seamless way from Julia. 

The path to this conda environment needs to be specified in the `ENV["PYTHON"]` variable in Julia, for PyCall to find it. This configuration is very easy to implement, it just requires providing the Python path to PyCall and building it:

```julia
julia # start Julia session

julia> ENV["PYTHON"] = read(`which python`, String)[1:end-1] # trim backspace
julia> import Pkg; Pkg.build("PyCall")
julia> exit()

# Now you can run your code using ODINN in a new Julia session; e.g.:
using ODINN
```

So now you can start working with ODINN with PyCall correctly configured. These configuration step only needs to be done the first time, so from now on ODINN should be able to correctly find your Python libraries. If you ever want to change your conda environment, you would just need to repeat the steps above. 

### Using OGGM for the initial conditions of the training/simulations 

ODINN works as a back-end of OGGM, utilizing all its tools to retrieve RGI data, topographical data, climate data and other datasets from the OGGM shop. We use these data to specify the initial state of the simulations, and to retrieve the climate data to force the model. Everything related to the mass balance and ice flow dynamics models is written 100% in Julia. This allows us to run tests with this toy model for any glacier on Earth. In order to choose a glacier, you just need to specify the RGI ID, which you can find [here](https://www.glims.org/maps/glims). 

## How to use ODINN

ODINN's architecture makes it really straightforward to retrieve all the necessary glacier and climate data for both the initial conditions and the loss function of a problem. Here's a quick example based on a `FunctionalInversion` using Universal Differential Equations:

```julia
using ODINN

# Data are automatically downloaded, retrieve the local paths
rgi_paths = get_rgi_paths()

# We create the necessary parameters
params = Parameters(simulation = SimulationParameters(working_dir=working_dir,
									tspan=(2010.0, 2015.0),
									workers=5,
									rgi_paths=rgi_paths),
		    hyper = Hyperparameters(batch_size=4,
					    epochs=10,
					    optimizer=ODINN.ADAM(0.01)),
		    UDE = UDEparameters(target = "A")
		   )

# We define which glacier RGI IDs we want to work with
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351"]

# We specify a model based on an iceflow model, a mass balance model and a machine learning model
model = Model(iceflow = SIA2Dmodel(params),
	      mass_balance = mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
	      machine_learning = NN(params))

# We initialize the glaciers with all the necessary data
glaciers = initialize_glaciers(rgi_ids, params)

# We specify the type of simulation we want to perform
functional_inversion = FunctionalInversion(model, glaciers, params)

# And finally, we just run! the simulation
run!(functional_inversion)

````	

## How to cite 📖

If you want to cite this work, please use this BibTex citation from [our latest paper](https://gmd.copernicus.org/articles/16/6671/2023/gmd-16-6671-2023.html):
```
@article{bolibar_sapienza_universal_2023,
	title = {Universal differential equations for glacier ice flow modelling},
	author = {Bolibar, J. and Sapienza, F. and Maussion, F. and Lguensat, R. and Wouters, B. and P\'erez, F.},
	journal = {Geoscientific Model Development},
	volume = {16},
	year = {2023},
	number = {22},
	pages = {6671--6687},
	url = {https://gmd.copernicus.org/articles/16/6671/2023/},
	doi = {10.5194/gmd-16-6671-2023}
}
```
