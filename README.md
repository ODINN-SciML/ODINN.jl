# ODINN

[![Build Status](https://github.com/ODINN-SciML/ODINN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ODINN-SciML/ODINN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ODINN-SciML/ODINN.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ODINN-SciML/ODINN.jl)
[![CompatHelper](https://github.com/ODINN-SciML/ODINN.jl/actions/workflows/CompatHelper.yml/badge.svg)](https://github.com/ODINN-SciML/ODINN.jl/actions/workflows/CompatHelper.yml) 

[![docs](https://img.shields.io/badge/documentation-main-blue.svg)](https://odinn-sciml.github.io/ODINN.jl/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8033313.svg)](https://doi.org/10.5281/zenodo.8033313)

<img src="https://github.com/ODINN-SciML/ODINN.jl/blob/main/plots/ODINN_sticker_original.png?raw=true" width="250">

For a detailed description of the model and the application of Universal Differential Equations to glacier ice flow modelling, take a look at [our recent publication at Geoscientific Model Development](https://gmd.copernicus.org/articles/16/6671/2023/gmd-16-6671-2023.html). 

## About ODINN.jl

Global glacier evolution model using Universal Differential Equations to model and discover processes of climate-glacier interactions. 

`ODINN.jl` uses neural networks and differential equations in order to combine mechanistic models describing glacier physical processes (e.g. ice creep, basal sliding, surface mass balance) with machine learning. Neural networks are used to learn parts of the equations. ODINN uses the Open Global Glacier Model ([OGGM](oggm.org/), Maussion et al., 2019) through [Gungnir](https://github.com/ODINN-SciML/Gungnir) as a basic framework to retrieve all the topographical and climate data for the initial state of the simulations. Then, all the simulations and processing are performed in Julia, benefitting from its high performance and the SciML ecosystem. 

<center><img src="https://github.com/ODINN-SciML/odinn_toy/blob/main/plots/overview_figure.png" width="700"></center>

> **Overview of `ODINN.jl`’s workflow to perform functional inversions of glacier physical processes using Universal Differential Equations**. The parameters ($θ$) of a function determining a given physical process ($D_θ$), expressed by a neural network $NN_θ$, are optimized in order to minimize a loss function. In this example, the physical to be inferred law was constrained only by climate data, but any other proxies of interest can be used to design it. The climate data, and therefore the glacier mass balance, are downscaled (i.e. it depends on $S$), with $S$ being updated by the solver, thus dynamically updating the state of the simulation for a given timestep.

## Installing ODINN 

In order to install `ODINN` in a given environment, just do in the REPL:
```julia
julia> ] # enter Pkg mode
(@v1.10) pkg> activate MyEnvironment # or activate whatever path for the Julia environment
(MyEnvironment) pkg> add ODINN
```

### Using OGGM for the initial conditions of the training/simulations 

OGGM works as a front-end of ODINN, utilizing all its tools to retrieve RGI data, topographical data, climate data and other datasets from the OGGM shop. We use these data to specify the initial state of the simulations, and to retrieve the climate data to force the model. Everything related to the mass balance and ice flow dynamics models is written 100% in Julia. This allows us to run tests with this toy model for any glacier on Earth. In order to choose a glacier, you just need to specify the RGI ID, which you can find [here](https://www.glims.org/maps/glims). 

## How to use ODINN

ODINN's architecture makes it really straightforward to retrieve all the necessary glacier and climate data for both the initial conditions and the loss function of a problem. Here's a quick example based on a `FunctionalInversion` using Universal Differential Equations:

```julia
using ODINN

# Data are automatically downloaded, retrieve the local paths
rgi_paths = get_rgi_paths()

# We create the necessary parameters
params = Parameters(
	simulation = SimulationParameters(
		working_dir=working_dir,
		tspan=(2010.0, 2015.0),
		workers=5,
		rgi_paths=rgi_paths
		),
	hyper = Hyperparameters(
		batch_size=4,
		epochs=10,
		optimizer=ODINN.ADAM(0.01)
		),
	UDE = UDEparameters(
		target = :A
		)
	)

# We define which glacier RGI IDs we want to work with
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351"]

# We specify a model based on an iceflow model, a mass balance model and a machine learning model

nn_model = NeuralNetwork(params)
model = Model(iceflow = SIA2Dmodel(params; A=LawA(nn_model, params)),
	      mass_balance = mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
	      regressors = (; A=nn_model))

# We initialize the glaciers with all the necessary data
glaciers = initialize_glaciers(rgi_ids, params)

# We specify the type of simulation we want to perform
functional_inversion = FunctionalInversion(model, glaciers, params)

# And finally, we just run! the simulation
run!(functional_inversion)

```

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
