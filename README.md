# ODINN

<img src="https://github.com/ODINN-SciML/odinn_toy/blob/main/plots/ODINN_logo_final.png" width="250">

## OGGM (Open Global Glacier Model) + DIfferential equation Neural Networks

Global glacier model using Neural Differential Equations to model and discover climate-glacier interactions.  

It uses neural networks and differential equations in order to combine mechanistic models describing glaciological processes (e.g. enhanced temperature-index model or the Shallow Ice Approximation) with machine learning. Neural networks are used to learn parts of the equations, which then can be interpreted in a mathematical form in order to update the original equation from the process. The goal of this toy model is to provide a proof of concept for the discovery and modelling framework before scaling it up with OGGM. 

## Running the toy model

A demostration of our method is included in `scripts/toy_model/ODINN.jl`. The `Manifest.toml` and `Project.toml` include all the required dependencies. If you are running this code from zero, you may need to install the libraries using `Pkg.instantiate()`. In case you want to include this package to the project manifest, you can also use `Pkg.resolve()` before instantiating the project. 

## Integration with OGGM

In order to call OGGM in Python from Julia, a Python installation is needed, which then can be used in Julia using [PyCall](https://github.com/JuliaPy/PyCall.jl). We recommend splitting the Julia (i.e. ODINN) and Python (i.e. OGGM) files in separate folders, which we chose to name `Julia` and `Python`, both placed at root level. As indicated in the [OGGM documentation](https://docs.oggm.org/en/stable/installing-oggm.html), when installing OGGM it is best to create a new dedicated conda environment for it (e.g. `oggm_env`). The path to this conda environment needs to be specified in the `ENV["PYTHON"]` variable in Julia, for PyCall to find it. After specifying the path, it is necessary to build PyCall using `Pkg.build("PyCall")`. 
In the same environment, install also the [OGGM Mass-Balance sandbox](https://github.com/OGGM/massbalance-sandbox) following the instructions in the repository.
From this point, it is possible to run Python from Julia running the different commands available in the PyCall documentation. In order to get a better idea on how this works, we recommend checking the toy model interface [ODINN.jl](https://github.com/ODINN-SciML/ODINN_toy/blob/main/scripts/toy_model/ODINN.jl). 

### Using OGGM for the initial conditions of the training/simulations

ODINN works as a back-end of OGGM, utilizing all its tools to retrieve RGI data, topographical data, climate data and other datasets from the OGGM shop. We use these data to specify the initial state of the simulations, and to retrieve the climate data to force the model. Everything related to the mass balance and ice flow dynamics models is written 100% in Julia. This allows us to run tests with this toy model for any glacier on Earth. In order to choose a glacier, you just need to specify the RGI ID, which you can find [here](https://www.glims.org/maps/glims). 
