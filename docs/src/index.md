# ODINN.jl documentation

Welcome to the `ODINN.jl` documentation, a large-scale scientific machine learning (SciML) glacier model, leveraging differentiable programming in Julia. This documentation provides the necessary information to understand the ecosystem built around `ODINN.jl`, its APIs, workflows, and some examples of the main usage types of the model(s).

## Quick install

`ODINN.jl` is a registered [Julia](https://julialang.org/) package, so installing it is as easy as:

```julia
julia> using Pkg

julia> Pkg.add("ODINN")

```

!!! info
    
    `ODINN.jl` and the rest of the ODINN ecosystem packages require Julia v1.10 or later, which are the ones used and tested during continuous integration (CI).

## Documentation overview

The documentation, which you can navigate through the left-hand-side panel, is structured in the following way:

  - *Quick start*: Provides a straight-to-the-point example of a simple use-case of `ODINN.jl` to get an idea of the basic interface and API.
  - *Tutorials*: They provide clear examples of the main types of simulations and workflows that you can work with in `ODINN.jl`.
  - *How to use ODINN*: Explains the basic building blocks (i.e. types) used in `ODINN.jl` simulations, and how they are assembled together.
  - *Inversions*: Addresses more advanced questions related to inverse modelling, computing gradients, optimization, and machine learning.
  - *Community*: Explains the aspects related to how to contribute to the model(s), code style, building the documentation in local, and the code of conduct.
  - *Ongoing changes and future plans*: Self-explanatory.
  - *References*: List of scientific papers used in this documentation.

## Vision

Rather than focusing on global-scale simulations and sea-level rise contributions, `ODINN.jl` has, for now, a regional and catchment-scale focus, aiming to exploit the latest remote sensing and in situ observations to capture missing or subgrid glacier processes. In order to do so, `ODINN.jl` leverages Universal Differential Equations (UDEs) combining PDEs describing ice flow dynamics with data-driven regressors, such as neural networks [rackauckas_universal_2020](@cite). For this, `ODINN.jl` relies heavily on the [SciML](https://sciml.ai/) Julia ecosystem and the native automatic differentiation (AD) support. Therefore `ODINN.jl` has a two-fold goal:

  - To advance the application of SciML and differentiable programming for large-scale geoscientific modelling.
  - To advance the inference of new parametrizations to characterize key missing or subgrid processes of glaciers to improve large-scale glacier simulations.

## Architecture

`ODINN.jl` is a modular model, split into multiple packages, each one handling a specific task:

  - [`ODINN.j`](https://github.com/ODINN-SciML/ODINN.jl) is the high-level interface to the whole ODINN ecosystem, containing the SciML functionalities related to automatic differentiation and sensitivity of hybrid models, mixing differential equations and data-driven regressors.
  - [`Huginn.jl`](https://github.com/ODINN-SciML/Huginn.jl) is the ice flow dynamics module of ODINN. It contains all the information regarding glacier ice flow models, including the numerical methods to solve the PDEs using [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl).
  - [`Muninn.jl`](https://github.com/ODINN-SciML/Muninn.jl) is the surface mass balance module of ODINN. It  contains all the information regarding glacier interactions with the atmosphere (i.e. surface mass balance processes). For now we support simple temperature-index models, but soon we are planning to incorporate machine learning models coming from the [`MassBalanceMachine`](https://github.com/ODINN-SciML/MassBalanceMachine).
  - [`Sleipnir.jl`](https://github.com/ODINN-SciML/Sleipnir.jl) is the core package of ODINN, holding all the basic data structures and functions, common to the whole ecosystem. It directly reads the files provided by [`Gungnir`](https://github.com/ODINN-SciML/Gungnir).
  - [`Gungnir`](https://github.com/ODINN-SciML/Gungnir) is a Python package, using [OGGM](https://github.com/OGGM/oggm) to retrieve all the necessary files (i.e. rasters and climate data) for the initial conditions and simulations in all the ODINN ecosystem. The user has the possibility to either store those files locally, or to use the ones we provide in a server. This is work in progress, so we will progressively cover more and more glaciers and regions in the near future.

```@raw html
<img src="./assets/odinn_ecosystem_v4.png" alt="ODINN ecosystem overview" width="700"/>
```

## Developers

`ODINN.jl` is being developed by [Jordi Bolibar](https://jordibolibar.org/) (*CNRS, IGE*), [Facundo Sapienza](https://facusapienza.org/) (*Stanford University*) and [Alban Gossard](https://albangossard.github.io/) (*Université Grenoble Alpes, IGE*).

Past developers include Mathieu Le Séac'h (*Université Grenoble Alpes, IGE*), Lucille Gimenes (*Université Grenoble Alpes, IGE*), and Vivek Gajadhar (*TU Delft*).

## Citing

If you use `ODINN.jl` for research, teaching or other activities, please use the following citation from [our latest publication](https://gmd.copernicus.org/articles/16/6671/2023/gmd-16-6671-2023.html):

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

## Funding

The ODINN project has been funded by an ANR TRACCS early career call, an IRGA fellowship from the Multidisciplinary Institute on Artificial Intelligence (Grenoble, France), the Nederlandse Organisatie voor Wetenschappelijk Onderzoek, Stichting voor de Technische Wetenschappen (Vidi grant 016.Vidi.171.063), the National Science Foundation (EarthCube programme under
awards 1928406 and 1928374) and a TU Delft Climate Action grant.
