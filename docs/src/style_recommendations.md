# Code style and recommendations

In this page we compile the general recommendations in terms of best coding practices, conventions and style for developers contributing to the ODINN ecosystem.

## Running the documentation locally

This section contains the instructions to run the documentation locally.

To generate the documentation on your local computer, in the `docs/` folder run:
```julia
include("make.jl")
```

Then in another REPL, in the `docs/` folder, activate the docs environment and run the server:
```julia
using Pkg
Pkg.activate()
using LiveServer
serve()
```

This will print a localhost URL that you can open in your browser to visualize the documentation.

!!! warning "Multiprocessing in the documentation"

    In order to use multiprocessing in the documentation, we set up a specific number of workers in the Julia session in the [`documentation.yml`](https://github.com/ODINN-SciML/ODINN.jl/blob/main/.github/workflows/documentation.yml) file. It is imperative that the number of workers set there matches the ones set in the Julia code run in the documentation. By default, we have set them to `-p 3` in `documentation.yml`, meaning that 3 workers will be added on top of the head one. This will match the default number of workers in `SimulationParameters`, but if you manually specify them, make sure to set them to 4 (the number of parameters in ODINN DOES include the head worker). This is often a source of confusion, so refrain from playing with the number of workers in the documentation. 