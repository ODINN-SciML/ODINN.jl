# Code style and recommendations

In this page we compile the general recommendations in terms of best coding practices, conventions and style for developers contributing to the ODINN ecosystem.

## How to contribute

We welcome all types of contributions to the ODINN ecosystem! In order to contribute, please make a fork of the repository you would like to contribute to, and then submit a pull request (PR). We will review it and provide feedback. 

If you are looking for ideas, you can check the [opened issues](https://github.com/ODINN-SciML/ODINN.jl/issues) in our repositories. 

!!! tip 
    Please submit your PRs to the `dev` branch. `main` is only updated from time to time when enough meaningful commits are available to perform a new release.



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
