# Code style and recommendations

In this page we compile the general recommendations in terms of best coding practices, conventions and style for developers contributing to the ODINN ecosystem.

## Running the documentation locally

This section contains the instructions to run the documentation locally.

To generate the documentation on your local computer, in the `docs/` folder run:
```julia
include("makeLocal.jl")
```

Then in another REPL, in the `docs/` folder, activate the docs environment and run the server:
```julia
using Pkg
Pkg.activate()
using LiveServer
serve()
```

This will print a localhost URL that you can open in your browser to visualize the documentation.
