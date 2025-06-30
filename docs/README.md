# Instructions to develop the documentation on local

To generate the documentation locally run:
```julia
shell> cd docs/
julia> include("makeLocal.jl")
```

Then in another REPL, activate the docs environment and run the server:
```julia
shell> cd docs/
pkg> activate .
julia> using LiveServer
julia> serve()
```

This will print a localhost link that you can open in your browser to visualize the documentation.
