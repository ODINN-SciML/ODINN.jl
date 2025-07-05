# This file runs all the tutorials in order to detect potential bugs before
# Literate.jl is actually called and tries to generate the documentation

using Pkg

# Change back to the `docs` directory if necessary
if basename(pwd()) != "docs"
    cd("docs")
end

Pkg.activate(".")

using ODINN
using Test

@testset "Run all tutorials" begin

@testset "Forward simulation" include("src/forward_simulation.jl")
@testset "Functional inversion" include("src/functional_inversion.jl")
@testset "Laws" include("src/laws.jl")

end
