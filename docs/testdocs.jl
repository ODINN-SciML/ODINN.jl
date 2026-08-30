# This file runs all the tutorials in order to detect potential bugs before
# Literate.jl is actually called and tries to generate the documentation

using Pkg

Pkg.activate(@__DIR__)
Pkg.resolve()

using Revise
using ODINN
using Test

@testset "Run all tutorials" begin
    @testset "Forward simulation" include(pkgdir(ODINN)*"/docs/src/forward_simulation.jl")
    @testset "Classical inversion" include(pkgdir(ODINN)*"/docs/src/classical_inversion.jl")
    @testset "Functional inversion" include(pkgdir(ODINN)*"/docs/src/functional_inversion.jl")
    @testset "Laws" include(pkgdir(ODINN)*"/docs/src/laws.jl")
    @testset "Laws VJPs" include(pkgdir(ODINN)*"/docs/src/vjp_laws.jl")
    @testset "Laws inputs" include(pkgdir(ODINN)*"/docs/src/input_laws.jl")
    @testset "Quick start" include(pkgdir(ODINN)*"/docs/src/quick_start.jl")
end
